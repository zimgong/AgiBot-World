import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import draccus
import json_numpy
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image
from transformers import AutoTokenizer

from go1.internvl.model.go1 import GO1Model, GO1ModelConfig
from go1.internvl.train.constants import IMG_END_TOKEN
from go1.internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internvl2_5

json_numpy.patch()


def _to_pil_image(x) -> Image.Image:
    """
    Accepts numpy/torch with shapes:
      - [C, H, W] with C in {1,3,4}
      - [H, W, C] with C in {1,3,4}
      - [1, C, H, W] or [1, H, W, C]  (squeezed)
      - [H, W] (grayscale)
    Returns a PIL Image. Converts float [0,1] to uint8.
    """
    # Fast path if already PIL
    if isinstance(x, Image.Image):
        return x
    
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)

    # Drop leading batch dim of size 1, if present
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # CHW -> HWC if channels-first
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # HWC

    # If HWC with single channel, squeeze to HW
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]

    # Validate final geometry
    if not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))):
        raise ValueError(
            f"Unsupported image shape {arr.shape}. Expected HW, HWC (C=3/4), "
            f"or CHW with C=1/3/4 (optionally batched with leading size 1)."
        )

    # Convert dtype: float [0,1] -> uint8
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)

    return Image.fromarray(arr)

def _slice_nd(v, i):
    # Only slice if there is a batch dimension > 1
    return v[i] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] > 1 else v


def normalize(data, stats):
    return (data - stats["mean"]) / (stats["std"] + 1e-6)


def unnormalize(data, stats):
    return data * stats["std"] + stats["mean"]


def get_stats_tensor(stats_json, space_repack):
    stats_tensor = {}
    for name in ["state", "action"]:
        # Use the mapped key from space_repack to access stats_json
        mapped_key = space_repack[name]
        stats_tensor[space_repack[name]] = {}
        for key in ["mean", "std"]:
            stats_tensor[space_repack[name]][key] = torch.from_numpy(np.array(stats_json[mapped_key][key]))
    return stats_tensor


def multi_image_get_item(
    raw_target: Dict[str, Any],
    img_transform,
    text_tokenizer,
    num_image_token,
    cam_keys: List[str] = [
        "cam_head_color",
        "cam_hand_right_color",
        "cam_hand_left_color",
    ],
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=6,
    image_size=448,
):
    images, num_tiles = [], []
    num_image = 0
    for cam_key in cam_keys:
        if cam_key in raw_target:
            num_image += 1
            if dynamic_image_size:
                image = dynamic_preprocess(
                    raw_target[cam_key],
                    min_num=min_dynamic_patch,
                    max_num=max_dynamic_patch,
                    image_size=image_size,
                    use_thumbnail=use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:
                images.append(raw_target[cam_key])
                num_tiles.append(1)

    pixel_values = [img_transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    num_patches = pixel_values.size(0)

    # Preprocess the conversations and generate the return dictionary
    num_image_tokens = [num_image_token * num_tile for num_tile in num_tiles]
    ntp_target = raw_target.get("ntp_target", "")
    conversation = [
        {"from": "human", "value": f"{'<image>'*num_image}{raw_target['final_prompt']}"},
        {"from": "gpt", "value": ntp_target},
    ]
    ret = preprocess_internvl2_5(
        "internvl2_5",
        [conversation],
        text_tokenizer,
        num_image_tokens,
        num_image=num_image,
        group_by_length=True,
    )

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    image_end_token_id = text_tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    assert (ret["input_ids"][0] == image_end_token_id).sum() == num_image, "image tokens are truncated"

    # Create the final return dictionary
    final_ret = dict(
        input_ids=ret["input_ids"][0],
        labels=ret["labels"][0],
        attention_mask=ret["attention_mask"][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
    )
    return final_ret


class GO1Infer:
    def __init__(
        self,
        model_path: Union[str, Path],
        data_stats_path: Union[str, Path] = None,
        cfg_path: Union[str, Path] = None,
        batch_size: int = 1,
    ) -> Path:
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        # Enable performance optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.config = GO1ModelConfig.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=False,
        )
        self.image_size = self.config.force_image_size
        self.num_image_token: int = int(
            (self.image_size // self.config.vision_config.patch_size) ** 2 * (self.config.downsample_ratio**2)
        )
        self.dynamic_image_size = self.config.dynamic_image_size

        self.go1 = GO1Model.from_pretrained(model_path, config=self.config)
        self.go1.to(torch.bfloat16).to(self.device)

        self.img_transform = build_transform(
            is_train=False, input_size=self.image_size, pad2square=self.config.pad2square
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=False
        )

        self.norm = getattr(self.config, "norm", False)

        if cfg_path is not None:
            from go1.internvl.train.go1_train import get_config_args
            cfg, dataset_args, model_args, training_args, space_args = get_config_args(cfg_path)
            self.space_repack = space_args.space_repack
        else:
            self.space_repack = {"state": "state", "action": "action"}

        if self.norm:
            assert data_stats_path is not None, "data_stats_path must be provided when norm is True"
            with open(data_stats_path, "rb") as f:
                self.data_stats = get_stats_tensor(json.load(f), self.space_repack)
        
        self.batch_size = batch_size
        if self.batch_size > 1:
            print(f"GO1Infer initialized in BATCHED mode with batch_size={self.batch_size}")
        
        # Action queue for sequential action execution
        self.action_queue = deque()
        self.current_batch_id = None

    def predict_action(self, inputs: Dict[str, Any]) -> str:
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = inputs["position_ids"]
        image_flags = inputs["image_flags"]
        ctrl_freqs = inputs["ctrl_freqs"]

        state = inputs[self.space_repack["state"]]
        # Normalize state if needed
        if self.norm:
            state = normalize(state, self.data_stats[self.space_repack["state"]])

        start_time = time.time()
        device = self.device
        with torch.no_grad():
            action = self.go1(
                pixel_values=pixel_values.to(dtype=torch.bfloat16, device=device),
                input_ids=input_ids.to(device).unsqueeze(0),
                attention_mask=attention_mask.to(device).unsqueeze(0),
                position_ids=position_ids.to(device).unsqueeze(0),
                image_flags=image_flags.to(device),
                state=state.to(dtype=torch.bfloat16, device=device).unsqueeze(0),
                ctrl_freqs=ctrl_freqs.to(dtype=torch.bfloat16, device=device).unsqueeze(0),
            )
        print(f"Model inference time: {(time.time() - start_time)*1000:.3f} ms")
        outputs = action[1][0].float().cpu()

        # Unnormalize action if needed
        if self.norm:
            outputs = unnormalize(outputs, self.data_stats[self.space_repack["action"]])

        outputs = outputs.numpy()

        return outputs

    def _generate_action_sequence(self, payload: Dict[str, Any]) -> np.ndarray:
        """Generate a full action sequence and store in queue"""
        if self.batch_size > 1:
            batch = self.build_batched_samples(payload)
            action_sequence = self.predict_action_batched(batch)  # [B, Seq, Dim]
        else:
            action_sequence = self._inference_single(payload)  # [Seq, Dim]
            action_sequence = action_sequence.unsqueeze(0)  # [1, Seq, Dim]
        
        return action_sequence

    def _populate_action_queue(self, action_sequence: np.ndarray):
        """Populate the action queue with individual actions from the sequence"""
        self.action_queue.clear()
        B, Seq, Dim = action_sequence.shape
        
        # Store each timestep action for each batch item
        for t in range(Seq):
            action = action_sequence[:, t, :]  # [Dim]
            self.action_queue.append(action)
        
        print(f"Populated action queue with {len(self.action_queue)} actions (B={B}, Seq={Seq})")

    def _get_next_action(self) -> np.ndarray:
        """Get the next action from the queue"""
        if not self.action_queue:
            raise ValueError("Action queue is empty. Call with force_reinfer=True to regenerate.")
        
        action = self.action_queue.popleft()
        return action

    def _build_single_sample_inputs(self, sample_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build inputs for a single sample using existing multi_image_get_item logic"""
        # Aliases → PIL
        if "top" in sample_payload:
            sample_payload["cam_head_color"] = _to_pil_image(sample_payload["top"])
        if "right" in sample_payload:
            sample_payload["cam_hand_right_color"] = _to_pil_image(sample_payload["right"])
        if "left" in sample_payload:
            sample_payload["cam_hand_left_color"] = _to_pil_image(sample_payload["left"])

        for k in ['cam_hand_right_color', 'cam_hand_left_color', 'cam_head_color']:
            if k in self.space_repack and self.space_repack[k] in sample_payload:
                sample_payload[k] = _to_pil_image(sample_payload[self.space_repack[k]])

        prompt = sample_payload["instruction"]
        sample_payload["final_prompt"] = f"What action should the robot take to {prompt}?"

        ret = multi_image_get_item(
            raw_target=sample_payload,
            img_transform=self.img_transform,
            text_tokenizer=self.text_tokenizer,
            num_image_token=self.num_image_token,
            dynamic_image_size=self.dynamic_image_size,
            use_thumbnail=self.config.use_thumbnail,
            min_dynamic_patch=self.config.min_dynamic_patch,
            max_dynamic_patch=self.config.max_dynamic_patch,
            image_size=self.image_size,
        )

        # State / ctrl_freqs
        ret["state"] = torch.from_numpy(np.array(sample_payload[self.space_repack["state"]]).copy())
        ret["ctrl_freqs"] = torch.from_numpy(np.array(sample_payload["ctrl_freqs"]).copy())
        return ret

    def build_batched_samples(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a true batched batch, padding text to max len, stacking states, and
        keeping pixel patches in lists to avoid padding cost.
        Returns:
          dict with keys:
            input_ids [B, L], attention_mask [B, L], position_ids [B, L]
            pixel_values_list: List[Tensor[Ni,3,H,W]]
            image_flags_list:  List[Tensor[Ni]]
            state [B, S], ctrl_freqs [B, 1]
        """
        B = self.batch_size
        samples = []

        # Pre-read shared fields to broadcast
        global_instruction = payload["instruction"]
        # ctrl_freqs may be scalar, [1], or [B]; normalize to a 1D np array
        global_cf = payload["ctrl_freqs"]
        if global_cf is not None:
            global_cf = np.array(global_cf).reshape(-1)  # shape [n]
            # If it's scalar or len==1, we'll broadcast; if len==B, we'll per-sample index; else error
            if not (len(global_cf) in (1, B)):
                raise ValueError(f"ctrl_freqs length {len(global_cf)} inconsistent with batch size {B}")

        for i in range(B):
            sp: Dict[str, Any] = {}
            # Slice only truly-batched arrays/tensors
            for k, v in payload.items():
                if k in ("instruction", "ctrl_freqs"):  # handle below
                    continue
                if isinstance(v, (list, tuple)):
                    sp[k] = v
                else:
                    sp[k] = _slice_nd(v, i)

            # Broadcast instruction
            sp["instruction"] = global_instruction

            # Broadcast or index ctrl_freqs
            if global_cf is not None:
                cf_i = global_cf[0] if len(global_cf) == 1 else global_cf[i]
                sp["ctrl_freqs"] = np.array([cf_i], dtype=np.float32)  # enforce shape [1]

            # Ensure cams are PIL
            for cam in ("top","left","right","cam_head_color","cam_hand_right_color","cam_hand_left_color"):
                if cam in sp:
                    sp[cam] = _to_pil_image(sp[cam])

            samples.append(self._build_single_sample_inputs(sp))

        # Collate text by padding
        input_ids = rnn_utils.pad_sequence([s["input_ids"] for s in samples], batch_first=True,
                                           padding_value=self.text_tokenizer.pad_token_id)
        attention_mask = rnn_utils.pad_sequence([s["attention_mask"] for s in samples], batch_first=True, padding_value=0)
        position_ids = rnn_utils.pad_sequence([s["position_ids"] for s in samples], batch_first=True, padding_value=0)

        # Collate states / ctrl_freqs
        state = torch.stack([s["state"] for s in samples], dim=0)  # [B, S]
        ctrl_freqs = torch.stack([s["ctrl_freqs"].view(-1) for s in samples], dim=0)  # [B, 1]

        pixel_values_list = [s["pixel_values"] for s in samples]
        image_flags_list  = [s["image_flags"] for s in samples]

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values_list=pixel_values_list,
            image_flags_list=image_flags_list,
            state=state,
            ctrl_freqs=ctrl_freqs,
        )

    def predict_action_batched(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        True batched forward for GO1Model with concatenated vision:
        pixel_values_all: [K, 3, H, W]
        image_flags_all : [K, 1]  (ones -> valid patches)
        input_ids/attention_mask/position_ids: [B, L]
        state: [B, S], ctrl_freqs: [B, 1]
        Returns np.ndarray [B, A] or [B, H, A] depending on config.
        """
        device = self.device
        with torch.no_grad():
            # --- text ---
            input_ids      = batch["input_ids"].to(device)                      # [B, L], long
            attention_mask = batch["attention_mask"].to(device, dtype=torch.bool)  # [B, L], bool
            position_ids   = batch["position_ids"].to(device)                   # [B, L], long

            # --- vision: concat per-sample Ni -> K ---
            pv_list = batch["pixel_values_list"]   # List[Tensor[Ni,3,H,W]]
            fl_list = batch["image_flags_list"]    # List[Tensor[Ni]]

            K = sum(int(p.shape[0]) for p in pv_list) if pv_list else 0
            if K == 0:
                H = W = self.image_size
                pixel_values_all = torch.zeros((0, 3, H, W), dtype=torch.bfloat16, device=device)
                image_flags_all  = torch.zeros((0, 1), dtype=torch.long, device=device)
            else:
                pixel_values_all = torch.cat(
                    [p.to(device=device, dtype=torch.bfloat16) for p in pv_list], dim=0
                )  # [K,3,H,W]
                image_flags_all = torch.cat(
                    [f.to(device=device, dtype=torch.long).view(-1, 1) for f in fl_list], dim=0
                )  # [K,1]  -> common_process will squeeze(-1) to [K]

            # --- low-dim cond ---
            state      = batch["state"].to(device, dtype=torch.bfloat16)        # [B, S]
            ctrl_freqs = batch["ctrl_freqs"].to(device, dtype=torch.bfloat16)   # [B, 1]

            if self.norm:
                s_mean = self.data_stats[self.space_repack["state"]]["mean"].to(device, dtype=torch.bfloat16)
                s_std  = self.data_stats[self.space_repack["state"]]["std"].to(device, dtype=torch.bfloat16)
                state = (state - s_mean) / (s_std + 1e-6)
            
            # Model expects state as [B, 1, S], not [B, S]
            if state.ndim == 2:
                state = state.unsqueeze(1)  # [B, S] -> [B, 1, S]
            
            out = self.go1(
                pixel_values=pixel_values_all,    # [K,3,H,W]
                input_ids=input_ids,              # [B,L]
                attention_mask=attention_mask,    # [B,L] (bool)
                position_ids=position_ids,        # [B,L]
                image_flags=image_flags_all,      # [K,1]
                state=state,                      # [B,S]
                ctrl_freqs=ctrl_freqs,            # [B,1]
            )

            # Support both tuple and dataclass outputs
            if isinstance(out, tuple):
                logits = out[1]
            else:
                logits = out.action_logits

            logits = logits.float().cpu()  # [B,A] or [B,H,A] per config

            if self.norm:
                a_mean = self.data_stats[self.space_repack["action"]]["mean"].cpu()
                a_std  = self.data_stats[self.space_repack["action"]]["std"].cpu()
                logits = logits * a_std + a_mean

            return logits.numpy()



    def _inference_single(self, payload: Dict[str, Any]):
        """Single sample inference (original logic)"""
        if "top" in payload:
            payload["cam_head_color"] = _to_pil_image(payload["top"])
        if "right" in payload:
            payload["cam_hand_right_color"] = _to_pil_image(payload["right"])
        if "left" in payload:
            payload["cam_hand_left_color"] = _to_pil_image(payload["left"])

        for raw_images_keys in ['cam_hand_right_color', 'cam_hand_left_color', 'cam_head_color']:
            if raw_images_keys in self.space_repack and self.space_repack[raw_images_keys] in payload:
                payload[raw_images_keys] = _to_pil_image(payload[self.space_repack[raw_images_keys]])

        prompt = payload["instruction"]
        payload["final_prompt"] = f"What action should the robot take to {prompt}?"

        inputs = multi_image_get_item(
            raw_target=payload,
            img_transform=self.img_transform,
            text_tokenizer=self.text_tokenizer,
            num_image_token=self.num_image_token,
            dynamic_image_size=self.dynamic_image_size,
            use_thumbnail=self.config.use_thumbnail,
            min_dynamic_patch=self.config.min_dynamic_patch,
            max_dynamic_patch=self.config.max_dynamic_patch,
            image_size=self.image_size,
        )

        inputs["state"] = torch.from_numpy(payload["state"].copy())
        inputs["ctrl_freqs"] = torch.from_numpy(payload["ctrl_freqs"].copy())

        return self.predict_action(inputs)

    def inference(self, payload: Dict[str, Any]):
        """
        Main inference method with action queue support.
        
        Args:
            payload: Input payload containing:
                - All normal inference inputs (state, images, etc.)
                - force_reinfer (optional): If True, regenerate action sequence and reset queue
        
        Returns:
            Single action [Dim] from the action queue, or full sequence if force_reinfer=True
        """
        force_reinfer = payload.pop("force_reinfer", False)
        
        # If force_reinfer is True or queue is empty, generate new action sequence
        if force_reinfer or not self.action_queue:
            print("Generating new action sequence...")
            if force_reinfer:
                self.action_queue.clear()
            action_sequence = self._generate_action_sequence(payload)
            self._populate_action_queue(action_sequence)
        
        # Return the next action from the queue
        return self._get_next_action()


class GO1Server:
    def __init__(
        self,
        model_path: Union[str, Path],
        data_stats_path: Union[str, Path] = None,
        cfg_path: Union[str, Path] = None,
        batch_size: int = 1,
    ) -> Path:
        self.model = GO1Infer(
            model_path=model_path,
            data_stats_path=data_stats_path,
            cfg_path=cfg_path,
            batch_size=batch_size,
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()

        @self.app.post("/act")
        async def act_endpoint(request: Request):
            payload = await request.json()
            actions = self.model.inference(payload)

            return actions.tolist() if hasattr(actions, "tolist") else actions

        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # Model Configuration
    model_path: Union[str, Path]
    data_stats_path: Union[str, Path] = None
    cfg_path: Union[str, Path] = None

    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 9000  # Host Port
    
    # Batch Configuration
    batch_size: int = 1


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = GO1Server(
        cfg.model_path,
        cfg.data_stats_path,
        cfg.cfg_path,
        batch_size=cfg.batch_size,
    )
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
