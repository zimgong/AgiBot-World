import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal

import torch
import torch.nn.functional as F
import transformers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from timm.models.vision_transformer import Mlp
from torch import nn
from torch.func import jvp as jvp_func
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from go1.internvl.conversation import get_conv_template
from go1.internvl.model.internlm2.modeling_internlm2 import InternLM2RMSNorm
from go1.internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
from go1.internvl.model.internvl_chat.modeling_internvl_chat import version_cmp

from .configuration_go1 import GO1ModelConfig
from .modeling_action_expert import ActionExpertModel
from .modeling_internlm2_go1 import InternLM2ForCausalLMGO1
from .modeling_latent_action_expert import LatentPlannerModel

logger = logging.get_logger(__name__)


@dataclass
class ActionModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    action_loss: Optional[torch.FloatTensor] = None
    action_logits: Optional[torch.FloatTensor] = None
    action_gts: Optional[torch.FloatTensor] = None
    # NEW: for RL/SDE sampling (sum over time and dims per batch item)
    action_logprob: Optional[torch.FloatTensor] = None


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = InternLM2RMSNorm(hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class GO1Model(PreTrainedModel):
    config_class = GO1ModelConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = [
        "InternVisionModel",
        "InternLM2DecoderLayerGO1",
        "ActionExertDecoderLayer",
    ]
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: GO1ModelConfig,
        vision_model=None,
        language_model=None,
        action_model=None,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(config)
        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        self.torch_dtype = torch_dtype

        # Decoder configuration (back-compat default)
        self.decoder_type: Literal["DDPM", "mean_flow", "flow_matching"] = getattr(
            config, "decoder_type", "DDPM"
        )
        self.dispersive_cfg = getattr(config, "dispersive", None)
        self.use_dispersive = bool(self.dispersive_cfg and self.dispersive_cfg.get("use", False))
        
        # NEW: FM-SDE sampler config (inference-time only by default)
        # Example:
        # flow_sde = dict(
        #   use=False, mode="churn", s_churn=0.7, eps=1e-4,
        #   learn_sigma=False, sigma_init=1.0, clip_t=1e-5, return_logprob=False
        # )
        self.flow_sde_cfg = getattr(config, "flow_sde", {}) or {}
        self.flow_sde_use = bool(self.flow_sde_cfg.get("use", False))
        self.flow_sde_return_logprob = bool(self.flow_sde_cfg.get("return_logprob", False))
        self.flow_sde_clip_t = float(self.flow_sde_cfg.get("clip_t", 1e-5))
        # Optional learnable scalar multiplier for sigma(t). No new layers/heads.
        self._sigma_learnable: Optional[nn.Parameter] = None
        if bool(self.flow_sde_cfg.get("learn_sigma", False)):
            init = float(self.flow_sde_cfg.get("sigma_init", 1.0))
            self._sigma_learnable = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        
        # Optional warning for MeanFlow + FlashAttention compatibility
        if self.decoder_type == "mean_flow" and hasattr(config, "action_config") and config.action_config.attn_implementation == "flash_attention_2":
            logger.warning("MeanFlow + JVP may conflict with FlashAttention. Consider setting attn_implementation='eager' for training.")

        self.max_new_tokens = 1024
        self.img_context_token_id = getattr(config, "img_context_token_id", 92546)
        self.pad_token_id = config.pad_token_id

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
            if self.vision_model.dtype != self.torch_dtype:
                self.vision_model.to(self.torch_dtype)
        else:
            self.vision_model = InternVisionModel(config.vision_config).to(self.torch_dtype)
        if language_model is not None:
            self.language_model = language_model
            if language_model.dtype != self.torch_dtype:
                self.language_model.to(self.torch_dtype)
        else:
            if "InternLM2ForCausalLM" in config.llm_config.architectures[0]:
                self.language_model = InternLM2ForCausalLMGO1(config.llm_config).to(self.torch_dtype)
            else:
                raise NotImplementedError(f"{config.llm_config.architectures[0]} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        llm_head_dim = llm_hidden_size // config.llm_config.num_attention_heads

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, dtype=self.torch_dtype),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size, dtype=self.torch_dtype),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, dtype=self.torch_dtype),
        )

        self.conv_template = get_conv_template(self.template)
        if hasattr(config, "system_message"):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        # Action Expert related initialization
        action_config = config.action_config
        if action_model is not None:
            self.action_model = action_model
            if self.action_model.dtype != self.torch_dtype:
                self.action_model.to(self.torch_dtype)
        else:
            self.action_model = ActionExpertModel(action_config).to(self.torch_dtype)
        action_hidden_size = action_config.hidden_size
        self.action_dim = action_config.action_dim
        action_head_dim = action_hidden_size // action_config.num_attention_heads

        # latent action expert related initialization
        latent_planner_config = config.latent_planner_config
        if config.latent_planning:
            self.enable_lam = True
            self.latent_planner = LatentPlannerModel(
                action_head_dim,
                config,
                latent_planner_config,
                self.torch_dtype,
            )
            latent_action_head_dim = self.latent_planner.latent_action_head_dim
        else:
            self.enable_lam = False

        self.k_proj_layers = nn.ModuleList(
            [
                nn.Linear(
                    llm_head_dim,  # 128 to 64
                    latent_action_head_dim if config.latent_planning else action_head_dim,
                    dtype=self.torch_dtype,
                )
                for _ in range(config.llm_config.num_hidden_layers)
            ]
        )
        self.v_proj_layers = nn.ModuleList(
            [
                nn.Linear(
                    llm_head_dim,
                    latent_action_head_dim if config.latent_planning else action_head_dim,
                    dtype=self.torch_dtype,
                )
                for _ in range(config.llm_config.num_hidden_layers)
            ]
        )
        self.time_embedder = TimestepEmbedder(action_hidden_size, dtype=self.torch_dtype)
        self.freq_embedder = TimestepEmbedder(action_hidden_size, dtype=self.torch_dtype)
        
        # MeanFlow/FM: extra time embedder for end-time r (for FM we set r=t)
        self.time_embedder_r = TimestepEmbedder(action_hidden_size, dtype=self.torch_dtype)

        self.state_adaptor = nn.Sequential(
            nn.Linear(
                (action_config.state_dim),
                action_config.hidden_size,
                dtype=self.torch_dtype,
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
        )

        self.action_adaptor = nn.Sequential(
            nn.Linear(
                action_config.action_dim,
                action_config.hidden_size,
                dtype=self.torch_dtype,
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
        )

        self.final_layer = FinalLayer(action_hidden_size, self.action_dim).to(self.torch_dtype)
        self.init_linear_weights()

        # Noise scheduluar related initialization
        self.action_chunk_size = config.action_chunk_size
        noise_scheduler_config = config.noise_scheduler_config
        self.num_train_timesteps = noise_scheduler_config["num_train_timesteps"]
        self.num_inference_timesteps = noise_scheduler_config["num_inference_timesteps"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_schedule=noise_scheduler_config["beta_schedule"],
            prediction_type=noise_scheduler_config["prediction_type"],
            clip_sample=noise_scheduler_config["clip_sample"],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_schedule=noise_scheduler_config["beta_schedule"],
            prediction_type=noise_scheduler_config["prediction_type"],
        )

    def init_linear_weights(self) -> None:
        # Initialize the kv cache projector
        for k_linear, v_linear in zip(self.k_proj_layers, self.v_proj_layers):
            nn.init.xavier_uniform_(k_linear.weight)
            nn.init.xavier_uniform_(v_linear.weight)
            if k_linear.bias is not None:
                nn.init.constant_(k_linear.bias, 0)
            if v_linear.bias is not None:
                nn.init.constant_(v_linear.bias, 0)

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        if int(h * scale_factor) / scale_factor != h or int(w * scale_factor) / scale_factor != w:
            w_ = int(int(w * scale_factor) / scale_factor)
            h_ = int(int(h * scale_factor) / scale_factor)
            x = F.interpolate(x.permute(0, 3, 1, 2), size=(w_, h_), mode="bilinear", align_corners=False)
            x = x.permute(0, 2, 3, 1).contiguous()
            n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def calc_action_diffusion_loss(
        self,
        action_logits: torch.FloatTensor,
        action_gts: Optional[torch.Tensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        pred_type: str = "sample",
    ) -> torch.Tensor:
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = action_gts
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(action_logits, target)
        return loss

    # === MeanFlow: utilities ===
    def _embed_T_R_F(self, t: torch.Tensor, r: torch.Tensor, ctrl_freqs: torch.Tensor):
        """
        Returns (t_token, r_token, f_token) with shape [B, 1, C] each.
        t and r should be floats in [0, 1].
        """
        t_token = self.time_embedder(t)           # [B, 1, C]
        r_token = self.time_embedder_r(r)        # [B, 1, C]
        f_token = self.freq_embedder(ctrl_freqs) # [B, 1, C]
        return t_token, r_token, f_token

    def _u_theta(self, z_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor,
                 state: torch.Tensor, ctrl_freqs: torch.Tensor,
                 attention_mask: torch.Tensor,
                 vlm_key_values_downsample: Tuple[Tuple[torch.FloatTensor]]):
        """
        u_theta network head. Returns predicted average velocity over the action chunk.
        Shapes:
          z_t: [B, H, action_dim]  (trajectory at time t)
          t,r: [B, 1]  in [0,1]
        """
        # Adapters
        state_traj = self.state_adaptor(state)      # [B, 1, C]
        action_traj = self.action_adaptor(z_t)      # [B, H, C]

        # Tokens
        t_tok, r_tok, f_tok = self._embed_T_R_F(t, r, ctrl_freqs)

        # Pack to action model: [T, R, F, State, Action...]
        state_action_trajs_w_tfps = torch.cat([t_tok, r_tok, f_tok, state_traj, action_traj], dim=1)

        # Forward through Action Expert (decoder)
        outputs = self.action_model(
            state_action_traj=state_action_trajs_w_tfps,
            attention_mask=attention_mask,
            vlm_key_values=vlm_key_values_downsample,
        )
        hidden = outputs[0]
        action_h = hidden[:, -self.action_chunk_size:, ...]
        # Project to action_dim: interpret as u_theta (average velocity)
        u_pred = self.final_layer(action_h)        # [B, H, action_dim]
        return u_pred

    # === FlowMatching: reuse the same head, passing r=t so shapes stay identical ===
    def _v_theta(self, z_t: torch.Tensor, t: torch.Tensor,
                 state: torch.Tensor, ctrl_freqs: torch.Tensor,
                 attention_mask: torch.Tensor,
                 vlm_key_values_downsample: Tuple[Tuple[torch.FloatTensor]]):
        return self._u_theta(
            z_t=z_t, t=t, r=t, state=state,
            ctrl_freqs=ctrl_freqs,
            attention_mask=attention_mask,
            vlm_key_values_downsample=vlm_key_values_downsample,
        )

    # === FM-SDE utils ===
    def _sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sigma schedule in latent action space. If learnable is enabled, we
        multiply the schedule by softplus(parameter) to keep it >= 0.
        """
        cfg = self.flow_sde_cfg
        mode = (cfg.get("mode", "churn") or "churn").lower()  # "churn" | "fixed"
        eps = float(cfg.get("eps", 1e-4))
        s_churn = float(cfg.get("s_churn", 0.0))
        if mode == "fixed":
            base = torch.ones_like(t)
        else:
            # user-proposed churn-like schedule: sqrt(1 - t/(t+eps))
            base = torch.sqrt(torch.clamp(1.0 - t / (t + eps), min=0.0))
        scale = s_churn
        if self._sigma_learnable is not None:
            scale = scale * F.softplus(self._sigma_learnable)  # >= 0
        return (scale * base).to(t.dtype)

    def _fm_sde_drift(self, z: torch.Tensor, t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """
        Marginal-preserving SDE drift:
          v_tilde = v + 0.5 * sigma(t)^2 * ((t * v - z) / (1 - t))
        (Uses the score–velocity identity for linear FM.)
        """
        # clamp 1 - t for numerical stability near t=1
        one_minus_t = torch.clamp(1.0 - t, min=self.flow_sde_clip_t)
        sigma = self._sigma_t(t)                       # [B,1]
        score = (t * v_pred - z) / one_minus_t        # broadcast over [B,H,D]
        return v_pred + 0.5 * (sigma ** 2) * score

    def calc_action_meanflow_loss(
        self,
        action_gts: torch.Tensor,                         # [B, H, action_dim]
        state: torch.Tensor,                              # [B, state_dim]
        ctrl_freqs: torch.Tensor,                         # [B, 1] ints ok
        attention_mask: torch.Tensor,                     # [B, vlm_len] (bool)
        vlm_key_values_downsample: Tuple[Tuple[torch.FloatTensor]],
    ):
        """
        Implements Eq. (MeanFlow identity):
          u(z_t,r,t) = v(z_t,t) - (t - r) * d/dt u(z_t,r,t),
        with d/dt u computed by JVP along direction (v, 1) w.r.t inputs (z_t, t).
        v = a - e under linear interpolation z_t = (1 - t) * e + t * a.
        """
        B, H, D = action_gts.shape
        device = action_gts.device
        dtype  = action_gts.dtype

        # Sample Gaussian noise in action space, and times t<r
        e = torch.randn_like(action_gts)                         # [B, H, D]
        t = torch.rand(B, 1, device=device, dtype=torch.float32) # [B,1] in [0,1]
        # r ~ Uniform(t, 1]
        r = t + (1.0 - t) * torch.rand_like(t)

        # Linear interpolation (make t broadcast across H and D)
        t_b = t.view(B, 1, 1).to(e.dtype)                       # [B,1,1]
        z_t = (1.0 - t_b) * e + t_b * action_gts                # [B, H, D]
        v   = action_gts - e                                    # [B, H, D]

        # closure for JVP: f(z, t) -> u_theta(z,t,r,cond)
        def f_func(z_var, t_var):
            return self._u_theta(
                z_t=z_var, t=t_var, r=r, state=state,
                ctrl_freqs=ctrl_freqs,
                attention_mask=attention_mask,
                vlm_key_values_downsample=vlm_key_values_downsample,
            )

        # Compute u_pred and its total derivative wrt time (directional derivative):
        # dudt = ∂u/∂z ⋅ v + ∂u/∂t ⋅ 1  (r is held constant)
        # NOTE: if your environment mixes bf16/flash-attn, you may want to temporarily disable autocast.
        u_pred, dudt = jvp_func(f_func, (z_t, t), (v, torch.ones_like(t)))

        # MeanFlow target from identity
        coeff = (t - r).view(B, 1, 1).to(dudt.dtype)            # [B,1,1]
        target = v.to(dudt.dtype) - coeff * dudt

        # L2 loss on average velocity
        mf_loss = F.mse_loss(u_pred, target)

        # Optional dispersive regularization on embeddings (see DM1 Fig.2)
        disp_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.use_dispersive:
            disp_loss = self.calc_dispersive_loss(state, ctrl_freqs, t, r)

        return mf_loss, u_pred, disp_loss

    # === FlowMatching loss (no JVP) ===
    def calc_action_flowmatch_loss(
        self,
        action_gts: torch.Tensor,                         # [B, H, D]
        state: torch.Tensor,                              # [B, state_dim]
        ctrl_freqs: torch.Tensor,                         # [B, 1]
        attention_mask: torch.Tensor,                     # [B, vlm_len]
        vlm_key_values_downsample: Tuple[Tuple[torch.FloatTensor]],
    ):
        B, H, D = action_gts.shape
        device = action_gts.device
        dtype  = action_gts.dtype

        # Noise and interpolation time
        e = torch.randn_like(action_gts)
        t = torch.rand(B, 1, device=device, dtype=torch.float32)  # U[0,1]
        t_b = t.view(B, 1, 1).to(e.dtype)
        z_t = (1.0 - t_b) * e + t_b * action_gts                  # [B,H,D]
        v   = (action_gts - e)                                    # [B,H,D]

        # Predict instantaneous velocity field
        v_pred = self._v_theta(
            z_t=z_t, t=t, state=state,
            ctrl_freqs=ctrl_freqs,
            attention_mask=attention_mask,
            vlm_key_values_downsample=vlm_key_values_downsample,
        )

        fm_loss = F.mse_loss(v_pred, v.to(v_pred.dtype))

        disp_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.use_dispersive:
            # same light-weight dispersion used for MeanFlow
            disp_loss = self.calc_dispersive_loss(state, ctrl_freqs, t, t)  # r=t

        return fm_loss, v_pred, disp_loss

    # === MeanFlow: dispersive losses (lightweight variants) ===
    def _pairwise_cos(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        return x @ x.T  # [B,B]

    def _pairwise_l2(self, x: torch.Tensor) -> torch.Tensor:
        # squared L2 distances
        xx = (x * x).sum(-1, keepdim=True)
        d2 = xx + xx.T - 2.0 * (x @ x.T)
        return d2.clamp_min_(0)

    def _hinge_disp(self, h: torch.Tensor, margin: float) -> torch.Tensor:
        # encourage ||h_i - h_j||_2 >= margin
        d = torch.sqrt(self._pairwise_l2(h) + 1e-9)
        B = d.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool, device=h.device)
        return F.relu(margin - d[mask]).mean()

    def _cov_disp(self, h: torch.Tensor, min_var: float = 1e-3, var_w: float = 0.0) -> torch.Tensor:
        # decorrelate features & enforce minimum variance per dim
        h = h - h.mean(dim=0, keepdim=True)
        C = (h.T @ h) / (h.shape[0] - 1 + 1e-6)  # [C,C]
        offdiag = C - torch.diag(torch.diag(C))
        cov_loss = (offdiag**2).mean()
        if var_w > 0.0:
            var = torch.diag(C)
            cov_loss = cov_loss + var_w * F.relu(min_var - var).mean()
        return cov_loss

    def _infonce_cos(self, h: torch.Tensor, tau: float) -> torch.Tensor:
        # InfoNCE-style "all-negatives" cosine separation (no positives): push apart batch.
        # logits_ij = cos_ij / tau ; remove diagonal; maximize normalization entropy.
        cos = self._pairwise_cos(h)  # [B,B]
        B = cos.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool, device=h.device)
        logits = cos[mask].view(B, B-1) / max(tau, 1e-6)
        # maximize uniformity: minimize logsumexp (equivalent to spreading directions)
        # use -logsumexp as proxy since there is no designated positive.
        return -torch.logsumexp(logits, dim=-1).mean()

    def calc_dispersive_loss(self, state: torch.Tensor, ctrl_freqs: torch.Tensor,
                             t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Apply dispersive regularization to embeddings of T, R and Cond (state+freq)."""
        cfg = self.dispersive_cfg or {}
        loss_type = cfg.get("loss_type", "infonce_cos")
        weight    = float(cfg.get("weight", 0.0))
        if weight <= 0.0:
            return torch.tensor(0.0, device=state.device, dtype=state.dtype)

        targets = cfg.get("target", ["T","R","Cond"])
        tau     = float(cfg.get("temperature", 0.3))
        margin  = float(cfg.get("margin", 0.3))
        var_w   = float(cfg.get("variance_weight", 0.0))

        # Build the three embeddings (shape [B, 1, C] -> [B, C])
        t_h, r_h, f_h = self._embed_T_R_F(t, r, ctrl_freqs)
        cond_h = (self.state_adaptor(state) + f_h)[:, 0, :]  # [B,C]
        t_h = t_h[:,0,:]; r_h = r_h[:,0,:]

        def disp(h):
            if loss_type == "hinge":
                return self._hinge_disp(h, margin)
            elif loss_type == "cov":
                return self._cov_disp(h, min_var=1e-3, var_w=var_w)
            else:  # default: "infonce_cos"
                return self._infonce_cos(h, tau=tau)

        disp_losses = []
        if "T" in targets:    disp_losses.append(disp(t_h))
        if "R" in targets:    disp_losses.append(disp(r_h))
        if "Cond" in targets: disp_losses.append(disp(cond_h))
        if not disp_losses:
            return torch.tensor(0.0, device=state.device, dtype=state.dtype)
        return weight * torch.stack(disp_losses).sum()

    def condition_sample(
        self,
        state: torch.Tensor,
        vlm_key_values_downsample: torch.FloatTensor,
        attention_mask: torch.Tensor,
        ctrl_freqs: int = 30,
    ):
        state_traj = self.state_adaptor(state)  # (B, 1, C)
        device, dtype = state_traj.device, state_traj.dtype
        # Sample noise that we'll add to the actions
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.action_chunk_size, self.action_dim), device=device, dtype=dtype
        )
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            action_traj = self.action_adaptor(noisy_action)  # (B, H, C)
            timestep_tokens = self.time_embedder(
                t * torch.ones_like(ctrl_freqs, dtype=ctrl_freqs.dtype).to(device)
            )  # (B, 1, C)
            freq_tokens = self.freq_embedder(ctrl_freqs)  # (B, 1, C)
            state_action_trajs_w_tfps = torch.cat([timestep_tokens, freq_tokens, state_traj, action_traj], dim=1)

            # Predict the model output
            model_output = self.action_model(state_action_trajs_w_tfps, attention_mask, vlm_key_values_downsample)

            state_action_output_tokens = model_output[0]
            action_output_tokens = state_action_output_tokens[:, -self.action_chunk_size :, ...]
            action_output = self.final_layer(action_output_tokens)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(action_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(dtype)

        return noisy_action

    def meanflow_sample(
        self,
        state: torch.Tensor,                                   # [B, state_dim]
        vlm_key_values_downsample: Tuple[Tuple[torch.FloatTensor]],
        attention_mask: torch.Tensor,                          # [B, vlm_len] bool
        ctrl_freqs: torch.Tensor,                              # [B, 1] (int or float ok)
    ) -> torch.Tensor:
        """
        Multi-step MeanFlow sampling using the scheduler's timesteps as a grid.

        We construct an increasing grid 0 = t_0 < ... < t_K = 1 from
        self.noise_scheduler_sample.timesteps (integer grid) and apply:
            z_{t_{i+1}} = z_{t_i} + u_theta(z_{t_i}, r=t_{i+1}, t=t_i) * (t_{i+1} - t_i)

        When K==1 this recovers the original one-step rule.
        """
        device = state.device
        dtype  = self.torch_dtype

        B = state.shape[0]
        # Initialize z_{t0} from the noise prior (same shape as actions)
        z = torch.randn((B, self.action_chunk_size, self.action_dim), device=device, dtype=dtype)

        # Build the time grid from the scheduler
        # Example: DPMSolver gives descending ints like [999, ..., 0].
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        raw_ts = self.noise_scheduler_sample.timesteps.to(device).float()  # [K], typically descending

        if raw_ts.numel() == 0:
            # Fallback to strict one-step (0 -> 1)
            t = torch.zeros((B, 1), device=device, dtype=torch.float32)
            r = torch.ones((B, 1),  device=device, dtype=torch.float32)
            u = self._u_theta(
                z_t=z, t=t, r=r, state=state,
                ctrl_freqs=ctrl_freqs, attention_mask=attention_mask,
                vlm_key_values_downsample=vlm_key_values_downsample,
            )
            return (z + u).to(dtype)

        # Map integer 'raw_ts' to normalized [0,1] with ascending order:
        #   largest raw_ts -> 0.0, smallest raw_ts -> 1.0
        t_max = max(raw_ts.max().item(), 1.0)
        t_norm = (t_max - raw_ts) / t_max                     # ascending in [0,1]
        # Append the terminal time 1.0 to form pairs (t_i, t_{i+1})
        t_grid = torch.cat([t_norm, t_norm.new_tensor([1.0])], dim=0)  # [K+1]

        # Multi-step MeanFlow updates
        K = raw_ts.numel()
        for i in range(K):
            t_cur  = t_grid[i].item()
            t_next = t_grid[i + 1].item()

            # Batch broadcasted time tokens
            t_b = torch.full((B, 1), t_cur,  device=device, dtype=torch.float32)
            r_b = torch.full((B, 1), t_next, device=device, dtype=torch.float32)

            u_pred = self._u_theta(
                z_t=z, t=t_b, r=r_b, state=state,
                ctrl_freqs=ctrl_freqs, attention_mask=attention_mask,
                vlm_key_values_downsample=vlm_key_values_downsample,
            )  # [B, H, action_dim]

            # Δt as a scalar; broadcast over [B,H,D]
            dt = torch.tensor(t_next - t_cur, device=device, dtype=z.dtype)
            z = (z + dt * u_pred).to(dtype)

        return z

    def flowmatch_sample(
        self,
        state: torch.Tensor,                                   # [B, state_dim]
        vlm_key_values_downsample: Tuple[Tuple[torch.FloatTensor]],
        attention_mask: torch.Tensor,                          # [B, vlm_len]
        ctrl_freqs: torch.Tensor,                              # [B, 1]
    ) -> torch.Tensor:
        """
        Multi-step ODE integration for Flow Matching:
          z_{t_{i+1}} = z_{t_i} + v_theta(z_{t_i}, t_i) * (t_{i+1} - t_i)
        using the same scheduler grid you use for DPMSolver.
        """
        device = state.device
        dtype  = self.torch_dtype
        B = state.shape[0]
        z = torch.randn((B, self.action_chunk_size, self.action_dim), device=device, dtype=dtype)

        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        raw_ts = self.noise_scheduler_sample.timesteps.to(device).float()  # descending ints
        if raw_ts.numel() == 0:
            t = torch.zeros((B, 1), device=device, dtype=torch.float32)
            v_pred = self._v_theta(
                z_t=z, t=t, state=state,
                ctrl_freqs=ctrl_freqs, attention_mask=attention_mask,
                vlm_key_values_downsample=vlm_key_values_downsample,
            )
            return (z + v_pred).to(dtype)

        t_max = max(raw_ts.max().item(), 1.0)
        t_norm = (t_max - raw_ts) / t_max  # ascending [0,1]
        t_grid = torch.cat([t_norm, t_norm.new_tensor([1.0])], dim=0)
        K = raw_ts.numel()
        for i in range(K):
            t_cur  = t_grid[i].item()
            t_next = t_grid[i + 1].item()
            t_b = torch.full((B, 1), t_cur,  device=device, dtype=torch.float32)
            v_pred = self._v_theta(
                z_t=z, t=t_b, state=state,
                ctrl_freqs=ctrl_freqs, attention_mask=attention_mask,
                vlm_key_values_downsample=vlm_key_values_downsample,
            )
            dt = torch.tensor(t_next - t_cur, device=device, dtype=z.dtype)
            z = (z + dt * v_pred).to(dtype)
        return z

    def flowmatch_sde_sample(
        self,
        state: torch.Tensor,                                   # [B, state_dim]
        vlm_key_values_downsample: Tuple[Tuple[torch.FloatTensor]],
        attention_mask: torch.Tensor,                          # [B, vlm_len]
        ctrl_freqs: torch.Tensor,                              # [B, 1]
        return_logprob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        FM-SDE sampler (Euler–Maruyama) on the same time grid as ODE sampling.
        Returns (actions, logprob) if return_logprob=True, else (actions, None).
        Each step is a Gaussian transition -> SAC-style log-prob accumulation.
        """
        device = state.device
        dtype  = self.torch_dtype
        B = state.shape[0]
        # init from prior in latent action space
        z = torch.randn((B, self.action_chunk_size, self.action_dim), device=device, dtype=dtype)
        logprob = None
        if return_logprob:
            logprob = torch.zeros((B,), device=device, dtype=dtype)

        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        raw_ts = self.noise_scheduler_sample.timesteps.to(device).float()  # descending ints
        # normalized ascending grid in [0,1]
        if raw_ts.numel() == 0:
            t = torch.zeros((B, 1), device=device, dtype=torch.float32)
            v = self._v_theta(z, t, state, ctrl_freqs, attention_mask, vlm_key_values_downsample)
            vtilde = self._fm_sde_drift(z, t, v)
            # single step to t=1
            dt = torch.tensor(1.0, device=device, dtype=z.dtype)
            sigma = self._sigma_t(t)
            eps = torch.randn_like(z)
            z = (z + vtilde * dt + sigma * torch.sqrt(dt) * eps).to(dtype)
            if return_logprob:
                # SAC-style logprob: gradients only to learnable noise (sigma), not to main model (v_theta)
                if bool(self.flow_sde_cfg.get("logprob_sigma_only", True)):
                    # epsilon-based logprob: depends only on sigma (and dt)
                    # z_new = mu + sigma*sqrt(dt)*eps  ==>  logN(eps;0,I) - sum_j log(sigma*sqrt(dt))
                    D = self.action_chunk_size * self.action_dim
                    log_det = D * torch.log(sigma.squeeze(1) * torch.sqrt(dt.clamp_min(0) + 1e-12))
                    nll_eps = 0.5 * (D * torch.log(2 * torch.pi) + (eps * eps).sum(dim=(1, 2)))
                    logprob += -(nll_eps + log_det)
                else:
                    # fallback: classic formula (still yields zero grad to mu for unsquashed Gaussians)
                    var = (sigma ** 2) * dt  # [B,1]
                    quad = (z - mu).pow(2).sum(dim=(1,2)) / (var.squeeze(1) + 1e-12)
                    const = z.new_tensor(self.action_chunk_size * self.action_dim, dtype=dtype)
                    logprob += -0.5 * (const * torch.log(2 * torch.pi * var.squeeze(1) + 1e-12) + quad)
            return z, logprob

        t_max = max(raw_ts.max().item(), 1.0)
        t_norm = (t_max - raw_ts) / t_max  # ascending in [0,1]
        t_grid = torch.cat([t_norm, t_norm.new_tensor([1.0])], dim=0)
        K = raw_ts.numel()
        for i in range(K):
            t_cur  = t_grid[i].item()
            t_next = t_grid[i + 1].item()
            dt = torch.tensor(t_next - t_cur, device=device, dtype=z.dtype)
            t_b = torch.full((B, 1), t_cur, device=device, dtype=torch.float32)

            # predict velocity and convert to marginal-preserving SDE drift
            v = self._v_theta(z, t_b, state, ctrl_freqs, attention_mask, vlm_key_values_downsample)
            vtilde = self._fm_sde_drift(z, t_b, v)

            # Gaussian transition
            sigma = self._sigma_t(t_b)  # [B,1]
            eps = torch.randn_like(z)
            noise = sigma * torch.sqrt(dt.clamp_min(0)) * eps
            mu = z + vtilde * dt
            z = (mu + noise).to(dtype)

            if return_logprob:
                # SAC-style logprob: gradients only to learnable noise (sigma), not to main model (v_theta)
                if bool(self.flow_sde_cfg.get("logprob_sigma_only", True)):
                    # epsilon-based logprob: depends only on sigma (and dt)
                    # z_new = mu + sigma*sqrt(dt)*eps  ==>  logN(eps;0,I) - sum_j log(sigma*sqrt(dt))
                    D = self.action_chunk_size * self.action_dim
                    log_det = D * torch.log(sigma.squeeze(1) * torch.sqrt(dt.clamp_min(0) + 1e-12))
                    nll_eps = 0.5 * (D * torch.log(2 * torch.pi) + (eps * eps).sum(dim=(1, 2)))
                    logprob += -(nll_eps + log_det)
                else:
                    # fallback: classic formula (still yields zero grad to mu for unsquashed Gaussians)
                    var = (sigma ** 2) * dt  # [B,1]
                    quad = (z - mu).pow(2).sum(dim=(1,2)) / (var.squeeze(1) + 1e-12)
                    const = z.new_tensor(self.action_chunk_size * self.action_dim, dtype=dtype)
                    logprob += -0.5 * (const * torch.log(2 * torch.pi * var.squeeze(1) + 1e-12) + quad)
        return z, logprob


    def common_process(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        # Align with original InternVL implementation style
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(-1, C)

        input_ids = input_ids.reshape(-1)

        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            raise ValueError(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, vit_embeds.shape={vit_embeds.shape}"
            )

        input_embeds = input_embeds.reshape(B, N, C)

        vlm_outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
            labels=labels,
        )
        return vlm_outputs

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        state: torch.Tensor = None,
        ctrl_freqs: int = None,
        action_gts: torch.Tensor = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, ActionModelOutputWithPast]:
        vlm_outputs = self.common_process(
            pixel_values,
            input_ids,
            attention_mask,
            position_ids,
            image_flags,
            return_dict,
            labels=labels,
        )
        # Project vlm kv_cache head dim into action expert head dim
        vlm_key_values = vlm_outputs.past_key_values  # kv_cache(multi_head) of each decoder layer

        vlm_key_values_downsample = []
        for vlm_key_value, k_proj, v_proj in zip(vlm_key_values, self.k_proj_layers, self.v_proj_layers):
            vlm_key_values_downsample.append((k_proj(vlm_key_value[0]), v_proj(vlm_key_value[1])))

        B = input_ids.shape[0]

        if self.enable_lam:
            (
                latent_vlm_key_values_downsample,
                outputs_latent,
            ) = self.latent_planner(
                vlm_key_values_downsample,
                attention_mask,
            )

        action_loss = torch.tensor(0.0, dtype=state.dtype, device=state.device)
        action_logits = None
        if self.training:
            if self.decoder_type == "DDPM":
                # --- existing diffusion training (unchanged) ---
                noise = torch.randn(action_gts.shape, dtype=action_gts.dtype, device=action_gts.device)
                timesteps = torch.randint(0, self.num_train_timesteps, (B, 1), device=action_gts.device).long()
                timestep_tokens = self.time_embedder(timesteps)  # (B, 1, C)
                freq_tokens = self.freq_embedder(ctrl_freqs)  # (B, 1, C)
                noisy_action = self.noise_scheduler.add_noise(action_gts, noise, timesteps)  # (B, H, C)

                state_trajs = self.state_adaptor(state)
                action_trajs = self.action_adaptor(noisy_action)
                state_action_trajs = torch.cat([state_trajs, action_trajs], dim=1)

                state_action_trajs_w_tfps = torch.cat(
                    [timestep_tokens, freq_tokens, state_action_trajs], dim=1
                )  # (B, H+3, C)

                # Action expert as diffusion head
                if self.enable_lam:
                    vlm_key_values_downsample = latent_vlm_key_values_downsample
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                B, self.latent_planner.latent_token_nums, dtype=torch.bool, device=attention_mask.device
                            ),
                        ),
                        dim=1,
                    )

                outputs = self.action_model(state_action_trajs_w_tfps, attention_mask, vlm_key_values_downsample)
                state_action_output_tokens = outputs[0]
                action_output_tokens = state_action_output_tokens[:, -self.action_chunk_size :, ...]
                action_logits = self.final_layer(action_output_tokens)

                # Calculate action mse loss using output action chunk
                action_loss = self.calc_action_diffusion_loss(
                    action_logits=action_logits,
                    action_gts=action_gts,
                    pred_type="sample",
                )
            elif self.decoder_type == "mean_flow":
                # --- NEW: MeanFlow training ---
                if self.enable_lam:
                    vlm_key_values_downsample = latent_vlm_key_values_downsample
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                B, self.latent_planner.latent_token_nums, dtype=torch.bool, device=attention_mask.device
                            ),
                        ),
                        dim=1,
                    )

                mf_loss, u_pred, disp_loss = self.calc_action_meanflow_loss(
                    action_gts=action_gts,
                    state=state,
                    ctrl_freqs=ctrl_freqs,
                    attention_mask=attention_mask,
                    vlm_key_values_downsample=vlm_key_values_downsample,
                )
                # For logging/compat, expose "action_logits" as denoised actions estimate: e + u_pred (not used in loss)
                # We rebuild e here for shape-consistent logging if needed.
                with torch.no_grad():
                    e_dbg = torch.zeros_like(action_gts)  # or keep last e if you prefer
                action_logits = e_dbg + u_pred
                action_loss = mf_loss + disp_loss
            elif self.decoder_type == "flow_matching":
                # --- NEW: FlowMatching training ---
                if self.enable_lam:
                    vlm_key_values_downsample = latent_vlm_key_values_downsample
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                B, self.latent_planner.latent_token_nums, dtype=torch.bool, device=attention_mask.device
                            ),
                        ),
                        dim=1,
                    )
                fm_loss, v_pred, disp_loss = self.calc_action_flowmatch_loss(
                    action_gts=action_gts,
                    state=state,
                    ctrl_freqs=ctrl_freqs,
                    attention_mask=attention_mask,
                    vlm_key_values_downsample=vlm_key_values_downsample,
                )
                # for logging consistency expose predicted velocity
                action_logits = v_pred
                action_loss = fm_loss + disp_loss
        else:
            # --- inference ---
            if self.enable_lam:
                vlm_key_values_downsample = latent_vlm_key_values_downsample
                attention_mask = torch.cat(
                    (attention_mask, torch.ones(B, 4, dtype=torch.bool, device=attention_mask.device)), dim=1
                )

            if self.decoder_type == "DDPM":
                action_logits = self.condition_sample(state, vlm_key_values_downsample, attention_mask, ctrl_freqs)
            elif self.decoder_type == "mean_flow":
                action_logits = self.meanflow_sample(state, vlm_key_values_downsample, attention_mask, ctrl_freqs)
            elif self.decoder_type == "flow_matching":
                # ODE (deterministic) or SDE (stochastic) based on config or runtime kwarg
                use_sde = self.flow_sde_use or bool(kwargs.get("sampler", "") == "sde")
                want_logp = self.flow_sde_return_logprob or bool(kwargs.get("return_logprob", False))
                if use_sde:
                    action_logits, logprob = self.flowmatch_sde_sample(
                        state, vlm_key_values_downsample, attention_mask, ctrl_freqs,
                        return_logprob=want_logp,
                    )
                    action_logprob = logprob
                else:
                    action_logits = self.flowmatch_sample(
                        state, vlm_key_values_downsample, attention_mask, ctrl_freqs
                    )
                    action_logprob = None

        loss = action_loss

        if not return_dict:
            return (loss, action_logits)

        return ActionModelOutputWithPast(
            loss=loss,
            action_loss=action_loss,
            action_logits=action_logits,
            action_gts=action_gts,
            action_logprob=locals().get("action_logprob", None),
        )
