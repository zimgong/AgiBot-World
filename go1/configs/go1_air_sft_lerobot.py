import os
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments

from go1.configs.go1_base_cfg import BaseDatasetArguments, BaseModelArguments, BaseSpaceArguments
from go1.tools.env_parse import get_bool_env

RUNNAME = os.environ.get("RUNNAME")
DEBUG_MODE = get_bool_env("DEBUG_MODE")


@dataclass
class DatasetArguments(BaseDatasetArguments):
    dataset_type: Optional[str] = field(default="lerobot")
    data_root_dir: Optional[List[str]] = field(
        default_factory=lambda: [
            "/home/zimu.gong/.cache/huggingface/lerobot/zimgong/lerobot_lift_visual_processed_280",
        ],
    )
    transforms: Optional[List[str]] = field(default_factory=lambda: [dict(type="Normalize")])


@dataclass
class GOModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default="agibot-world/GO-1-Air")
    freeze_llm: bool = field(default=False if not DEBUG_MODE else True)
    freeze_backbone: bool = field(default=False if not DEBUG_MODE else True)
    freeze_mlp: bool = field(default=False if not DEBUG_MODE else True)
    action_chunk_size: int = field(default=15)
    latent_planning: bool = field(default=False)


@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    overwrite_output_dir: bool = field(default=True)
    dataloader_num_workers: int = field(default=64 if not DEBUG_MODE else 0)
    bf16: bool = field(default=True)
    num_train_epochs: float = field(default=2000.0)
    per_device_train_batch_size: int = field(default=64 if not DEBUG_MODE else 2)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    do_train: bool = field(default=True)
    deepspeed: str = field(default="go1/zero_stage1_config.json")

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=5000)
    save_total_limit: int = field(default=100)
    logging_steps: int = field(default=10)
    report_to: str = field(default="wandb")


@dataclass
class SpaceArguments(BaseSpaceArguments):
    state_dim: int = field(default=17)
    action_dim: int = field(default=6)
    space_repack: dict = field(
        default_factory=lambda: {
            "state": "observation.state",
            "action": "action",
            "cam_hand_left_color": "observation.images.image_global",
            "final_prompt": "task",
        }
    )
    ctrl_freq: int = field(default=20)
