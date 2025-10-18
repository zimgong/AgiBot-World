from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class BaseDatasetArguments:
    dataset_type: Optional[str] = field(
        default="lerobot",
        metadata={"help": "Type of dataset. Only 'lerobot' is supported."},
    )
    data_root_dir: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "Path of the LeRobot dataset. Multiple paths can be provided."},
    )
    transforms: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "List of transformations to apply to the dataset. Currently only 'Normalize' is supported."},
    )


@dataclass
class BaseModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."},
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."},
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."},
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the LLM. Default is False."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the ViT. Default is False."},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the MLP. Default is False."},
    )
    output_logits: bool = field(
        default=False,
        metadata={"help": "Set to True to output logits. Default is False."},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={"help": "Specify the layer of ViT feature map to use. Default is -1 for the last layer."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing. Default is False."},
    )
    drop_path_rate: float = field(
        default=0.1,
        metadata={"help": "Set the drop path rate for the ViT. Default is 0.1"},
    )
    ps_version: Literal["v1", "v2"] = field(
        default="v2",
        metadata={"help": "Specify the version of pixel shuffle implementation. Default is v2."},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Set to True to use the fast mode of the tokenizer."},
    )
    # Latent Planner Config
    latent_planning: bool = field(default=False, metadata={"help": "if to predict latent action."})
    freeze_latent_planner: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the Latent Planner. Default is False."},
    )
    # DDPM Config
    num_train_timesteps: int = field(default=1000)
    num_inference_timesteps: int = field(default=5)
    prediction_type: str = field(default="sample")
    clip_sample: bool = field(default=False)
    beta_schedule: str = field(default="squaredcos_cap_v2")
    max_seq_length: int = field(
        default=4096,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    force_image_size: int = field(default=448, metadata={"help": "Set the desired size for the image. Default is 448."})
    down_sample_ratio: float = field(
        default=0.5, metadata={"help": "Set the desired down-sampling ratio for the image. Default is 0.5."}
    )
    pad2square: bool = field(
        default=False, metadata={"help": "Pad the image to a square shape if set to True. Default is False."}
    )
    conv_style: str = field(default="internlm2-chat", metadata={"help": "Prompt style for a conversation."})
    dynamic_image_size: bool = field(
        default=False, metadata={"help": "Set to True to use dynamic high resolution strategy. Default is False."}
    )
    use_thumbnail: bool = field(
        default=False, metadata={"help": "Set to True to add a thumbnail image. Default is False."}
    )
    min_dynamic_patch: Optional[int] = field(
        default=1, metadata={"help": "The minimum number of dynamic patches. Default is 1."}
    )
    max_dynamic_patch: Optional[int] = field(
        default=6, metadata={"help": "The maximum number of dynamic patches. Default is 6."}
    )
    normalize_type: Literal["imagenet", "clip", "siglip"] = field(
        default="imagenet", metadata={"help": "The normalization type for the image. Default is imagenet."}
    )
    action_chunk_size: int = field(default=30, metadata={"help": "The size of action chunks. Default is 30."})
    
    # MeanFlow decoder configuration
    decoder_type: Literal["DDPM", "mean_flow"] = field(
        default="DDPM", 
        metadata={"help": "Decoder type: 'DDPM' for diffusion or 'mean_flow' for MeanFlow. Default is 'DDPM'."}
    )
    dispersive_use: bool = field(
        default=False, 
        metadata={"help": "Whether to use dispersive regularization for MeanFlow. Default is False."}
    )
    dispersive_weight: float = field(
        default=0.5, 
        metadata={"help": "Weight for dispersive regularization. Default is 0.5."}
    )
    dispersive_loss_type: Literal["infonce_cos", "hinge", "cov"] = field(
        default="infonce_cos", 
        metadata={"help": "Type of dispersive loss: 'infonce_cos', 'hinge', or 'cov'. Default is 'infonce_cos'."}
    )
    dispersive_temperature: float = field(
        default=0.3, 
        metadata={"help": "Temperature for InfoNCE cosine dispersive loss. Default is 0.3."}
    )
    dispersive_margin: float = field(
        default=0.5, 
        metadata={"help": "Margin for hinge dispersive loss. Default is 0.5."}
    )
    dispersive_target: List[str] = field(
        default_factory=lambda: ["T", "R", "Cond"], 
        metadata={"help": "Target embeddings for dispersive regularization. Default is ['T', 'R', 'Cond']."}
    )


@dataclass
class BaseSpaceArguments:
    """
    Arguments for obs / action space.
    """

    state_dim: int = field(default=8)
    action_dim: int = field(default=7)
    space_repack: dict = field(
        default_factory=lambda: {
            "state": "state",
            "action": "action",
            "cam_head_color": "cam_head_color",
            "cam_hand_left_color": "cam_hand_left_color",
            "cam_hand_right_color": "cam_hand_right_color",
            "final_prompt": "final_prompt",
        }
    )
    default_prompt: Optional[str] = None
    ctrl_freq: int = field(default=30)
