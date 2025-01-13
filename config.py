from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-3B",
        metadata={"help": "预训练模型的路径或标识符"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )
    # 已移除 quantization_bit 参数

@dataclass
class DataArguments:
    train_file: str = field(
        default="/root/data/ChID/processed/train.json",
        metadata={"help": "训练集文件路径"}
    )
    eval_file: str = field(
        default="/root/data/ChID/processed/dev.json",
        metadata={"help": "验证集文件路径"}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "输入序列的最大长度"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=36,
        metadata={"help": "数据预处理的并行工作进程数"}
    )

@dataclass
class LoraArguments:
    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA秩，决定了适配器的参数量和表达能力"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha参数，通常设置为 rank 的 4 倍"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout概率"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        metadata={"help": "需要应用LoRA的模块名称列表"}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "是否使用梯度检查点以节省显存"}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "梯度累积步数"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "训练时每个设备的批次大小"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "评估时每个设备的批次大小"}
    )
    learning_rate: float = field(
        default=5e-4,
        metadata={"help": "初始学习率"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪的最大范数"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "学习率预热的步数比例"}
    )
    logging_steps: int = field(
        default=50,
        metadata={"help": "日志记录的步数间隔"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "模型保存策略，可选 steps 或 epoch"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "每多少步保存一次模型"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "保存的检查点总数限制"}
    )
    fp16: bool = field(
        default=True,  # 修改为 False 以使用完整精度
        metadata={"help": "是否使用混合精度训练"}
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": "是否使用Torch 2.0编译功能"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子，用于复现性"}
    )
    dataloader_num_workers: int = field(
        default=30,
        metadata={"help": "数据加载的并行工作进程数"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "最大训练步数，-1表示按轮数训练"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "是否移除未使用的列"}
    )
