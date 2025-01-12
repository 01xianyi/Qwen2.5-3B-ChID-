from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments as TransformersTrainingArguments
)
import transformers
import torch
import os
import subprocess

# 从 config_danka 引入 DeepSpeedConfig（如果需要动态生成）
from config_danka import DeepSpeedConfig

# 这里导入我们自定义的 Trainer 逻辑
from trainer import ModelTrainer

from typing import Optional, List
@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="/root/.cache/modelscope/hub/Qwen/Qwen2.5-3B",
        metadata={"help": "模型本地路径"}
    )
    quantization_bit: Optional[int] = field(
        default=8,
        metadata={"help": "量化位数，支持4/8位量化"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "是否使用flash attention"}
    )
    # LoRA 动态配置
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout概率"}
    )


@dataclass
class DataArguments:
    """数据相关参数（路径写死示例，仅单轮对话）"""
    # 这里将路径写死，如果你想改成命令行可指定，也可改为 field(default=...)。
    train_file: str = field(
        default="/root/data/ChID/processed/train.json",
        metadata={"help": "训练数据路径（已写死）"}
    )
    eval_file: str = field(
        default="/root/data/ChID/processed/dev.json",
        metadata={"help": "验证数据路径（已写死）"}
    )


@dataclass
class TrainingArguments(TransformersTrainingArguments):
    """训练相关参数"""
    output_dir: str = field(default="./output")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate: float = field(default=5e-4)
    save_steps: int = field(default=500)
    logging_steps: int = field(default=50)
    save_total_limit: int = field(default=2)
    fp16: bool = field(default=True)
    max_grad_norm: float = field(default=1.0, metadata={"help": "梯度裁剪的最大范数"})
    # 启用 TensorBoard 日志记录
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    logging_dir: str = field(default="./logs")
    deepspeed: Optional[str] = field(
        default="./deepspeed_config.json",
        metadata={"help": "DeepSpeed配置文件路径（写死示例）"}
    )




def create_model_and_tokenizer(model_args: ModelArguments):
    """创建模型和tokenizer"""
    from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, get_peft_model

    # 检查本地路径
    if not os.path.exists(model_args.model_name_or_path):
        raise ValueError(f"模型路径不存在: {model_args.model_name_or_path}")

    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True
    }

    # 设置量化参数
    if model_args.quantization_bit in [4, 8]:
        # 在4bit和8bit下使用 bitsandbytes
        load_in_4bit = (model_args.quantization_bit == 4)
        load_in_8bit = (model_args.quantization_bit == 8)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None
        )
        kwargs["quantization_config"] = quantization_config
    else:
        kwargs["torch_dtype"] = torch.float16

    print(f"模型加载参数: {kwargs}")

    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        pad_token="<|extra_0|>",
        local_files_only=True
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print("正在加载模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **kwargs
        )
    except ImportError as e:
        print(f"导入错误: {e}")
        print("正在安装必要的依赖...")
        subprocess.run(["pip", "install", "-U", "bitsandbytes"], check=True)
        print("依赖安装完成，重新加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **kwargs
        )

    # 准备模型进行kbit训练
    print("正在准备模型进行kbit训练...")
    model = prepare_model_for_kbit_training(model)

    # 配置LoRA（动态传入）
    print("正在应用LoRA配置...")
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 需要训练的模块
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 将模型转换为PEFT模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数的比例

    return model, tokenizer


def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 打印参数配置
    print("\n=== 训练配置 ===")
    print(f"模型路径: {model_args.model_name_or_path}")
    print(f"量化位数: {model_args.quantization_bit}")
    print(f"LoRA r: {model_args.lora_r}, alpha: {model_args.lora_alpha}, dropout: {model_args.lora_dropout}")
    print(f"训练数据: {data_args.train_file}")
    print(f"验证数据: {data_args.eval_file}")
    print(f"训练轮数: {training_args.num_train_epochs}")
    print(f"批次大小: {training_args.per_device_train_batch_size}")
    print(f"学习率: {training_args.learning_rate}")
    print(f"是否使用DeepSpeed: {'是' if training_args.deepspeed else '否'}")
    print(f"本地进程编号: {training_args.local_rank}")
    print("===============\n")

    # 创建模型和tokenizer
    model, tokenizer = create_model_and_tokenizer(model_args)

    # 如果要生成新的DeepSpeed配置文件，也可以在此调用:
    if training_args.deepspeed:
        ds_config = DeepSpeedConfig()
        ds_config.save_config(training_args.deepspeed)

    # 初始化训练器并开始训练
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_file=data_args.train_file,
        eval_file=data_args.eval_file,
        training_args=training_args
    )
    trainer.train()


if __name__ == "__main__":
    main()
