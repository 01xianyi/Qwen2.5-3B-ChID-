import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,  # 更改这里
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from modelscope import snapshot_download
from config import ModelArguments, DataArguments, LoraArguments


def load_and_process_dataset(data_path):
    """加载并处理数据集"""
    dataset = load_dataset('json', data_files=data_path)
    return dataset['train']


def tokenize_function(examples, tokenizer, max_length=512):
    """标记化函数，将文本转换为模型输入"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )


def create_model_and_tokenizer(model_args):
    """创建模型和 tokenizer"""
    # 下载模型
    model_dir = snapshot_download(model_args.model_name_or_path)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 基本模型加载参数
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "device_map": "auto",
    }

    # 根据量化设置添加相应参数
    if model_args.quantization_bit == 8:
        model_kwargs["load_in_8bit"] = True
    elif model_args.quantization_bit == 4:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    else:
        model_kwargs["torch_dtype"] = torch.float16

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        **model_kwargs
    )

    # 禁用缓存以支持梯度检查点
    model.config.use_cache = False

    return model, tokenizer


def create_peft_config(lora_args):
    """创建 LoRA 配置"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias="none",  # 量化训练时的推荐设置
    )


def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # 创建模型和 tokenizer
    model, tokenizer = create_model_and_tokenizer(model_args)

    # 加载数据集
    train_dataset = load_and_process_dataset(data_args.train_file)
    eval_dataset = load_and_process_dataset(data_args.eval_file)

    # 标记化数据集
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # 如果使用了量化，需要准备模型
    if model_args.quantization_bit in [4, 8]:
        model = prepare_model_for_kbit_training(model)

    # 创建 LoRA 配置并准备模型
    peft_config = create_peft_config(lora_args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(  # 更改这里
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型不使用 MLM
    )

    # 设置 TrainingArguments 的 remove_unused_columns 参数
    training_args.remove_unused_columns = False

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # 此处暂时保留，但注意 FutureWarning
    )

    # 验证是否使用了多个 GPU
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using them for training.")
    else:
        print("Single GPU detected.")

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存模型
    output_dir = training_args.output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
