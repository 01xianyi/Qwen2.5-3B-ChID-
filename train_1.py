import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from modelscope import snapshot_download
from config import (
    ModelArguments,
    DataArguments,
    LoraArguments,
    CustomTrainingArguments
)

def load_and_process_dataset(data_path, tokenizer, max_length):
    """加载并处理数据集"""
    dataset = load_dataset('json', data_files=data_path)

    def process_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    processed_dataset = dataset['train'].map(
        process_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Processing dataset",
    )

    return processed_dataset

def main():
    # 设置环境变量以禁用 Tokenizers 并行化警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    # 解析参数
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoraArguments,
        CustomTrainingArguments
    ))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # 加载模型目录
    model_dir = snapshot_download(model_args.model_name_or_path)

    # 配置 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 配置模型加载参数，使用 torch.float32 全精度
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": torch.float32
    }

    # 打印 model_kwargs 进行调试
    print("Model kwargs:", model_kwargs)

    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            **model_kwargs
        )
    except ValueError as e:
        print(f"Error loading model: {e}")
        raise e

    model.config.use_cache = False

    # 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # 设置训练参数，确保只有 LoRA 层的参数被训练
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 打印可训练参数信息
    model.print_trainable_parameters()

    # 处理数据集
    train_dataset = load_and_process_dataset(
        data_args.train_file,
        tokenizer,
        data_args.max_length
    )

    # 随机抽取1/100的训练集
    train_subset = train_dataset.train_test_split(train_size=0.01, seed=42)['train']
    train_dataset = train_subset

    eval_dataset = load_and_process_dataset(
        data_args.eval_file,
        tokenizer,
        data_args.max_length
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存模型
    trainer.save_model()
    trainer.save_state()

    print(f"Training completed. Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()
