import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
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
    # 设置策略以使用所有可用的GPU
    training_args_dict = {
        "local_rank": -1,  # 禁用DDP
        "deepspeed": None,  # 禁用Deepspeed
        "parallel_mode": "not_parallel",  # 禁用并行模式
    }

    # 解析参数
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoraArguments,
        CustomTrainingArguments
    ))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # 更新训练参数
    for key, value in training_args_dict.items():
        setattr(training_args, key, value)

    # 加载模型和tokenizer
    model_dir = snapshot_download(model_args.model_name_or_path)

    # 配置tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 配置模型加载参数
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "device_map": "balanced",  # 平衡分配到所有GPU
    }

    # 设置量化参数
    if model_args.quantization_bit == 4:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif model_args.quantization_bit == 8:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        **model_kwargs
    )
    model.config.use_cache = False

    # 准备量化训练
    if model_args.quantization_bit in [4, 8]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    # 配置LoRA
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

    # 设置训练参数
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

    # **新增部分：随机抽取1/100的训练集**
    # 通过train_test_split方法将训练集划分为1%的子集和剩余的99%（这里只保留1%的子集）
    train_subset = train_dataset.train_test_split(train_size=0.01, seed=42)['train']
    train_dataset = train_subset
    # **新增部分结束**

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
