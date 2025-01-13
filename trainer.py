import json
import torch
import os
from typing import Dict, List, Optional, Union, Tuple

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import deepspeed

from torch import nn

class ChIDDataset(Dataset):
    def __init__(self, file_path: str, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer

        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = item['text']

        # 分割用户输入和助手回答（单轮对话）
        parts = text.split('<|im_start|>assistant\n')
        user_text = parts[0]
        assistant_text = parts[1].split('<|im_end|>')[0]

        # 构建完整的输入序列
        full_text = text

        # 编码文本
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors="pt"
        )

        # 获取输入ID和注意力掩码
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]

        # 创建标签，将非助手回答部分的标签设为-100
        labels = input_ids.clone()

        # 找到助手回答的起始位置
        assistant_start = text.find('<|im_start|>assistant\n')
        prefix_encoding = self.tokenizer(
            text[:assistant_start],
            truncation=True,
            max_length=512
        )
        prefix_length = len(prefix_encoding['input_ids'])

        # 将非助手回答部分的标签设为-100
        labels[:prefix_length] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def get_example_text(self, idx):
        """用于调试：获取原始文本"""
        return self.examples[idx]['text']


class CustomTrainer(Trainer):
    """
    自定义 Trainer，仅重写 compute_loss 方法。
    不再手动管理 AMP/GradScaler。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = 0  # 用于打印日志等

    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失函数（可在此添加自定义逻辑）"""
        outputs = model(**inputs)
        loss = outputs.loss

        self.total_steps += 1
        if self.total_steps % self.args.logging_steps == 0:
            print(f"Step {self.total_steps}, Loss: {loss.item():.4f}")

        return (loss, outputs) if return_outputs else loss

    # 移除 training_step 和 optimizer_step 的自定义实现
    # 让 Trainer 内部处理这些步骤，包括 AMP 和梯度缩放


class ModelTrainer:
    def __init__(self, model, tokenizer, train_file, eval_file, training_args: TrainingArguments):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.train_file = train_file
        self.eval_file = eval_file

        print("正在加载训练数据...")
        self.train_dataset = ChIDDataset(self.train_file, tokenizer)
        print(f"训练集大小: {len(self.train_dataset)}")

        if self.eval_file:
            print("正在加载验证数据...")
            self.eval_dataset = ChIDDataset(self.eval_file, tokenizer)
            print(f"验证集大小: {len(self.eval_dataset)}")
        else:
            self.eval_dataset = None

    def get_data_collator(self):
        return None

    def train(self):
        """开始训练"""
        print("正在配置训练器...")

        trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.get_data_collator(),
        )

        print("\n=== 训练配置 ===")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"批次大小: {self.training_args.per_device_train_batch_size}")
        print(f"梯度累积步数: {self.training_args.gradient_accumulation_steps}")
        print(f"有效批次大小: {self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps}")
        print(f"训练轮数: {self.training_args.num_train_epochs}")
        print(f"学习率: {self.training_args.learning_rate}")
        print(f"FP16: {'开启' if self.training_args.fp16 else '关闭'}")
        print("===============\n")

        print("开始训练...")
        train_result = trainer.train()

        print("训练完成，保存最终模型...")
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return train_result
