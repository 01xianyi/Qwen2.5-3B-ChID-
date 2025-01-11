import json
import torch
import os
from typing import Dict, List, Optional, Union, Tuple

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import deepspeed


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        """重写计算损失函数以支持DeepSpeed"""
        # 确保所有输入张量都在正确的设备上
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        # 使用DeepSpeed引擎进行前向传播 (model.module 兼容多卡包裹)
        if hasattr(model, 'module'):
            outputs = model.module(**inputs)
        else:
            outputs = model(**inputs)

        loss = outputs.loss

        # 记录总步数
        self.total_steps += 1

        # 每隔logging_steps步打印一次损失（只对 rank=0 有意义，
        # 但目前此处的 print 会在所有进程执行，如果想完全屏蔽可再做 rank 判断）
        if self.total_steps % self.args.logging_steps == 0:
            print(f"Step {self.total_steps}, Loss: {loss.item():.4f}")

        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """创建优化器和调度器"""
        # 当使用DeepSpeed时，优化器会在训练开始时自动创建
        if self.args.deepspeed:
            return
        # 为非DeepSpeed训练创建优化器和调度器
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)


class ModelTrainer:
    def __init__(self, model, tokenizer, train_file, eval_file, training_args: TrainingArguments):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args

        # 训练和验证文件路径
        self.train_file = train_file
        self.eval_file = eval_file

        # DeepSpeed配置文件路径
        ds_config_path = self.training_args.deepspeed

        # 加载数据集
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
        """获取数据整理器"""
        return None  # 使用默认的数据整理器

    def train(self):
        """开始训练"""
        print("正在配置训练器...")

        # 创建训练器
        trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.get_data_collator(),
        )

        # 获取 local_rank (单机多卡场景)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # --------------------------
        # 只在 rank=0 才输出进度条和日志
        # --------------------------
        if local_rank != 0:
            # 禁用进度条
            trainer.args.disable_tqdm = True
            # 如果启用了 tensorboard 等日志，可以改为不报告
            trainer.args.report_to = ["none"]

        # 打印训练配置信息 (可选，也可只在 rank=0 打印)
        if local_rank == 0:
            print("\n=== 训练配置 ===")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"批次大小 (每个设备): {self.training_args.per_device_train_batch_size}")
            print(f"总批次大小: {self.training_args.per_device_train_batch_size * torch.cuda.device_count()}")
            print(f"梯度累积步数: {self.training_args.gradient_accumulation_steps}")
            print(f"有效批次大小: {self.training_args.per_device_train_batch_size * torch.cuda.device_count() * self.training_args.gradient_accumulation_steps}")
            print(f"训练轮数: {self.training_args.num_train_epochs}")
            print(f"学习率: {self.training_args.learning_rate}")
            print(f"是否使用DeepSpeed: {'是' if self.training_args.deepspeed else '否'}")
            print(f"是否使用FP16: {'是' if self.training_args.fp16 else '否'}")
            print("===============\n")

        print("开始训练...")
        train_result = trainer.train()

        # --------------------------
        # 只在 rank=0 上保存模型和日志
        # --------------------------
        if local_rank == 0:
            print("训练完成，保存最终模型...")
            trainer.save_model()

            # 保存训练状态
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        return train_result
