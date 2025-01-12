from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os

@dataclass
class DeepSpeedConfig:
    """DeepSpeed相关配置"""
    enabled: bool = True
    stage: int = 2
    offload_optimizer: bool = True
    offload_param: bool = False
    overlap_comm: bool = True
    allgather_bucket_size: int = int(5e8)
    reduce_bucket_size: int = int(5e8)
    min_bucket_size: int = 100
    contiguous_gradients: bool = True
    cpu_offload_pin_memory: bool = True
    local_rank: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """转换为DeepSpeed配置字典"""
        return {
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "weight_decay": "auto",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto"
                }
            },
            "fp16": {
                "enabled": True,
                "auto_cast": True,
                "loss_scale": 0,      # 0 表示使用动态损失缩放
                "initial_scale_power": 12,  # 2^12 = 4096 作为初始缩放值
                "loss_scale_window": 2000,
                "hysteresis": 4,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": self.stage,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": self.cpu_offload_pin_memory
                } if self.offload_optimizer else None,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": self.cpu_offload_pin_memory
                } if self.offload_param else None,
                "overlap_comm": self.overlap_comm,
                "allgather_partitions": True,
                "allgather_bucket_size": self.allgather_bucket_size,
                "reduce_scatter": True,
                "reduce_bucket_size": self.reduce_bucket_size,
                "contiguous_gradients": self.contiguous_gradients
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 50,
            "wall_clock_breakdown": False
        }

    def save_config(self, path: str = "deepspeed_config.json"):
        """保存DeepSpeed配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)
