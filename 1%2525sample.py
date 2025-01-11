import json
import random
from typing import List, Tuple
import os


def load_json_lines(file_path: str) -> List[dict]:
    """加载jsonl格式的数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_json_lines(data: List[dict], file_path: str):
    """保存数据为jsonl格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def sample_data(data: List[dict], train_ratio: float, eval_ratio: float) -> Tuple[List[dict], List[dict]]:
    """随机抽样数据集

    Args:
        data: 原始数据列表
        train_ratio: 训练集占比
        eval_ratio: 评估集占比

    Returns:
        训练集和评估集的元组
    """
    # 打乱数据
    random.shuffle(data)

    # 计算样本数量
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    eval_size = int(total_samples * eval_ratio)

    # 抽取样本
    train_data = data[:train_size]
    eval_data = data[train_size:train_size + eval_size]

    return train_data, eval_data


def main():
    # 设置随机种子确保可重复性
    random.seed(42)

    # 配置路径
    input_path = "/root/data/ChID/processed/train.json"
    output_dir = "/root/data/ChID/processed/sampled"
    os.makedirs(output_dir, exist_ok=True)

    # 设置采样比例
    train_ratio = 0.01  # 1%的数据用于训练
    eval_ratio = 0.002  # 0.2%的数据用于评估

    print(f"正在加载数据: {input_path}")
    data = load_json_lines(input_path)
    print(f"总样本数: {len(data)}")

    # 进行采样
    train_data, eval_data = sample_data(data, train_ratio, eval_ratio)
    print(f"采样后训练集样本数: {len(train_data)}")
    print(f"采样后评估集样本数: {len(eval_data)}")

    # 保存采样后的数据
    train_output = os.path.join(output_dir, "train_sampled.json")
    eval_output = os.path.join(output_dir, "eval_sampled.json")

    save_json_lines(train_data, train_output)
    save_json_lines(eval_data, eval_output)

    print(f"采样后的训练集已保存到: {train_output}")
    print(f"采样后的评估集已保存到: {eval_output}")


if __name__ == "__main__":
    main()