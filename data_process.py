import json
import os


def process_single_example(example):
    """处理单个数据样本"""
    conversations = []
    content = example['content']
    idiom_count = example['realCount']

    # 为每个成语空缺处生成一个训练样本
    for idx in range(idiom_count):
        # 准备当前位置的候选成语和答案
        current_candidates = example['candidates'][idx]
        current_answer = example['groundTruth'][idx]

        # 构建输入文本，处理多个空缺
        temp_content = content
        for i in range(idiom_count):
            if i == idx:
                # 当前需要填写的空缺
                temp_content = temp_content.replace("#idiom#", "<mask>", 1)
            else:
                # 其他空缺用占位符替换
                temp_content = temp_content.replace("#idiom#", "[待填]", 1)

        # 构建训练样本
        prompt = (f"请帮我完成下面的成语填空题：\n\n"
                  f"文段：{temp_content}\n\n"
                  f"候选成语：{', '.join(current_candidates)}\n\n"
                  f"请从候选成语中选择最合适的填入<mask>处。")

        response = f"经过分析，最适合填入<mask>处的成语是'{current_answer}'。"

        sample = {
            "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        }
        conversations.append(sample)

    return conversations


def convert_file(input_path, output_path):
    """转换整个文件"""
    all_samples = []

    # 读取原始数据文件
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 处理每一行数据
    for line in lines:
        if not line.strip():
            continue

        example = json.loads(line.strip())
        samples = process_single_example(example)
        all_samples.extend(samples)

    # 将结果写入输出文件，每个样本占一行
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def process_all_files(input_dir, output_dir):
    """处理所有数据文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理训练集、验证集和测试集
    file_pairs = [
        ('train_data.txt', 'train.json'),
        ('dev_data.txt', 'dev.json'),
        ('test_data.txt', 'test.json')
    ]

    for input_file, output_file in file_pairs:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)

        if os.path.exists(input_path):
            print(f"Processing {input_file}...")
            convert_file(input_path, output_path)
            print(f"Converted {input_file} to {output_file}")


if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = "/root/data/ChID/raw/ChID"
    output_dir = "/root/data/ChID/processed"

    # 执行转换
    process_all_files(input_dir, output_dir)