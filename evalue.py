#python  evalue.py --adapter_path ./output/checkpoint-20280 --base_model_path /root/.cache/modelscope/hub/Qwen/Qwen2.5-3B --eval_file /root/data/ChID/processed/test.json 
import os
import json
import torch
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)
from peft import PeftModel
from tqdm import tqdm
from typing import List, Dict


@dataclass
class EvalArguments:
    """评估相关参数"""
    base_model_path: str = field(
        default="/root/.cache/modelscope/hub/Qwen/Qwen2.5-3B",
        metadata={"help": "基座模型路径"}
    )
    adapter_path: str = field(
        default="./output/checkpoint-20280",
        metadata={"help": "adapter检查点路径"}
    )
    eval_file: str = field(
        default="/root/data/ChID/processed/test.json",
        metadata={"help": "测试数据路径"}
    )
    batch_size: int = field(
        default=322,
        metadata={"help": "评估时的批量大小"}
    )
    max_new_tokens: int = field(
        default=64,
        metadata={"help": "生成的最大新tokens数"}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "生成时的束搜索数"}
    )


def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    print(f"正在加载基座模型: {args.base_model_path}")

    try:
        # 使用默认配置加载模型
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        print("基座模型加载完成。")
    except Exception as e:
        print(f"加载基座模型时出错: {str(e)}")
        raise e

    try:
        # 加载adapter权重
        print(f"正在加载adapter权重: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()
        print("adapter权重加载完成。")
    except Exception as e:
        print(f"加载adapter权重时出错: {str(e)}")
        raise e

    try:
        # 加载tokenizer
        print("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        print("tokenizer加载完成。")
    except Exception as e:
        print(f"加载tokenizer时出错: {str(e)}")
        raise e

    # 确保有padding token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        print("pad_token未定义，已设置为eos_token。")

    # 获取设备信息
    device = next(model.parameters()).device
    print(f"模型所在设备: {device}")

    return model, tokenizer


def process_batch(texts: List[str], tokenizer, model, args) -> List[Dict]:
    """处理一个批次的样本"""
    try:
        # 构造输入
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )

        # 获取设备信息
        device = next(model.parameters()).device

        # 将输入移到正确的设备上
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成回答
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解码输出
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

        results = []
        for text, response in zip(texts, responses):
            # 提取用户输入和正确答案
            parts = text.split('<|im_start|>assistant\n')
            if len(parts) < 2:
                print("文本分割后部分不足，跳过此样本。")
                continue
            user_query = parts[0].strip()
            assistant_response = parts[1].split('<|im_end|>')[0].strip()

            # 提取正确答案
            ground_truth = ""
            try:
                if "'" in assistant_response:
                    start_idx = assistant_response.find("'")
                    end_idx = assistant_response.find("'", start_idx + 1)
                    ground_truth = assistant_response[start_idx + 1:end_idx].strip()
                elif '"' in assistant_response:
                    start_idx = assistant_response.find('"')
                    end_idx = assistant_response.find('"', start_idx + 1)
                    ground_truth = assistant_response[start_idx + 1:end_idx].strip()
            except Exception as e:
                print(f"提取 ground_truth 时出错: {str(e)}")
                continue

            if not ground_truth:
                print("未找到 ground_truth，跳过此样本。")
                continue

            # 提取预测答案
            prediction = ""
            try:
                if "'" in response:
                    start_idx = response.find("'")
                    end_idx = response.find("'", start_idx + 1)
                    prediction = response[start_idx + 1:end_idx].strip()
                elif '"' in response:
                    start_idx = response.find('"')
                    end_idx = response.find('"', start_idx + 1)
                    prediction = response[start_idx + 1:end_idx].strip()
            except Exception as e:
                print(f"提取 prediction 时出错: {str(e)}")
                prediction = ""

            results.append({
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": prediction == ground_truth,
                "full_response": response
            })

        return results

    except Exception as e:
        print(f"处理批次时出错: {str(e)}")
        return []


def main():
    # 解析参数
    parser = HfArgumentParser(EvalArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args)

    # 加载评估数据
    print(f"正在加载评估数据: {args.eval_file}")
    eval_data = []
    try:
        with open(args.eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))
        print(f"加载了 {len(eval_data)} 条评估数据")
    except Exception as e:
        print(f"加载评估数据时出错: {str(e)}")
        return

    # 评估
    results = []
    correct_count = 0
    total_count = 0

    batch_size = args.batch_size
    num_batches = (len(eval_data) + batch_size - 1) // batch_size

    for batch_num in tqdm(range(num_batches), desc="Evaluating"):
        batch = eval_data[batch_num * batch_size : (batch_num + 1) * batch_size]
        texts = [item.get('text', '') for item in batch]
        batch_results = process_batch(texts, tokenizer, model, args)

        for result in batch_results:
            results.append(result)
            if result['correct']:
                correct_count += 1
            total_count += 1

        # 实时输出当前准确率
        if total_count > 0:
            current_accuracy = (correct_count / total_count) * 100
            print(f"\n当前准确率: {current_accuracy:.2f}% ({correct_count}/{total_count})")

    # 计算最终指标
    final_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    # 打印结果
    print("\n=== 最终评估结果 ===")
    print(f"准确率: {final_accuracy:.2f}%")
    print(f"正确数量: {correct_count}")
    print(f"总样本数: {total_count}")

    # 保存结果
    output = {
        "metrics": {
            "accuracy": final_accuracy,
            "correct_count": correct_count,
            "total_count": total_count
        },
        "detailed_results": results[:100]  # 根据需要调整保存的详细结果数量
    }

    output_file = os.path.join(args.adapter_path, "eval_results.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"\n评估结果已保存至: {output_file}")
    except Exception as e:
        print(f"保存评估结果时出错: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"脚本运行时发生未捕获的错误: {str(e)}")
