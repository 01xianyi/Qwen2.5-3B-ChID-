import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from tqdm import tqdm
import os
import re
import sys
from typing import List, Dict
from rich.live import Live
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout

console = Console()


def create_progress_table(accuracy: float = 0.0, processed: int = 0, total: int = 0):
    """创建进度显示面板"""
    layout = Layout()
    progress_text = f"当前准确率: {accuracy:.4f} [{processed}/{total}]"
    return Panel(progress_text, title="评估进度", border_style="cyan")


def evaluate_model(model, tokenizer, test_file: str) -> Dict:
    """评估模型在测试集上的表现"""
    correct = 0
    total = 0
    results = []

    # 首先计算总行数
    with open(test_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())

    # 使用rich.live来创建实时更新的显示
    with Live(create_progress_table(0.0, 0, total_lines), refresh_per_second=4) as live:
        with torch.no_grad():
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    text = data['text']

                    # 获取真实答案
                    answer_parts = text.split('经过分析，最适合填入<mask>处的成语是')
                    if len(answer_parts) < 2:
                        continue
                    true_answer = extract_answer(answer_parts[1])

                    # 获取用户问题部分
                    question = answer_parts[0] + "经过分析，最适合填入<mask>处的成语是"

                    # 生成答案
                    inputs = tokenizer(question, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=4,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1
                    )

                    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_answer = extract_answer(predicted_text)

                    # 记录结果
                    is_correct = (predicted_answer == true_answer)
                    correct += int(is_correct)
                    total += 1

                    # 更新进度显示
                    current_accuracy = correct / total
                    live.update(create_progress_table(current_accuracy, total, total_lines))

                    results.append({
                        'question': text,
                        'true_answer': true_answer,
                        'predicted_answer': predicted_answer,
                        'is_correct': is_correct
                    })

    # 计算最终准确率
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }

    return metrics, results


def load_model_and_tokenizer(base_model_path: str, adapter_path: str):
    """加载模型和tokenizer"""
    console.print("[yellow]正在加载tokenizer...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        pad_token="<|extra_0|>",
        local_files_only=True
    )

    console.print("[yellow]正在加载基础模型...[/yellow]")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    console.print("[yellow]正在加载LoRA权重...[/yellow]")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    console.print("[bold green]✓[/bold green] 模型加载完成！")
    return model, tokenizer


def extract_answer(text: str) -> str:
    """从回答中提取成语"""
    match = re.search(r"'(.*?)'", text)
    if match:
        return match.group(1)
    return ""


def main():
    # 模型路径配置
    base_model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2.5-3B"
    adapter_path = "./output/checkpoint-2000"
    test_file = "/root/data/ChID/processed/sampled/eval_sampled.json"

    # 创建输出目录
    output_dir = "./evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    console.print("\n[bold cyan]成语填空模型评估[/bold cyan]")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(base_model_path, adapter_path)

    # 执行评估
    console.print("\n[bold cyan]开始评估...[/bold cyan]")
    metrics, results = evaluate_model(model, tokenizer, test_file)

    # 打印评估结果
    console.print(Panel.fit(
        f"[green]准确率:[/green] {metrics['accuracy']:.4f}\n"
        f"[green]正确预测数:[/green] {metrics['correct_predictions']}\n"
        f"[green]总样本数:[/green] {metrics['total_samples']}",
        title="评估结果",
        border_style="green"
    ))

    # 保存详细结果
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]✓[/bold green] 详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()