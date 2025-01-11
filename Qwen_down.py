import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

# 模型路径
model_name = "Qwen/Qwen2.5-7B"

# 使用 snapshot_download 来解决国内下载问题
model_dir = snapshot_download(model_name)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto"
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 你可以在此处进行一些操作，例如测试模型加载
print(f"Model loaded from {model_dir}")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

# 模型路径，改为 base 版本
model_name = "Qwen/Qwen2.5-3B"

# 使用 snapshot_download 来解决国内下载问题
model_dir = snapshot_download(model_name)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # 使用 float16 来减少内存占用
    device_map="auto",
    trust_remote_code=True      # Qwen 模型需要这个参数
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True      # Qwen tokenizer 也需要这个参数
)

# 打印确认信息
print(f"模型已从 {model_dir} 加载完成")