import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from json_repair import repair_json
from transformers import AutoConfig
from time import time

# ===== WSL 专属内存优化 =====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_DISABLE_RICH"] = "true"
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# ===== CPU Offload 配置 (关键!) =====
OFFLOAD_DIR = Path("/mnt/d/offload_cache")  # ⚠️ 必须使用WSL可访问的磁盘目录 (非/tmp!)
OFFLOAD_DIR.mkdir(exist_ok=True, parents=True)


# 配置
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # 正式模型ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 更激进但更高效的 GPU 内存分配 ===
if DEVICE == "cuda":
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # 保守只留 0.8~1.0GB 给系统（WSL 驱动开销较小）
    gpu_mem_gb = max(1, int(total_mem_gb - 0.8))  # 16GB GPU → 15GB
    GPU_MEM = f"{gpu_mem_gb}GB"
else:
    GPU_MEM = "0GB"

# === 智能内存分配策略 ===
MAX_MEMORY = {
    0: GPU_MEM,      # 尽可能使用 GPU（自动检测或你可手动设为 "20GB" 等）
    "cpu": "64GB",   # 当 GPU 不足时，用 CPU 承载剩余层
    # "disk": "32GB" # 一般不建议启用 disk offload（极慢），除非内存也爆了
}

print("DEVICE: ",DEVICE)

print(f"🚀 启动 DeepSeek-R1-0528-Qwen3-8B | Offload目录: {OFFLOAD_DIR}")
print(f"🧠 内存策略: GPU={MAX_MEMORY[0]}, CPU={MAX_MEMORY['cpu']}")

# ===== 智能加载模型 (带 offload) =====
print(f"🚀 启动 {MODEL_NAME}")
print(f"🧠 内存策略: GPU={MAX_MEMORY[0]}, CPU={MAX_MEMORY['cpu']}")


from transformers import BitsAndBytesConfig



# === 加载 Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("🔧 修复模型配置...")
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if hasattr(config, 'rope_scaling') and config.rope_scaling:
    if config.rope_scaling.get('rope_type') == 'yarn':
        config.rope_scaling.pop('attn_factor', None)
        config.rope_scaling['factor'] = 4.0
        config.rope_scaling['original_max_position_embeddings'] = 32768

print("🚀 加载 8-bit 模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,  # 使用修复后的配置
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
    ),
    device_map="cuda:0",  # 强制全 GPU
    trust_remote_code=True,
    # use_cache=True,
    attn_implementation="flash_attention_2",  # 显式启用
)

# 加速策略
model = torch.compile(model, mode="reduce-overhead")

# 创建 pipeline (无 device 参数)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0,  # 显式指定 GPU
    framework="pt",
    batch_size=1,
    # 生成参数
    max_new_tokens=128,
    temperature=0.2,
    top_p=0.85,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)



def extract_causal_relations(text: str, max_retries=2):
    """带 CPU offload 优化的因果推断"""

    FEW_SHOT_EXAMPLES = [
        # 示例1: 经济政策影响 (多实体)
        {
            "input": "央行宣布加息0.5%，导致抵押贷款利率上升，同时储蓄账户收益增加。",
            "output": [
                {
                    "event": "央行加息0.5%",
                    "entity": "抵押贷款利率",
                    "effect": "负面：上升"
                },
                {
                    "event": "央行加息0.5%",
                    "entity": "储蓄账户收益",
                    "effect": "正面：增加"
                }
            ]
        },

        # 示例2: 医疗干预效果 (隐性因果)
        {
            "input": "临床试验显示，每日服用维生素D补充剂6个月后，参与者骨折风险降低了22%。",
            "output": [
                {
                    "event": "每日服用维生素D补充剂6个月",
                    "entity": "骨折风险",
                    "effect": "正面：降低22%"
                }
            ]
        },

        # 示例3: 无明确因果关系 (边界情况)
        {
            "input": "会议将于下周三举行，地点在总部大楼3楼会议室。",
            "output": []
        },

        # 示例4: 复杂链式因果 (高级模式)
        {
            "input": "供应链中断引发芯片短缺，迫使汽车制造商减产，进而导致二手车价格上涨30%。",
            "output": [
                {
                    "event": "供应链中断",
                    "entity": "芯片供应",
                    "effect": "负面：短缺"
                },
                {
                    "event": "芯片短缺",
                    "entity": "汽车产量",
                    "effect": "负面：减产"
                },
                {
                    "event": "汽车减产",
                    "entity": "二手车价格",
                    "effect": "负面：上涨30%"
                }
            ]
        }
    ]

    # 构建 prompt (同前文)
    system_prompt = "你是一个因果推断专家系统...\n### 参考示例:\n"
    for ex in FEW_SHOT_EXAMPLES:
        system_prompt += f"输入: \"{ex['input']}\"\n输出: {json.dumps(ex['output'])}\n"

    user_prompt = f"### 待分析文本\n\"\"\"{text}\"\"\"\n### 严格JSON输出:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    # Offload 专用重试机制
    for attempt in range(max_retries + 1):
        try:
            # 关键: 清理缓存避免 offload 内存泄漏
            torch.cuda.empty_cache()
            if attempt > 0:
                print(f"  ♻️  重试 #{attempt} (清理缓存后)")

            start_time = time()
            response = text_generator(
                messages,  # ← 直接传入消息
                return_full_text=False
            )[0]['generated_text']
            duration = time() - start_time

            # 速度监控
            gen_tokens = len(tokenizer.encode(response))
            print(f"⚡ 速度: {gen_tokens / duration:.1f} token/s ({gen_tokens} tokens in {duration:.2f}s)")

            # 修复 JSON
            repaired = repair_json(response, return_objects=True)
            return _validate_results(repaired)

        except Exception as e:
            print(f"  ❌  处理失败 (尝试 {attempt + 1}): {str(e)}")
            torch.cuda.empty_cache()

    print("  🛑  所有重试失败，返回空结果")
    return []


def _validate_results(results):
    """验证结果结构 (offload 时更严格)"""
    if not isinstance(results, list):
        return []

    valid_items = []
    for item in results:
        if not (isinstance(item, dict) and
                all(k in item for k in ['event', 'entity', 'effect'])):
            continue

        # 确保 effect 包含极性
        if "正面" not in item['effect'] and "负面" not in item['effect']:
            item['effect'] = f"中性：{item['effect']}"

        valid_items.append(item)
    return valid_items


# ===== 智能监控工具 =====
def monitor_resources():
    """实时监控 WSL 资源使用"""
    print("\n" + "=" * 50)
    print("📊 资源监控报告")
    print("-" * 50)

    # GPU 显存
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU 显存: 已分配 {allocated:.1f}GB / 保留 {reserved:.1f}GB")

    # CPU 内存
    import psutil
    mem = psutil.virtual_memory()
    print(f"CPU 内存: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB ({mem.percent}%)")

    # Offload 目录大小
    offload_size = sum(f.stat().st_size for f in OFFLOAD_DIR.glob('**/*') if f.is_file()) / 1e9
    print(f"Offload 空间: {offload_size:.1f}GB (目录: {OFFLOAD_DIR})")

    # 磁盘空间
    disk = psutil.disk_usage(str(OFFLOAD_DIR))
    print(f"磁盘空间: {disk.used / 1e9:.1f}GB / {disk.total / 1e9:.1f}GB ({disk.percent}%)")
    print("=" * 50)


# ===== 测试运行 =====
if __name__ == "__main__":
    monitor_resources()

    sample_text = (
        "全球气候变暖导致北极冰盖融化加速，引发海平面上升威胁沿海城市。"
        "同时，可再生能源投资增长降低了太阳能板成本，推动了绿色技术普及。"
    )

    print("\n🔍 分析文本:", sample_text)
    results = extract_causal_relations(sample_text)

    print("\n✅ 提取结果:")
    for i, rel in enumerate(results, 1):
        print(f"\n{i}. 事件: {rel['event']}")
        print(f"   实体: {rel['entity']}")
        print(f"   影响: {rel['effect']}")

    # 保存结果
    with open("deepseek_causal_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果保存至: deepseek_causal_results.json")

    monitor_resources()