#!/usr/bin/env python3
"""
DeepSeek-R1-0528-Qwen3-8B 因果推断优化版
✅ 修复 YARN RoPE 兼容性问题
✅ 启用 Flash Attention 2
✅ 8-bit 量化 + 全 GPU 推理
✅ WSL2 专属性能优化
✅ 实时速度监控 (token/s)
✅ 内存泄漏防护

环境要求:
- Python 3.12+
- torch 2.3.0+cu121 (官方版本)
- transformers>=4.45.0
- flash-attn>=2.5.0
- bitsandbytes
- json-repair
- psutil
"""

import os
import json
import torch
import time
from pathlib import Path
from typing import List, Dict, Any, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig
)
from json_repair import repair_json
import psutil

# ===== WSL2 专属性能优化 =====
os.environ.update({
    "CUDA_LAUNCH_BLOCKING": "0",  # 禁用同步调试
    "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync",  # 异步内存分配
    "TF_ENABLE_ONEDNN_OPTS": "0",  # 禁用 TensorFlow 冲突
    "TOKENIZERS_PARALLELISM": "false",  # 避免 tokenizer 冲突
    "ACCELERATE_DISABLE_RICH": "true",  # 禁用 rich 输出
    "BITSANDBYTES_NOWELCOME": "1",  # 禁用 bitsandbytes 欢迎消息
    "PYTORCH_NO_CUDA_MEMORY_CACHING": "0"  # 启用缓存
})

# PyTorch 性能优化
torch.backends.cuda.enable_flash_sdp(True)  # 启用 Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(False)  # 禁用低效模式
torch.backends.cuda.enable_math_sdp(False)  # 禁用数学模式
torch.backends.cudnn.benchmark = True  # 启用 cuDNN benchmark
torch.set_float32_matmul_precision('high')  # 启用 TF32 加速
torch.cuda.empty_cache()  # 启动时清理缓存

# ===== 配置参数 =====
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
OFFLOAD_DIR = Path("/mnt/d/offload_cache")  # WSL2 可访问的磁盘目录
OFFLOAD_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 系统初始化 | 设备: {DEVICE} | Offload目录: {OFFLOAD_DIR}")
print(f"🔧 PyTorch 版本: {torch.__version__} | CUDA 版本: {torch.version.cuda}")


# ===== 三重修复：安全加载模型 =====
def load_model_safely(model_name: str) -> AutoModelForCausalLM:
    """安全加载模型，解决 YARN RoPE 兼容性问题"""
    print("\n" + "=" * 60)
    print("🔧 三重修复：安全加载模型")
    print("-" * 60)

    # 第一重修复：创建兼容的配置
    print("✅ 第一重修复: 创建兼容配置...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # 深度修复 YARN RoPE 配置
    if hasattr(config, 'rope_scaling') and config.rope_scaling:
        print("  🛠️  深度修复 YARN RoPE 配置:")
        original_keys = set(config.rope_scaling.keys())

        # 创建标准兼容配置
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,  # 4倍上下文扩展
            "original_max_position_embeddings": 32768,  # 原始最大长度
            "beta_fast": 32,
            "beta_slow": 1
        }

        removed_keys = original_keys - set(config.rope_scaling.keys())
        print(f"  🔑  移除不兼容字段: {removed_keys}")
        print(f"  ✅  新配置: {config.rope_scaling}")

    # 第二重修复：8-bit 量化配置
    print("\n✅ 第二重修复: 配置 8-bit 量化...")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4"  # 更高效的量化类型
    )

    # 第三重修复：加载模型
    print("\n✅ 第三重修复: 加载模型 (启用 Flash Attention 2)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quant_config,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # 强制启用
            dtype=torch.bfloat16,
            use_safetensors=True,
            ignore_mismatched_sizes=True,  # 忽略配置不匹配
            low_cpu_mem_usage=True
        )
        print("🎉 三重修复成功！模型加载完成")
        return model

    except Exception as e:
        print(f"❌ 三重修复失败，尝试备用方案: {str(e)}")
        print("🔄 回退到基础配置...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            ignore_mismatched_sizes=True
        )
        print("🟡 备用方案加载成功")
        return model


# ===== 高性能生成函数 =====
def generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128
) -> tuple[str, int, float]:
    """
    高性能文本生成（绕过 pipeline 瓶颈）

    Args:
        model: 加载的模型
        tokenizer: 分词器
        messages: 对话消息列表
        max_new_tokens: 最大生成 token 数

    Returns:
        (response_text, generated_tokens, duration_seconds)
    """
    # 1. 构建 prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 2. Tokenize 并移至 GPU
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to("cuda")

    # 3. 生成文本（启用 Flash Attention）
    torch.cuda.synchronize()  # 确保 GPU 操作同步
    start_time = time.time()

    with torch.no_grad():  # 禁用梯度
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # 启用 KV 缓存
                return_dict_in_generate=True,
                output_scores=False
            )

    torch.cuda.synchronize()  # 确保生成完成
    duration = time.time() - start_time

    # 4. 解码响应（跳过输入部分）
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences.shape[1] - input_length
    response = tokenizer.decode(
        outputs.sequences[0, input_length:],
        skip_special_tokens=True
    )

    return response, generated_tokens, duration


# ===== 因果推断函数 =====
def extract_causal_relations(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        text: str,
        max_retries: int = 1
) -> List[Dict[str, str]]:
    """
    从文本中提取因果关系

    Args:
        model: 加载的模型
        tokenizer: 分词器
        text: 待分析文本
        max_retries: 最大重试次数

    Returns:
        因果关系列表
    """
    # 少样本示例
    FEW_SHOT_EXAMPLES = [
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
        {
            "input": "会议将于下周三举行，地点在总部大楼3楼会议室。",
            "output": []
        },
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

    # 构建系统提示
    system_prompt = (
        "你是一个因果推断专家系统。请从文本中精确识别因果关系，遵循严格规则：\n"
        "1. 仅当存在明确因果动词（导致、引发、降低、推动等）时才提取\n"
        "2. 每个因果关系包含三个字段：event（原因事件）、entity（受影响实体）、effect（影响描述）\n"
        "3. effect 必须包含极性（正面/负面）和具体变化\n"
        "4. 无明确因果关系时返回空列表 []\n"
        "5. 严格输出 JSON 格式，无任何额外文本\n\n"
        "### 参考示例:\n"
    )

    for ex in FEW_SHOT_EXAMPLES:
        system_prompt += f"输入: \"{ex['input']}\"\n输出: {json.dumps(ex['output'], ensure_ascii=False)}\n\n"

    # 构建用户提示
    user_prompt = f"### 待分析文本\n\"\"\"{text}\"\"\"\n### 严格JSON输出:"

    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 生成响应
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"  ♻️  重试 #{attempt} (清理缓存后)")
                torch.cuda.empty_cache()

            response, generated_tokens, duration = generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=128
            )

            # 计算并显示速度
            tokens_per_sec = generated_tokens / duration if duration > 0 else 0
            print(f"🚀 速度: {tokens_per_sec:.1f} token/s ({generated_tokens} tokens in {duration:.2f}s)")
            print(f"📝 模型响应: {response[:150]}..." if len(response) > 150 else f"📝 模型响应: {response}")

            # 修复 JSON
            repaired = repair_json(response, return_objects=True)
            return _validate_results(repaired)

        except Exception as e:
            print(f"  ❌  处理失败 (尝试 {attempt + 1}): {str(e)}")
            torch.cuda.empty_cache()

    print("  🛑  所有重试失败，返回空结果")
    return []


def _validate_results(results: Any) -> List[Dict[str, str]]:
    """验证并清理结果"""
    if not isinstance(results, list):
        return []

    valid_items = []
    for item in results:
        if not (isinstance(item, dict) and
                all(k in item for k in ['event', 'entity', 'effect'])):
            continue

        # 确保 effect 包含极性
        effect = item['effect']
        if "正面" not in effect and "负面" not in effect:
            # 尝试推断极性
            negative_words = ["上升", "增加", "上涨", "恶化", "降低", "减少", "下降", "消失"]
            if any(word in effect for word in negative_words):
                effect = f"负面：{effect}"
            else:
                effect = f"正面：{effect}"

        valid_items.append({
            "event": item['event'],
            "entity": item['entity'],
            "effect": effect
        })

    return valid_items


# ===== 资源监控 =====
def monitor_resources() -> Dict[str, Any]:
    """监控系统资源使用情况"""
    report = {}

    # GPU 显存
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        report['gpu'] = {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            # 'utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        }
        print(f"📊 GPU 显存: 已分配 {allocated:.1f}GB / 保留 {reserved:.1f}GB |")

    # CPU 内存
    mem = psutil.virtual_memory()
    report['cpu'] = {
        'used_gb': mem.used / 1e9,
        'total_gb': mem.total / 1e9,
        'percent': mem.percent
    }
    print(f"🧠 CPU 内存: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB ({mem.percent}%)")

    # Offload 目录大小
    offload_size = sum(f.stat().st_size for f in OFFLOAD_DIR.glob('**/*') if f.is_file()) / 1e9
    report['offload'] = {
        'size_gb': offload_size,
        'path': str(OFFLOAD_DIR)
    }
    print(f"💾 Offload 空间: {offload_size:.1f}GB (目录: {OFFLOAD_DIR})")

    # 磁盘空间
    disk = psutil.disk_usage(str(OFFLOAD_DIR))
    report['disk'] = {
        'used_gb': disk.used / 1e9,
        'total_gb': disk.total / 1e9,
        'percent': disk.percent
    }
    print(f"💽 磁盘空间: {disk.used / 1e9:.1f}GB / {disk.total / 1e9:.1f}GB ({disk.percent}%)")

    return report


# ===== 主程序 =====
def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🚀 DeepSeek-R1-0528-Qwen3-8B 因果推断系统启动")
    print("=" * 70)

    # 1. 资源监控 (启动前)
    print("\n🔍 启动前资源状态:")
    monitor_resources()

    # 2. 加载模型
    print("\n🧠 加载模型...")
    model = load_model_safely(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()  # 启用推理模式

    # 3. 验证配置
    print("\n✅ 模型配置验证:")
    print(f"  • RoPE 类型: {getattr(model.config, 'rope_scaling', {}).get('rope_type', '未设置')}")
    print(f"  • Flash Attention: {getattr(model.config, '_attn_implementation', '未设置')}")
    print(f"  • 量化类型: {getattr(model, 'quantization_method', '未量化')}")

    # 4. 资源监控 (加载后)
    print("\n🔍 模型加载后资源状态:")
    monitor_resources()

    # 5. 测试分析
    sample_text = (
        "全球气候变暖导致北极冰盖融化加速，引发海平面上升威胁沿海城市。"
        "同时，可再生能源投资增长降低了太阳能板成本，推动了绿色技术普及。"
    )

    print(f"\n🔍 分析文本: {sample_text}")
    results = extract_causal_relations(model, tokenizer, sample_text)

    # 6. 显示结果
    print("\n✅ 提取结果:")
    if results:
        for i, rel in enumerate(results, 1):
            print(f"\n{i}. 事件: {rel['event']}")
            print(f"   实体: {rel['entity']}")
            print(f"   影响: {rel['effect']}")
    else:
        print("  ⚠️  未检测到因果关系")

    # 7. 保存结果
    output_file = "deepseek_causal_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果保存至: {output_file}")

    # 8. 最终资源监控
    print("\n🔍 最终资源状态:")
    monitor_resources()

    print("\n" + "=" * 70)
    print("✅ 因果推断分析完成")
    print("=" * 70)


# ===== 启动程序 =====
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 严重错误: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 资源清理完成")