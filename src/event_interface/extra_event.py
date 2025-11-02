import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from json_repair import repair_json  # 需安装：pip install json-repair

# 配置（使用Qwen1.5-1.8B作为替代，DeepSeek版本发布后替换MODEL_NAME）
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 替换为"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"当可用时
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:\t",DEVICE)

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    # device_map=DEVICE,
    trust_remote_code=True,
    use_cache=True
)

# 创建文本生成管道
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=512,
    temperature=0.3,  # 低温度确保确定性输出
    top_p=0.85,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)


def extract_causal_relations(text):
    """从文本中提取因果关系三元组"""
    # 专业提示模板（针对因果推断优化）
    system_prompt = """你是一个因果推断专家AI，严格按JSON格式输出。从文本中提取：
1. 事件(event): 明确的动作或状态变化
2. 实体(entity): 受影响的对象（人/组织/物体）
3. 影响(effect): 事件对实体的具体影响（正面/负面/中性+具体描述）
要求：
- 只输出JSON数组，包含字段：event, entity, effect
- 每个JSON对象代表一个独立因果关系
- 无额外文本，无注释，无markdown
- 若无因果关系则返回空数组[]"""

    user_prompt = f"""分析以下文本，提取所有因果关系：
"{text}"\n\n输出严格的JSON格式："""

    # 构建Qwen1.5聊天模板
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 应用聊天模板（Qwen1.5专用格式）
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 生成并解析结果
    try:
        response = generator(
            prompt,
            return_full_text=False,
            clean_up_tokenization_spaces=True
        )[0]['generated_text']

        # 修复不完整JSON（处理模型截断问题）
        repaired_json = repair_json(response, return_objects=True)
        if isinstance(repaired_json, list):
            return repaired_json
        return []

    except Exception as e:
        print(f"生成错误: {str(e)}")
        return []


# 示例使用
if __name__ == "__main__":
    sample_text = (
        "由于新政策实施，小型企业的运营成本增加了15%。"
        "同时，疫苗接种率的提升显著降低了住院率。"
        "但供应链中断导致电子产品价格上涨。"
    )

    print("原始文本:")
    print(sample_text)
    print("\n提取的因果关系:")

    results = extract_causal_relations(sample_text)

    # 美化输出
    for i, item in enumerate(results, 1):
        print(f"\n关系 #{i}")
        print(f"事件: {item.get('event', 'N/A')}")
        print(f"实体: {item.get('entity', 'N/A')}")
        print(f"影响: {item.get('effect', 'N/A')}")

    # 保存结果（实际应用中）
    with open("causal_relations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到 causal_relations.json")