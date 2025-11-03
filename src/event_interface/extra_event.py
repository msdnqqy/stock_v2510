import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from json_repair import repair_json

# 配置（使用Qwen1.5-1.8B作为替代，DeepSeek版本发布后替换MODEL_NAME）
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # 替换为"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"当可用时
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:\t",DEVICE)

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    # device_map="auto",
    trust_remote_code=True,
    use_cache=True
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=384,  # 增加以适应 few-shot
    temperature=0.2,  # 降低温度提高确定性
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)


def extract_causal_relations(text: str, max_retries=2):
    """
    增强版因果关系提取 (带 few-shot 学习)

    :param text: 输入文本
    :param max_retries: JSON 解析失败重试次数
    :return: 结构化因果关系列表
    """

    # ===== 专业级 Few-Shot 示例库 =====
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

    # ===== 构建 Few-Shot Prompt =====
    system_prompt = """你是一个因果推断专家系统，必须严格遵守以下规则：
1. 仅提取文本中明确的因果关系（事件→实体→影响）
2. 影响描述必须包含：
   - 极性（正面/负面/中性）
   - 具体变化（如"增加15%"、"显著降低"）
3. 输出严格的JSON数组格式，无额外文本
4. 无因果关系时返回空数组 []

### 参考示例（学习格式和逻辑）:
"""

    # 添加 few-shot 示例
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        system_prompt += f"\n示例 #{i}:\n"
        system_prompt += f"输入: \"{ex['input']}\"\n"
        system_prompt += f"输出: {json.dumps(ex['output'], ensure_ascii=False)}\n"

    # 用户查询
    user_prompt = f"""### 待分析文本
\"\"\"{text}\"\"\"\n
### 严格JSON输出（仅JSON，无其他内容）:"""

    # 构建对话
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 重试机制
    for attempt in range(max_retries + 1):
        try:
            # 生成响应
            response = text_generator(
                prompt,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )[0]['generated_text']

            # 修复 JSON
            repaired = repair_json(response, return_objects=True)

            # 验证结构
            if isinstance(repaired, list):
                # 验证每个条目
                valid_items = []
                for item in repaired:
                    if isinstance(item, dict) and {'event', 'entity', 'effect'}.issubset(item.keys()):
                        valid_items.append(item)
                return valid_items

            if attempt < max_retries:
                print(f"  ⚠️ 格式无效 (尝试 {attempt + 1}/{max_retries + 1})，重试中...")
                continue
            return []

        except Exception as e:
            print(f"  ❌ 尝试 {attempt + 1} 失败: {str(e)}")
            if attempt == max_retries:
                torch.cuda.empty_cache()
                return []

    return []


# ===== 高级测试套件 =====
TEST_CASES = [
    {
        "text": "新环保法规实施后，工厂排放量减少了40%，但合规成本增加了中小企业负担。",
        "expected": 2  # 预期2个因果关系
    },
    {
        "text": "研究发现，每天阅读30分钟可使认知能力提升15%，而久坐不动则降低心血管健康。",
        "expected": 2
    },
    {
        "text": "会议记录存档于共享文件夹，项目截止日期是下周五。",
        "expected": 0  # 无因果关系
    },
    {
        "text": "5G网络部署加速了物联网设备普及，引发数据安全担忧，促使新加密标准出台。",
        "expected": 3  # 链式因果
    }
]

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 因果推断引擎 (Few-Shot 增强版)")
    print(f"🚀 设备: {DEVICE.upper()} | 模型: {MODEL_NAME}")
    print("=" * 60)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n🧪 测试 #{i}:")
        print(f"📝 文本: \"{case['text']}\"")

        results = extract_causal_relations(case['text'])

        print(f"\n✅ 发现 {len(results)} 个因果关系 (预期: {case['expected']}):")
        for j, rel in enumerate(results, 1):
            print(f"  {j}. 事件: {rel['event']}")
            print(f"     实体: {rel['entity']}")
            print(f"     影响: {rel['effect']}")

        # 评估准确性
        accuracy = min(len(results), case['expected']) / max(len(results), case['expected'], 1)
        status = "🟢 通过" if abs(len(results) - case['expected']) <= 1 else "🟡 警告" if accuracy > 0.7 else "🔴 失败"
        print(f"\n📊 评估: {status} (准确率: {accuracy:.0%})")

    # 保存完整结果
    full_results = []
    for case in TEST_CASES:
        full_results.append({
            "input": case['text'],
            "output": extract_causal_relations(case['text'])
        })

    with open("causal_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存至: causal_analysis_results.json")