# ===== 顶部优化配置 =====
import os
import json
import torch
from time import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from json_repair import repair_json
from transformers import BitsAndBytesConfig

# WSL2 专属优化
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_DISABLE_RICH"] = "true"

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ===== 模型加载 =====
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507-FP8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 修复配置
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if hasattr(config, 'rope_scaling') and config.rope_scaling:
    if config.rope_scaling.get('rope_type') == 'yarn':
        config.rope_scaling.pop('attn_factor', None)
        config.rope_scaling['factor'] = 4.0
        config.rope_scaling['original_max_position_embeddings'] = 32768

# 加载 8-bit 模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    # quantization_config=BitsAndBytesConfig(
    #     # load_in_8bit=True,
    #     bnb_8bit_compute_dtype=torch.bfloat16,
    #     bnb_8bit_use_double_quant=True,
    # ),
    device_map="cuda:0",
    trust_remote_code=True,
)

model.eval()  # 启用推理模式
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"✅ 模型加载成功 | Flash Attention: {getattr(model.config, '_attn_implementation', 'unknown')}")


# ===== 高性能生成函数 =====
def generate_response_direct(messages, max_new_tokens=128):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        with torch.no_grad():
            start_time = time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            duration = time() - start_time

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.shape[1] - input_length
    print(f"✅ 模型生成完成 | 耗时: {duration:.2f}s | 生成字数: {generated_tokens} | 速度: {generated_tokens / duration:.2f}wps")
    response = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)

    return response, generated_tokens, duration


def use_template_validation(text):
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
        #
        # # 示例2: 医疗干预效果 (隐性因果)
        # {
        #     "input": "临床试验显示，每日服用维生素D补充剂6个月后，参与者骨折风险降低了22%。",
        #     "output": [
        #         {
        #             "event": "每日服用维生素D补充剂6个月",
        #             "entity": "骨折风险",
        #             "effect": "正面：降低22%"
        #         }
        #     ]
        # },
        #
        # # 示例3: 无明确因果关系 (边界情况)
        # {
        #     "input": "会议将于下周三举行，地点在总部大楼3楼会议室。",
        #     "output": []
        # },
        #
        # # 示例4: 复杂链式因果 (高级模式)
        # {
        #     "input": "供应链中断引发芯片短缺，迫使汽车制造商减产，进而导致二手车价格上涨30%。",
        #     "output": [
        #         {
        #             "event": "供应链中断",
        #             "entity": "芯片供应",
        #             "effect": "负面：短缺"
        #         },
        #         {
        #             "event": "芯片短缺",
        #             "entity": "汽车产量",
        #             "effect": "负面：减产"
        #         },
        #         {
        #             "event": "汽车减产",
        #             "entity": "二手车价格",
        #             "effect": "负面：上涨30%"
        #         }
        #     ]
        # }
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

    return generate_response_direct(messages)

if __name__ == "__main__":
    t = """
    今年以来，“A+H”上市热潮持续升温。截至10月中旬，今年已有11家A股公司成功实现“A+H”两地上市，上市数量在历年同期中位居第三，仅次于2015年的15家和1997年的13家。值得关注的是，仅10月以来，已有包括三一重工在内的四家A股公司相继通过H股上市聆讯，即将成为“A+H”阵营的新成员。
近日，又有A股上市企业披露了其在“A+H”上市方面的相关进展。智通财经APP获悉，日前，大金重工股份有限公司(以下简称“大金重工”)(002487.SZ)向港交所主板提交上市申请书，华泰国际、招商证券国际为其联席保荐人。
作为A股首家风电塔筒上市公司，此次冲击港交所，大金重工也有望成为港股“风电塔筒第一股”。那么其投资价值究竟如何？我们不妨从公司基本面入手分析。
行业发展前景可期
大金重工成立于2003年，2010年在深交所主板上市，是一家全球领先的海上风电核心装备供应商，深耕新能源行业近二十年，为全球大型海上风电开发商提供风电基础装备“建造+运输+交付”一站式解决方案。
为满足客户一站式、多样化需求，大金重工的产品与服务已从海上风电基础装备研发与制造，逐步延伸至远洋特种运输、船舶设计与建造、风电母港运营等领域，并积极布局新能源开发与运营业务，持续推动从产品供应商向系统服务商的战略转型。
大金重工的客户主要包括全球领先海上风电开发商及风电整机制造商。2023年公司将“两海战略”升级为“新两海战略”。 根据弗若斯特沙利文资料，截至2025年6月30日，大金重工是亚太地区唯一实现向欧洲批量交付单桩的供应商。2022年至2025年上半年，公司的海外业务突飞猛进，海外收入占总收入的比例从16.4%显著提升至79.0%，代表了“新两海战略”的持续落地以及客户的高度认可。
其实，近年来，在全球能源转型与碳中和目标的驱动下，风电已成为可再生能源发展中最具战略意义的板块之一。随着能源政策持续倾斜、技术成本不断下降以及绿色投资规模的快速扩张，全球风电市场进入了加速发展的新阶段。
从新增装机量来看，全球风电在过去几年保持稳健增长。新增装机量从2020年的95.3 GW增长至2024年的117.0 GW，复合年增长率为5.3%。随着电力需求结构优化 与大规模项目集中投产，预计到2030年新增装机量将进一步增至196.7GW，2024年至 2030年的复合年增长率将提升至9.0%。
而·海上风电在技术突破与政策驱动下正迎来爆发式增长，成为拉动行业增长的核心引擎。尽管海上风电当前市场占比仍处于较低水平，但未来增 长潜力显著，预计到2030年在全球风电新增装机量的占比将跃升至18.6%，新增装机 量从2024年的8.0 GW爆发式增长至2030年的36.7 GW，复合年增长率高达28.9%。 
中国与欧洲已成为推动全球海上风电发展的核心力量。截至2024年底，全球海上风电累计装机量中，中国贡献约一半的装机规模，而欧洲则以英国、德国、荷兰和丹麦为代表，中欧合计占全球装机量约94.5%。
    """
    print(use_template_validation(t))