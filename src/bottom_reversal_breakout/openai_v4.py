import argparse
import json
import re
import time

from openai import OpenAI

from config import *
from cv_utils import encode_image_vl, get_type

image_path_sample_1="/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/high_quality_chart_sample1.png"


"""
底部反转形态识别
"""

def _extract_json_object(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}\s*$", s)
    if not m:
        m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


def analyze_bottom_reversal(
    image_path: str,
    model: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repeat_penalty: float,
    min_p: float,
    timeout: int,
):
    client = OpenAI(
        api_key="sk-no-key-required",
        base_url=base_url,
        timeout=timeout,
    )

    prompt = (
        "看一下这个k线是什么形态，分析后续是什么走势"
    )

    start_process_time = time.time()
    token_count = 0
    first_token_time = None
    start_gen_time = None

    response = client.chat.completions.create(
        model=model,
        stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=True,
        extra_body={
            "repeat_penalty": repeat_penalty,
            "min_p": min_p,
        },
        # response_format={"type": "json_object"},
        messages=[
            # {"role": "system", "content":  """
            # 你是一个专业的金融分析师，你需要根据k线图以及VOL表现的形态，以及后续的走势。
            # 1、整体趋势 (Trend)
            # 2、k线形态
            # 3、后续走势
            # """},
            {
                "role": "user",
                "content": [
                   
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{get_type(image_path_sample_1)};base64,{encode_image_vl(image_path_sample_1)}"},
                    },
                     {"type": "text", "text":  "看一下这个k线是什么形态，分析后续是什么走势"}
                    #  {"type": "text", "text":  SYSTEM_PROMPT}
                ],
            },

            {
                "role": "assistant",
                "content": """好的，我们来详细分析一下这张K线图。

图表形态分析
这张图表展示了一段清晰的下降趋势，并在近期出现了明显的反转迹象。我们可以从以下几个方面进行分析：

整体趋势 (Trend)
图表上方的两条曲线（蓝色和橙色）是移动平均线（MA），它们都呈明显的向下倾斜状态，表明市场处于一个长期的下跌趋势中。
在0603左右出现了一个放量的大阴线（红色实体），这是趋势加速的信号，之后价格持续下探。
底部形态与反转信号 (Reversal Pattern & Signal)
在0618到0701期间，价格在低位形成了一个相对稳定的区域，并且出现了多根绿色（上涨）的K线，这通常预示着下跌动能的减弱。
最关键的信号出现在0701这一天：
大阳线（绿色）: 0701当天收出一根非常强劲的大阳线，其收盘价远高于开盘价，几乎覆盖了前一天的全部实体，这是一个强烈的看涨反转信号。
成交量放大: 下方的成交量柱状图显示，0701这一天的成交量（绿色柱子）是近期最大的一次，这被称为“放量突破”或“放量上涨”。巨大的成交量配合大阳线，意味着有大量买盘进场，确认了上涨的意愿，使反转信号更加可靠。
技术指标解读
移动平均线: 蓝色和橙色均线已经走平，并且开始有向上拐头的趋势，特别是0701之后，价格开始站上这两条均线，表明短期趋势可能正在由跌转涨。
成交量: 如前所述，0701的放量是整个图表中最显著的特征，它为这次反弹提供了强有力的支撑。
后续走势预测
综合以上分析，当前的形态是一个**典型的“底部反转”**形态。

短期走势: 鉴于0701的放量大阳线和价格站上均线，短期内价格有望继续上涨。下一个目标位可能是之前下跌趋势中的一个重要阻力位，例如0704附近的高点，或者更远一些的前期高点（如0512附近）。如果能有效突破并站稳，将确认上涨趋势成立。
中期走势: 这次反转的成功与否，很大程度上取决于后续的量能是否持续。如果接下来的几个交易日能够保持温和放量的上涨，那么上涨趋势会得到巩固。反之，如果价格上涨但成交量萎缩，可能会形成“假突破”，价格可能再次回落。
风险提示:
回踩确认: 在大幅上涨后，价格可能会出现一次回踩（Retest），即回调至之前的支撑位（如0701的低点或两条均线）附近，以确认该位置的有效性。这种回踩往往是健康的，可以消化获利盘，为后续上涨积蓄力量。
趋势未完全扭转: 尽管信号强烈，但整个大趋势的转变需要时间。目前仍需观察市场的持续性。如果后续几根K线是小阳线或十字星，说明多空双方仍在博弈，上涨动能可能不足。
总结
这个K线形态是一个成功的底部反转信号，核心是0701的放量大阳线。这表明市场情绪发生了逆转，卖压减弱，买盘强势介入。

后续走势的预期是：价格大概率会延续上涨趋势，挑战更高的阻力位。

操作建议:

短线投资者: 可以在0701大阳线确认后，考虑逢低买入或持有，目标看向前期高点。
中长线投资者: 应密切关注后续的成交量和价格能否持续站稳在均线上方。如果能持续放量上涨，则可以视为趋势反转的确认，可积极布局。同时，要警惕可能出现的回踩，可在回踩时寻找更好的入场点。"""
            },

            {
                "role": "user",
                "content": [
                   
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{get_type(image_path)};base64,{encode_image_vl(image_path)}"},
                    },
                     {"type": "text", "text":  "参考上下文给的例子，看一下这个现在给的k线是什么形态，分析后续是什么走势"}
                    #  {"type": "text", "text":  SYSTEM_PROMPT}
                ],
            },

            
        ],
    )

    result = ""
    parsed = None

    for chunk in response:
        delta = chunk.choices[0].delta

        if getattr(delta, "content", None) is not None:
            piece = delta.content
            print(piece, end="", flush=True)
            result += piece
            token_count += 1
            if first_token_time is None:
                first_token_time = time.time()
                start_gen_time = first_token_time

            try:
                parsed = _extract_json_object(result)
                break
            except Exception:
                pass

    end_time = time.time()
    print("\n")

    if token_count > 0 and start_gen_time:
        gen_duration = max(1e-6, end_time - start_gen_time)
        ttft = (first_token_time - start_process_time) if first_token_time else None
        speed = token_count / gen_duration
        print("=" * 30)
        print("统计：")
        if ttft is not None:
            print(f"- TTFT: {ttft:.2f}s")
        print(f"- tokens: {token_count}")
        print(f"- gen_s: {gen_duration:.2f}s")
        print(f"- tok/s: {speed:.2f}")
        print("=" * 30)

    # if parsed is None:
    #     parsed = _extract_json_object(result)

    # print(json.dumps(parsed, ensure_ascii=False, indent=2))
    # return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/high_quality_chart.png",
    )
    parser.add_argument("--model", default="qwen3-vl-32b-thinking")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=6000)
    args = parser.parse_args()

    analyze_bottom_reversal(
        image_path=args.image,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repeat_penalty=args.repeat_penalty,
        min_p=args.min_p,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()