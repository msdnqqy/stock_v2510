from openai import OpenAI
import base64
import io
import os
import time

from PIL import Image
from openai import OpenAI

from config import *
from openai import OpenAI
import os
import base64
import sys
import time  # <--- 新增 1: 导入 time 模块
import mimetypes  # 引入这个库来自动判断文件类型
import json
import re
from config import *
from cv_utils import encode_image_vl, get_type

client = OpenAI(
    api_key="sk-no-key-required",
    base_url="http://localhost:8080/v1"
)

token_count = 0
first_token_time = None
start_gen_time = None


result = ''
# 记录开始处理的时间
start_process_time = time.time()

image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_3/image1.png"

prompt = """
你是一台高精度金融图像扫描仪。
任务：基于这张K线图，输出 MA5 与 MA20 的交点对应的成交量。

要求：
1) 找出 MA5 与 MA20 的交点像素坐标 intersection_px（左上角为(0,0)）。
2) 将 intersection_px.x 映射到对应的当日蜡烛 candle_index（从左到右从1开始），并给出 candle_bbox=[ymin,xmin,ymax,xmax]（仅实体或包含影线均可，但必须一致）。
3) 找到 candle_index 对应的成交量柱 volume_bar_bbox=[ymin,xmin,ymax,xmax]。
4) 额外输出交点日前 5 根K线（candle_index-1 到 candle_index-5）的成交量柱 previous_volume_bars（每个同样给 bbox）。
5) 如果能读到量能坐标轴刻度，请输出 volume_axis_ticks=[{"y":int,"label":str},...]（至少2个刻度，包含0更好）。
6) 如果能直接读到交点当日成交量数值，输出 volume_value_text（原样字符串）。

严格只输出 JSON，不要输出任何解释性文字。
JSON schema：
{
  "intersection_px": {"x": 0, "y": 0},
  "candle": {"index": 0, "bbox": [0,0,0,0]},
  "volume": {
    "bar_bbox": [0,0,0,0],
    "previous_bars": [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    "axis_ticks": [{"y": 0, "label": "0"}],
    "value_text": ""
  }
}
""".strip()

response = client.chat.completions.create(
    model="qwen3-vl-32b-thinking",
    stop=[
            "<|im_end|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ],
    temperature=0.0,
    top_p=0.1,
    max_tokens=16384,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stream=True,
    extra_body={
        "repeat_penalty": 1.0,
        "min_p": 0.05
    },
    # response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_vl(image_path)}"
                    }
                },
            ],
        }
    ],
)

# print(response.choices[0].message.content)

print("回答：", end="", flush=True)
think_content = ""
for chunk in response:
    # print(chunk)
    # print('reasoning_content' , chunk.choices[0].delta)

    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
        result += chunk.choices[0].delta.content
        token_count += 1
        # 捕获第一个 token 的时间
        if first_token_time is None:
            first_token_time = time.time()
            start_gen_time = first_token_time  # 开始生成的计时起点
            # 计算首字延迟 (Time to First Token)
            ttft = first_token_time - start_process_time
    elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
        think_content += chunk.choices[0].delta.reasoning_content
        token_count += 1
        # 捕获第一个 token 的时间
        if first_token_time is None:
            first_token_time = time.time()
            start_gen_time = first_token_time  # 开始生成的计时起点
            # 计算首字延迟 (Time to First Token)
            ttft = first_token_time - start_process_time

print("\n")
# 结束计时
end_time = time.time()
print("\n\n" + "=" * 30)

# --- 统计计算 ---
if token_count > 0 and start_gen_time:
    # 纯生成耗时 (扣除首字等待时间)
    gen_duration = end_time - start_gen_time
    # 首字延迟
    ttft = first_token_time - start_process_time

    # 计算速度 (Tokens Per Second)
    # 防止除以0 (虽然不太可能)
    speed = token_count / gen_duration if gen_duration > 0 else 0

    print(f"📊 统计报告:")
    print(f"   - 生成长度: {token_count} tokens")
    print(f"   - 首字延迟 (TTFT): {ttft:.2f} s (预处理耗时)")
    print(f"   - 生成耗时: {gen_duration:.2f} s")
    print(f"   - 平均速度: \033[1;32m{speed:.2f} tokens/s\033[0m")  # 绿色高亮显示速度
else:
    print("未生成有效内容。")
print("=" * 30 + "\n")
result = result.strip()
print("最终回复：", result)
print("=" * 30 + "\n")

def _extract_json_object(s):
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

def _bbox_height(b):
    if not isinstance(b, list) or len(b) != 4:
        return None
    try:
        return max(0.0, float(b[2]) - float(b[0]))
    except Exception:
        return None

def _parse_volume_text(t):
    if t is None:
        return None
    if not isinstance(t, str):
        t = str(t)
    s = t.strip().replace(",", "")
    if not s:
        return None
    mult = 1.0
    if "亿" in s:
        mult = 1e8
        s = s.replace("亿", "")
    elif "万" in s:
        mult = 1e4
        s = s.replace("万", "")
    try:
        return float(s) * mult
    except Exception:
        return None

try:
    data = _extract_json_object(result)
    candle_idx = data.get("candle", {}).get("index")
    bar_bbox = data.get("volume", {}).get("bar_bbox")
    prev_bars = data.get("volume", {}).get("previous_bars") or []

    bar_h = _bbox_height(bar_bbox)
    prev_hs = [h for h in (_bbox_height(b) for b in prev_bars) if h is not None and h > 0]
    prev_avg = (sum(prev_hs) / len(prev_hs)) if prev_hs else None
    spike_50 = (bar_h is not None and prev_avg is not None and bar_h > prev_avg * 1.5)

    vol_text = data.get("volume", {}).get("value_text")
    vol_value = _parse_volume_text(vol_text)

    summary = {
        "intersection_px": data.get("intersection_px"),
        "candle_index": candle_idx,
        "volume_value_text": vol_text,
        "volume_value_parsed": vol_value,
        "volume_bar_height_px": bar_h,
        "previous_avg_height_px": prev_avg,
        "volume_spike_over_50pct": spike_50,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
except Exception as e:
    print("解析/校验失败：", str(e))
    pass
