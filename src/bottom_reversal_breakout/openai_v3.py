import argparse
import json
import re
import time

from openai import OpenAI

from config import SYSTEM_PROMPT_1
from cv_utils import encode_image_vl


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
        "请严格按 SYSTEM_PROMPT 要求，分析该K线图是否满足“底部反转突破”形态。"
        "只输出 JSON，不要输出任何额外文字。"
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
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_1},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image_vl(image_path)}"},
                    },
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

    if parsed is None:
        parsed = _extract_json_object(result)

    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_3/image1.png",
    )
    parser.add_argument("--model", default="qwen3-vl-32b-thinking")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
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