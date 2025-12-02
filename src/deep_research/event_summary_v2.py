"""
事件抽取-deep research 版
1、判断是什么类型的新闻
2、判断使用哪个模型进行事件抽取（动态生成抽取模板）
"""

from event_summary_config_v2 import *
from llm_utils import *
from config import *
from test_data import *

if __name__ == "__main__":
    # 测试模型
    test_prompt = news_list[2]
    extract_result = chat_stream(EXTRA_PROMPT.replace('[输入文本]', test_prompt))
    print("finnal result:\t",extract_result)

