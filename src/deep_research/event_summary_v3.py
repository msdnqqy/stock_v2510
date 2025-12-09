"""
事件抽取-deep research 版
1、判断是什么类型的新闻
2、判断使用哪个模型进行事件抽取（动态生成抽取模板）
"""

from event_summary_config_v3 import *
from llm_utils import *
from config import *
from test_data import *

if __name__ == "__main__":
    # 测试模型
    # test_prompt = """某半导体公司宣布研发出 2nm 制程芯片，良率达到 90%，预计明年量产。"""
    test_prompt = news_list[-1]
    extract_result = chat_stream(PROMPT.replace('[输入文本]', test_prompt))
    print("finnal result:\t",extract_result)

