"""
事件抽取-deep research 版
1、判断是什么类型的新闻
2、判断使用哪个模型进行事件抽取（动态生成抽取模板）
"""

from event_summary_config_v1 import *
from llm_utils import *
from config import *

if __name__ == "__main__":
    # 测试模型
    test_prompt = user_input_d
    extract_result = chat_stream(EVENT_SUMMARY_DYNAMIC_PROMPT.replace('[news]', test_prompt))
    critic_result = chat_stream(CRITIC_PROMPT.replace('[原始文本]', test_prompt).replace('[分析报告]',extract_result))

    final_result= chat_stream(MODIFY_PROMPT.replace('[评审员反馈]', critic_result).replace('[原始文本]', test_prompt).replace('[分析报告]',extract_result))
    print("finnal result:\t",final_result)

