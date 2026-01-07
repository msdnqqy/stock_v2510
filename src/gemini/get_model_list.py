from google import genai

from config import *
# from .config import API_KEY

client = genai.Client(api_key=API_KEY)

# 列出所有支持生成内容的模型
print("当前可用的模型列表：")
for m in client.models.list():
    if 'generateContent' in m.supported_actions:
        print(f"ID: {m.name}") # 例如：models/gemini-3-flash