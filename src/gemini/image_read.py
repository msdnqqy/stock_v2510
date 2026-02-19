from google import genai
from google.genai import types
from config import *

with open('/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/high_quality_chart.png', 'rb') as f:
    image_bytes = f.read()

client = genai.Client(api_key=API_KEY)
response = client.models.generate_content(
model='gemini-3-flash-preview',
# model = 'gemini-3-pro-image-preview',
# model = 'gemma-3-27b-it',
# model = 'gemma-3n-e4b-it',
contents=[
    types.Part.from_bytes(
    data=image_bytes,
    mime_type='image/png',
    ),
    '1、数一下k线图中有多少跟k线；2、说说这个k线图是什么形态，判断一下后续走势'
]
)

print(response.text)