from google import genai
from google.genai import types
from config import *

with open('/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/high_quality_chart.png', 'rb') as f:
    image_bytes = f.read()

client = genai.Client(api_key=API_KEY_TEST)
response = client.models.generate_content(
model='gemini-2.5-flash',
contents=[
    types.Part.from_bytes(
    data=image_bytes,
    mime_type='image/png',
    ),
    '说说这个k线图是什么形态，判断一下后续走势'
]
)

print(response.text)