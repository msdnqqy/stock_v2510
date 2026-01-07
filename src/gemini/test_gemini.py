from google import genai
from config import *

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key = API_KEY_TEST)

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="什么是大模型的涌现能力,从原理说明，并给出例子"
)
print(response.text)