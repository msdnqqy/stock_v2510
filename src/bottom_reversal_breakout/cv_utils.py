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

def encode_image(p):
        with Image.open(p) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            img.thumbnail((768, 768))

            byte_arr = io.BytesIO()
            img.save(byte_arr, format='JPEG', quality=95)
            return base64.b64encode(byte_arr.getvalue()).decode('utf-8')


def encode_image_vl(p, max_side=1536):
    with Image.open(p) as img:
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size
        if max_side is not None and max_side > 0:
            scale = max_side / max(w, h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG', optimize=True)
        return base64.b64encode(byte_arr.getvalue()).decode('utf-8')


def get_type(path):
    mime_type_temp, _ = mimetypes.guess_type(path)
    if mime_type_temp is None:
        mime_type_temp = 'image/jpeg'  # 默认回退到 jpeg
    return mime_type_temp