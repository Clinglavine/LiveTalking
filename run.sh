#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=3
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 运行 Python 脚本(in conda env digital_human)

# 支持llm对话
# python app.py --transport webrtc  --max_session 6

# xtts克隆声音
# python app.py --tts xtts --transport webrtc --max_session 6 --REF_FILE  --TTS_SERVER http://localhost:

# gpt-sovits克隆声音
# python app.py --tts gpt-sovits --TTS_SERVER http://127.0.0.1: --REF_FILE  --REF_TEXT 

# cosyvoice克隆声音
# python app.py --tts cosyvoice --TTS_SERVER http://127.0.0.1: --REF_FILE  --REF_TEXT 


# 换数字人声音
# python app.py --model musetalk --avatar_id avator_3 --transport webrtc --max_session 6 --tts gpt-sovits --TTS_SERVER http://127.0.0.1:  --REF_FILE v --REF_TEXT 
python app.py --model musetalk --avatar_id avator_9 --transport webrtc --max_session 3 --tts cosyvoice --TTS_SERVER http://127.0.0.1:  --REF_FILE  --REF_TEXT 

# python app.py --model musetalk --avatar_id avator_3 --transport webrtc --max_session 6 --tts cosyvoice --TTS_SERVER http://127.0.0.1:  --REF_FILE --REF_TEXT 
# python app.py --model musetalk --avatar_id avator_3 --transport webrtc --max_session 6 --tts xtts --REF_FILE  --TTS_SERVER http://localhost:
