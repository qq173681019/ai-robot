"""
AI Robot - 电脑端 WebSocket 服务器
====================================
功能: 接收ESP32音频 → STT → LLM → TTS → 发回ESP32播放
依赖: pip install websockets openai pyaudio numpy

启动: python server.py
"""

import asyncio
import json
import struct
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("robot-server")

# ==================== 配置 ====================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765

# LLM 配置 (用智谱API，便宜好用)
LLM_API_KEY = "你的API_KEY"  # TODO: 改成你的key
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_MODEL = "glm-5-turbo"

# TTS 配置 (用 Edge TTS，免费)
# 或者用智谱的 TTS API

# STT 配置 (用 OpenAI Whisper API 或本地 whisper.cpp)
STT_ENGINE = "local"  # "local" = whisper.cpp, "api" = OpenAI API

# 系统提示词
SYSTEM_PROMPT = """你是一个友好的桌面AI机器人助手。你的名字叫"小呆"。
回答要简短自然，像真人聊天一样，每次回复不超过3句话。
你可以看到用户（通过摄像头），听到用户说话（通过麦克风）。
用中文回复。"""

# ==================== STT 语音识别 ====================
class STTEngine:
    def __init__(self, engine="local"):
        self.engine = engine

    async def transcribe(self, audio_bytes, sample_rate=16000):
        """将音频转为文字"""
        if self.engine == "local":
            return await self._local_whisper(audio_bytes, sample_rate)
        else:
            return await self._api_whisper(audio_bytes, sample_rate)

    async def _local_whisper(self, audio_bytes, sample_rate):
        """使用本地 whisper.cpp (需要安装)"""
        import tempfile
        import subprocess

        # 保存为临时wav文件
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        self._bytes_to_wav(audio_bytes, tmp_wav, sample_rate)

        try:
            # 调用 whisper.cpp
            result = subprocess.run(
                ["whisper-cpp", "-f", str(tmp_wav), "--language", "zh", "--model", "base"],
                capture_output=True, text=True, timeout=30
            )
            text = result.stdout.strip()
            # whisper.cpp 输出格式: [00:00:00.000 --> 00:00:03.000] 文字内容
            if "]" in text:
                text = text.split("]", 1)[1].strip()
            log.info(f"[STT] {text}")
            return text
        except FileNotFoundError:
            log.warning("[STT] whisper-cpp not found, using dummy input")
            return "你好"
        finally:
            tmp_wav.unlink(missing_ok=True)

    async def _api_whisper(self, audio_bytes, sample_rate):
        """使用 OpenAI Whisper API"""
        from openai import AsyncOpenAI
        import tempfile

        client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        self._bytes_to_wav(audio_bytes, tmp_wav, sample_rate)

        try:
            with open(tmp_wav, "rb") as f:
                resp = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="zh",
                )
            return resp.text
        finally:
            tmp_wav.unlink(missing_ok=True)

    def _bytes_to_wav(self, raw_audio, output_path, sample_rate):
        """将原始PCM数据转为WAV文件"""
        import wave
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(raw_audio)

# ==================== LLM 大语言模型 ====================
class LLMEngine:
    def __init__(self):
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.max_history = 20  # 保留最近10轮对话

    async def chat(self, user_text, image_base64=None):
        """发送对话并获取回复"""
        # 构造用户消息
        user_msg = {"role": "user", "content": user_text}
        self.history.append(user_msg)

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

            resp = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=self.history,
                max_tokens=200,
                temperature=0.7,
            )
            reply = resp.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})

            # 裁剪历史
            if len(self.history) > self.max_history + 1:
                self.history = [self.history[0]] + self.history[-(self.max_history):]

            log.info(f"[LLM] {reply}")
            return reply

        except Exception as e:
            log.error(f"[LLM] Error: {e}")
            return "不好意思，我脑子短路了，再说一遍？"

# ==================== TTS 语音合成 ====================
class TTSEngine:
    def __init__(self):
        self.engine = "edge-tts"  # 免费，质量好

    async def synthesize(self, text):
        """将文字转为音频，返回PCM bytes (16kHz 16bit mono)"""
        if self.engine == "edge-tts":
            return await self._edge_tts(text)
        else:
            return await self._local_tts(text)

    async def _edge_tts(self, text):
        """使用 Edge TTS"""
        import edge_tts
        import tempfile
        import wave

        communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
        tmp_mp3 = Path(tempfile.mktemp(suffix=".mp3"))

        try:
            await communicate.save(str(tmp_mp3))

            # MP3 → PCM (使用 ffmpeg)
            import subprocess
            tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
            subprocess.run([
                "ffmpeg", "-y", "-i", str(tmp_mp3),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                str(tmp_wav)
            ], capture_output=True, timeout=10)

            # 读取PCM数据
            with wave.open(str(tmp_wav), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())

            tmp_wav.unlink(missing_ok=True)
            return pcm

        except ImportError:
            log.warning("[TTS] edge-tts not installed, using dummy audio")
            return b'\x00' * 16000  # 0.5秒静音
        except Exception as e:
            log.error(f"[TTS] Error: {e}")
            return b'\x00' * 16000
        finally:
            tmp_mp3.unlink(missing_ok=True)

    async def _local_tts(self, text):
        """备用: 本地 pyttsx3"""
        # 不推荐，质量差，仅作后备
        return b'\x00' * 16000

# ==================== WebSocket 服务器 ====================
class RobotServer:
    def __init__(self):
        self.stt = STTEngine(engine=STT_ENGINE)
        self.llm = LLMEngine()
        self.tts = TTSEngine()

    async def handle_client(self, websocket):
        """处理单个ESP32客户端连接"""
        remote = websocket.remote_address
        log.info(f"[WS] Client connected: {remote}")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_frame(websocket, message)
                else:
                    log.warning(f"[WS] Unexpected text message: {message[:100]}")

        except Exception as e:
            log.error(f"[WS] Client error: {e}")
        finally:
            log.info(f"[WS] Client disconnected: {remote}")

    async def _handle_frame(self, websocket, frame):
        """处理一帧数据: 解析header + 音频"""
        try:
            # 解析帧: [4字节header长度][JSON header][音频数据]
            header_len = struct.unpack(">I", frame[:4])[0]
            header = json.loads(frame[4:4+header_len].decode())
            audio_data = frame[4+header_len:]

            msg_type = header.get("type", "unknown")
            log.info(f"[Frame] type={msg_type} audio_size={len(audio_data)}")

            if msg_type == "audio":
                # STT: 语音转文字
                user_text = await self.stt.transcribe(
                    audio_data,
                    header.get("sample_rate", 16000)
                )
                log.info(f"[User] {user_text}")

                if not user_text or len(user_text.strip()) < 2:
                    # 识别失败或太短，回复提示
                    reply_text = "我没听清，再说一遍？"
                else:
                    # LLM: 生成回复
                    reply_text = await self.llm.chat(user_text)

                # TTS: 文字转语音
                reply_audio = await self.tts.synthesize(reply_text)

                # 发送响应: [4字节header长度][JSON header][音频]
                resp_header = json.dumps({
                    "type": "audio",
                    "sample_rate": 16000,
                    "bits": 16,
                    "channels": 1,
                    "size": len(reply_audio),
                    "text": reply_text,
                }).encode()

                resp_frame = struct.pack(">I", len(resp_header)) + resp_header + reply_audio
                await websocket.send(resp_frame)
                log.info(f"[Reply] {reply_text} ({len(reply_audio)} bytes audio)")

        except Exception as e:
            log.error(f"[Frame] Error processing: {e}")
            import traceback
            traceback.print_exc()

    async def start(self):
        """启动WebSocket服务器"""
        import websockets

        log.info(f"[Server] Starting on ws://{SERVER_HOST}:{SERVER_PORT}")
        log.info(f"[Config] STT={STT_ENGINE}, LLM={LLM_MODEL}, TTS=edge-tts")

        async with websockets.serve(self.handle_client, SERVER_HOST, SERVER_PORT):
            log.info("[Server] Ready! Waiting for ESP32 connection...")
            await asyncio.Future()  # 永久运行

# ==================== 启动 ====================
async def main():
    print("=" * 50)
    print("  🤖 AI Robot Server v1.0")
    print("  电脑端 WebSocket 服务")
    print("=" * 50)
    print()
    print("依赖安装:")
    print("  pip install websockets openai edge-tts numpy")
    print("  还需要: ffmpeg (用于音频转码)")
    print()
    print("请确保:")
    print("  1. 电脑和ESP32在同一WiFi网络")
    print("  2. 防火墙放行 8765 端口")
    print("  3. ESP32固件里的SERVER_URL改成你电脑IP")
    print()

    server = RobotServer()
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
