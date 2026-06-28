"""
AI Robot - 电脑端 WebSocket 服务器
====================================
功能: 接收ESP32音频 → STT → LLM → TTS → 发回ESP32播放

启动: python server.py

依赖 (推荐用 venv):
  python3 -m venv venv
  source venv/bin/activate
  pip install anthropic websockets edge-tts numpy

另外需要: ffmpeg (Edge TTS 音频转码)
  macOS:  brew install ffmpeg
  Windows: winget install ffmpeg
"""

import asyncio
import json
import struct
import logging
import os
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("robot-server")

# ==================== 配置 ====================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8770  # 8765/8766 都被占/被绑 (WSL NAT 表残留), 8770 干净

# LLM 配置 — MiniMax (Anthropic 兼容接口)
# 推荐用环境变量管理，避免 key 写进 git
#   export ROBOT_LLM_API_KEY=sk-cp-...
DEFAULT_LLM_API_KEY = "sk-cp-...Sd7g"
LLM_API_KEY = os.environ.get("ROBOT_LLM_API_KEY", DEFAULT_LLM_API_KEY)
LLM_BASE_URL = "https://api.minimaxi.com/anthropic"
LLM_MODEL = "mm2.7"

# TTS 配置 (Edge TTS 免费)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # 中文女声

# STT 配置 — 本地 whisper.cpp（2026-06-21 装在 /tmp/whisper.cpp）
STT_ENGINE = "local"  # "local" = whisper.cpp, "api" = OpenAI Whisper API
# Windows 端 server 通过 wsl.exe 调用 WSL Ubuntu 内部的 whisper-cli (避免在 Windows 重编译)
# 注意: wsl.exe 默认进 docker-desktop (有 * 标记的), 必须 -d Ubuntu 显式指定
WHISPER_CLI = ["wsl.exe", "-d", "Ubuntu"]
WHISPER_CLI_WSL = "/tmp/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL_WSL = "/tmp/whisper.cpp/models/ggml-tiny.bin"

# 系统提示词
SYSTEM_PROMPT = """你是一个友好的桌面AI机器人助手。你的名字叫"小呆"。
回答要简短自然，像真人聊天一样，每次回复不超过3句话。
你可以看到用户（通过摄像头），听到用户说话（通过麦克风）。
用中文回复。"""


# ==================== 工具函数 ====================
def parse_frame(frame: bytes) -> tuple[dict, bytes]:
    """解析一帧数据: [4字节header长度][JSON header][音频数据]
    返回: (header_dict, audio_bytes)
    """
    if len(frame) < 4:
        raise ValueError("frame too short")
    header_len = struct.unpack(">I", frame[:4])[0]
    if len(frame) < 4 + header_len:
        raise ValueError("frame truncated at header")
    header = json.loads(frame[4:4 + header_len].decode("utf-8"))
    audio = frame[4 + header_len:]
    return header, audio


def build_frame(header: dict, audio: bytes = b"") -> bytes:
    """构造一帧响应: [4字节header长度][JSON header][音频数据]"""
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + audio


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
        """使用本地 whisper.cpp (whisper-cli) — Windows 端通过 wsl.exe 调用 WSL 内部的 whisper-cli"""
        import tempfile
        import subprocess

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        self._bytes_to_wav(audio_bytes, tmp_wav, sample_rate)

        try:
            # Windows 临时文件 WSL 通过 /mnt/c/... 可直接访问
            wav_path_str = str(tmp_wav).replace('\\', '/')
            # 如果是 C:\Users\...\Temp\foo.wav → /mnt/c/Users/.../Temp/foo.wav
            if wav_path_str[1] == ':':
                drive = wav_path_str[0].lower()
                wsl_wav_path = f"/mnt/{drive}{wav_path_str[2:]}"
            else:
                wsl_wav_path = wav_path_str
            log.info(f"[STT] wav file: {wsl_wav_path}")

            # 调用 whisper-cli (在 WSL Ubuntu 内部)
            result = subprocess.run(
                WHISPER_CLI + [
                    WHISPER_CLI_WSL,
                    "-m", WHISPER_MODEL_WSL,
                    "-f", wsl_wav_path,
                    "-l", "zh",
                    "--no-timestamps",
                    "-t", "4",
                ],
                capture_output=True, text=True, timeout=30,
            )

            # 提取识别文字
            text = ""
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                if (line.startswith("whisper_") or line.startswith("system_info")
                    or line.startswith("main:") or line.startswith("read_audio")
                    or line.startswith("[")
                    or ("=" in line and "time" in line)):
                    continue
                text = line

            log.info(f"[STT] {text!r} (rc={result.returncode})")
            if not text and result.stderr:
                log.warning(f"[STT] stderr: {result.stderr[:200]}")
            return text

        except subprocess.TimeoutExpired:
            log.error("[STT] whisper-cli timeout (>30s)")
            return ""
        except Exception as e:
            log.error(f"[STT] whisper-cli error: {e}", exc_info=True)
            return ""
        finally:
            tmp_wav.unlink(missing_ok=True)

    async def _api_whisper(self, audio_bytes, sample_rate):
        """使用 OpenAI Whisper API
        注意: STT 用的是 OpenAI 服务，跟 LLM 不是同一家，key 需单独配
        """
        from openai import AsyncOpenAI
        import tempfile

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            log.error("[STT] OPENAI_API_KEY not set, falling back to local")
            return await self._local_whisper(audio_bytes, sample_rate)

        client = AsyncOpenAI(api_key=openai_key)
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


# ==================== LLM 大语言模型 (MiniMax / Anthropic 兼容) ====================
class LLMEngine:
    def __init__(self):
        self.system = SYSTEM_PROMPT
        self.history: list[dict] = []  # 不含 system
        self.max_history = 20  # 保留最近 10 轮 (user + assistant)

    def _build_messages(self) -> list[dict]:
        """构造发送给 API 的 messages 列表"""
        msgs = list(self.history)
        if len(msgs) > self.max_history:
            msgs = msgs[-self.max_history:]
        return msgs

    async def chat(self, user_text, image_base64=None) -> str:
        """发送对话并获取回复"""
        from anthropic import AsyncAnthropic

        self.history.append({"role": "user", "content": user_text})

        try:
            client = AsyncAnthropic(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

            resp = await client.messages.create(
                model=LLM_MODEL,
                system=self.system,
                messages=self._build_messages(),
                max_tokens=200,
                temperature=0.7,
            )

            # 提取文本: content 可能是 list[TextBlock/ThinkingBlock/...]
            reply_parts = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    reply_parts.append(block.text)
            reply = "".join(reply_parts).strip()

            if not reply:
                reply = "……"
                log.warning("[LLM] No text content in response, raw: %r", resp.content)

            self.history.append({"role": "assistant", "content": reply})
            log.info(f"[LLM] {reply[:100]}{'...' if len(reply) > 100 else ''}")
            return reply

        except Exception as e:
            log.error(f"[LLM] Error: {e}", exc_info=True)
            # 出错时把刚才的 user 消息回滚，避免污染历史
            if self.history and self.history[-1].get("content") == user_text:
                self.history.pop()
            return "不好意思，我脑子短路了，再说一遍？"

    def reset(self):
        """清空对话历史"""
        self.history = []


# ==================== TTS 语音合成 (Edge TTS) ====================
class TTSEngine:
    def __init__(self, voice=TTS_VOICE):
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        """将文字转为音频，返回 PCM bytes (16kHz 16bit mono)"""
        if not text or not text.strip():
            return b""

        try:
            return await self._edge_tts(text)
        except Exception as e:
            log.error(f"[TTS] Error: {e}", exc_info=True)
            # 兜底: 返回 0.5s 静音
            return b'\x00' * 16000

    async def _edge_tts(self, text: str) -> bytes:
        """使用 Edge TTS (免费)"""
        import edge_tts
        import tempfile
        import wave
        import subprocess

        communicate = edge_tts.Communicate(text, self.voice)
        tmp_mp3 = Path(tempfile.mktemp(suffix=".mp3"))
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))

        try:
            await communicate.save(str(tmp_mp3))

            # MP3 → 16kHz 16bit mono PCM (使用 ffmpeg)
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(tmp_mp3),
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                    "-f", "wav", str(tmp_wav),
                ],
                capture_output=True, timeout=10,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'ignore')}")

            with wave.open(str(tmp_wav), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
            return pcm

        finally:
            tmp_mp3.unlink(missing_ok=True)
            tmp_wav.unlink(missing_ok=True)


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
            log.error(f"[WS] Client error: {e}", exc_info=True)
        finally:
            log.info(f"[WS] Client disconnected: {remote}")

    async def _handle_frame(self, websocket, frame: bytes):
        """处理一帧数据"""
        try:
            header, audio_data = parse_frame(frame)
        except (ValueError, json.JSONDecodeError) as e:
            log.error(f"[Frame] parse error: {e}")
            return

        msg_type = header.get("type", "unknown")
        log.info(f"[Frame] type={msg_type} audio_size={len(audio_data)}")

        if msg_type != "audio":
            log.warning(f"[Frame] unsupported type: {msg_type}")
            return

        # 1. STT: 语音转文字
        sample_rate = header.get("sample_rate", 16000)
        user_text = await self.stt.transcribe(audio_data, sample_rate)
        log.info(f"[User] {user_text}")

        # 2. 决定回复文本
        if not user_text or len(user_text.strip()) < 2:
            reply_text = "我没听清，再说一遍？"
        else:
            # 3. LLM: 生成回复
            reply_text = await self.llm.chat(user_text)

        # 4. TTS: 文字转语音
        reply_audio = await self.tts.synthesize(reply_text)

        # 5. 发送响应
        resp_header = {
            "type": "audio",
            "sample_rate": 16000,
            "bits": 16,
            "channels": 1,
            "size": len(reply_audio),
            "text": reply_text,
        }
        resp_frame = build_frame(resp_header, reply_audio)
        await websocket.send(resp_frame)
        log.info(f"[Reply] {reply_text[:50]}{'...' if len(reply_text) > 50 else ''} ({len(reply_audio)} bytes audio)")

    async def start(self):
        """启动WebSocket服务器"""
        import websockets

        log.info(f"[Server] Starting on ws://{SERVER_HOST}:{SERVER_PORT}")
        log.info(f"[Config] STT={STT_ENGINE}, LLM={LLM_MODEL}, TTS=edge-tts/{TTS_VOICE}")

        async with websockets.serve(self.handle_client, SERVER_HOST, SERVER_PORT, ping_interval=None, ping_timeout=None):
            log.info(f"[Server] Ready on port {SERVER_PORT}! Waiting for ESP32 connection...")
            await asyncio.Future()  # 永久运行


# ==================== 启动 ====================
async def main():
    print("=" * 50)
    print("  🤖 AI Robot Server v1.1 (minimax)")
    print("  电脑端 WebSocket 服务")
    print("=" * 50)
    print()
    print("依赖安装 (用 venv):")
    print("  python3 -m venv venv && source venv/bin/activate")
    print("  pip install anthropic websockets edge-tts numpy")
    print()
    print("还需要: ffmpeg (用于 Edge TTS 音频转码)")
    print()
    print("环境变量 (可选, 推荐):")
    print("  export ROBOT_LLM_API_KEY=sk-cp-...   # 覆盖 LLM API key")
    print("  export OPENAI_API_KEY=sk-...         # 仅 STT=api 模式需要")
    print()
    print("请确保:")
    print("  1. 电脑和ESP32在同一WiFi网络")
    print("  2. 防火墙放行 8765 端口")
    print("  3. ESP32固件里的 SERVER_URL 改成你电脑IP")
    print()

    server = RobotServer()
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
