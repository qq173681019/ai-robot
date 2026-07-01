"""
AI Robot - 电脑端 WebSocket 服务器 v2.0
=========================================
功能: 接收ESP32音频 → STT → 唤醒词检测 → LLM → TTS → 发回ESP32播放

v2.0 改动:
  - LLM 切换为智谱 GLM (OpenAI 兼容接口)
  - 唤醒词检测 ("小呆")
  - 连续对话上下文
  - whisper.cpp 持久路径 (~/.whisper.cpp)
  - 小模型 → small 模型 (中文识别率大幅提升)

启动: python server.py

依赖 (推荐用 venv):
  python3 -m venv venv
  source venv/bin/activate
  pip install openai websockets edge-tts numpy

另外需要: ffmpeg (Edge TTS 音频转码)
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
SERVER_PORT = 8770

# LLM 配置 — 智谱 GLM (OpenAI 兼容接口)
# ⚠️ Key 从环境变量读取，不要硬编码，防止上传 GitHub 泄露
LLM_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_MODEL = os.environ.get("ROBOT_LLM_MODEL", "glm-5-turbo")

# TTS 配置 (Edge TTS 免费)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # 中文女声

# STT 配置 — 本地 whisper.cpp (持久路径)
STT_ENGINE = "local"  # "local" = whisper.cpp, "api" = OpenAI Whisper API
# Windows 端 server 通过 wsl.exe 调用 WSL Ubuntu 内部的 whisper-cli
WHISPER_CLI = ["wsl.exe", "-d", "Ubuntu"]
WHISPER_CLI_WSL = os.environ.get("WHISPER_CLI_PATH", "/home/jerico/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL_WSL = os.environ.get("WHISPER_MODEL_PATH", "/home/jerico/whisper.cpp/models/ggml-small.bin")

# 唤醒词
WAKE_WORD = "小呆"
WAKE_WORD_ENABLED = False  # 设为 True 开启唤醒词（先测试基本功能）

# 系统提示词
SYSTEM_PROMPT = """你是一个友好的桌面AI机器人助手。你的名字叫"小呆"。
回答要简短自然，像真人聊天一样，每次回复不超过3句话。
你可以看到用户（通过摄像头），听到用户说话（通过麦克风）。
用中文回复。语气活泼可爱，偶尔用emoji。"""


# ==================== 工具函数 ====================
def parse_frame(frame: bytes) -> tuple[dict, bytes]:
    """解析一帧数据: [4字节header长度][JSON header][音频数据]"""
    if len(frame) < 4:
        raise ValueError("frame too short")
    header_len = struct.unpack(">I", frame[:4])[0]
    if len(frame) < 4 + header_len:
        raise ValueError("frame truncated at header")
    header = json.loads(frame[4:4 + header_len].decode("utf-8"))
    audio = frame[4 + header_len:]
    return header, audio


def build_frame(header: dict, audio: bytes = b"") -> bytes:
    """构造一帧响应"""
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
        """使用本地 whisper.cpp (whisper-cli)"""
        import tempfile
        import subprocess

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        self._bytes_to_wav(audio_bytes, tmp_wav, sample_rate)

        try:
            # Windows 临时文件 → WSL 路径
            wav_path_str = str(tmp_wav).replace('\\', '/')
            if wav_path_str[1] == ':':
                drive = wav_path_str[0].lower()
                wsl_wav_path = f"/mnt/{drive}{wav_path_str[2:]}"
            else:
                wsl_wav_path = wav_path_str
            log.info(f"[STT] wav: {wsl_wav_path}")

            # 检查 whisper-cli 是否存在
            check = subprocess.run(
                WHISPER_CLI + ["test", "-f", WHISPER_CLI_WSL],
                capture_output=True, timeout=5,
            )
            if check.returncode != 0:
                log.error(f"[STT] whisper-cli not found at {WHISPER_CLI_WSL}")
                return ""

            result = subprocess.run(
                WHISPER_CLI + [
                    WHISPER_CLI_WSL,
                    "-m", WHISPER_MODEL_WSL,
                    "-f", wsl_wav_path,
                    "-l", "zh",
                    "--no-timestamps",
                    "-t", "4",
                    "-pp",  # print progress
                ],
                capture_output=True, text=True, timeout=60,
            )

            # 提取识别文字
            text = ""
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # 跳过 whisper 的元信息行
                if (line.startswith("whisper_") or line.startswith("system_info")
                    or line.startswith("main:") or line.startswith("read_audio")
                    or line.startswith("[") or line.startswith("mel")
                    or ("=" in line and "time" in line)
                    or line.startswith("loading")
                    or line.startswith("print_progress")):
                    continue
                text = line

            log.info(f"[STT] {text!r} (rc={result.returncode})")
            if not text and result.stderr:
                log.warning(f"[STT] stderr: {result.stderr[:200]}")
            return text

        except subprocess.TimeoutExpired:
            log.error("[STT] whisper-cli timeout (>60s)")
            return ""
        except Exception as e:
            log.error(f"[STT] whisper-cli error: {e}", exc_info=True)
            return ""
        finally:
            tmp_wav.unlink(missing_ok=True)

    async def _api_whisper(self, audio_bytes, sample_rate):
        """使用 OpenAI Whisper API"""
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
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_audio)


# ==================== LLM 大语言模型 (智谱 GLM) ====================
class LLMEngine:
    def __init__(self):
        self.system = SYSTEM_PROMPT
        self.history: list[dict] = []
        self.max_history = 20  # 保留最近 10 轮对话

    def _build_messages(self) -> list[dict]:
        msgs = list(self.history)
        if len(msgs) > self.max_history:
            msgs = msgs[-self.max_history:]
        return msgs

    async def chat(self, user_text, image_base64=None) -> str:
        """发送对话并获取回复 (智谱 GLM, OpenAI 兼容接口)"""
        from openai import AsyncOpenAI

        self.history.append({"role": "user", "content": user_text})

        try:
            client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, http_client=__import__('httpx').AsyncClient(trust_env=False))

            kwargs = dict(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.system},
                ] + self._build_messages(),
                max_tokens=200,
                temperature=0.7,
            )

            # 如果有图片，添加视觉支持
            if image_base64:
                kwargs["messages"][-1]["content"] = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]

            resp = await client.chat.completions.create(**kwargs)
            reply = resp.choices[0].message.content.strip()

            if not reply:
                reply = "……"
                log.warning("[LLM] Empty response")

            self.history.append({"role": "assistant", "content": reply})
            log.info(f"[LLM] {reply[:100]}{'...' if len(reply) > 100 else ''}")
            return reply

        except Exception as e:
            log.error(f"[LLM] Error: {e}", exc_info=True)
            if self.history and self.history[-1].get("content") == user_text:
                self.history.pop()
            return "不好意思，我脑子短路了，再说一遍？"

    def reset(self):
        """清空对话历史"""
        self.history = []


# ==================== 唤醒词检测 ====================
class WakeWordDetector:
    def __init__(self, word=WAKE_WORD, enabled=WAKE_WORD_ENABLED):
        self.word = word
        self.enabled = enabled
        self.activated = False  # 唤醒后保持对话状态
        self.last_active_time = 0
        self.timeout_seconds = 30  # 30秒无交互后重新需要唤醒词

    def check(self, text: str) -> bool:
        """检查是否包含唤醒词或处于激活状态"""
        import time
        now = time.time()

        if not self.enabled:
            return True  # 关闭唤醒词，始终响应

        # 检查唤醒词
        if self.word in text:
            self.activated = True
            self.last_active_time = now
            log.info(f"[WakeWord] '{self.word}' detected, activated")
            # 移除唤醒词本身
            text_clean = text.replace(self.word, "").strip()
            return True

        # 检查是否在激活超时内
        if self.activated and (now - self.last_active_time) < self.timeout_seconds:
            self.last_active_time = now
            return True

        # 超时，需要重新唤醒
        if self.activated:
            log.info(f"[WakeWord] Timeout, need wake word again")
            self.activated = False

        return False

    def clean_text(self, text: str) -> str:
        """移除唤醒词"""
        return text.replace(self.word, "").strip()


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
            return b'\x00' * 16000  # 0.5s 静音兜底

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
        self.wake_word = WakeWordDetector()

    async def handle_client(self, websocket):
        """处理单个ESP32客户端连接"""
        remote = websocket.remote_address
        log.info(f"[WS] Client connected: {remote}")

        # 每个连接独立的唤醒词状态
        connection_wake = WakeWordDetector(
            word=WAKE_WORD, enabled=WAKE_WORD_ENABLED
        )

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_frame(websocket, message, connection_wake)
                else:
                    log.warning(f"[WS] Unexpected text: {message[:100]}")

        except Exception as e:
            log.error(f"[WS] Client error: {e}", exc_info=True)
        finally:
            log.info(f"[WS] Client disconnected: {remote}")

    async def _handle_frame(self, websocket, frame: bytes, wake: WakeWordDetector):
        """处理一帧数据"""
        try:
            header, audio_data = parse_frame(frame)
        except (ValueError, json.JSONDecodeError) as e:
            log.error(f"[Frame] parse error: {e}")
            return

        msg_type = header.get("type", "unknown")
        log.info(f"[Frame] type={msg_type} audio_size={len(audio_data)}")

        if msg_type == "image":
            # 摄像头画面 → 暂存，等下一次语音时附带
            image_b64 = audio_data.decode("utf-8") if audio_data else None
            log.info(f"[Image] Received camera frame ({len(audio_data)} bytes)")
            # 回个 ACK
            ack = build_frame({"type": "image_ack"})
            await websocket.send(ack)
            return

        if msg_type != "audio":
            log.warning(f"[Frame] unsupported type: {msg_type}")
            return

        # 1. STT: 语音转文字
        sample_rate = header.get("sample_rate", 16000)
        user_text = await self.stt.transcribe(audio_data, sample_rate)
        log.info(f"[User] {user_text}")

        if not user_text or len(user_text.strip()) < 1:
            # 没听清，不发回（让ESP32继续录）
            return

        # 2. 唤醒词检测
        if not wake.check(user_text):
            log.info(f"[WakeWord] Not activated, ignoring: '{user_text[:30]}'")
            # 发一个静默响应，让ESP32知道但不播放
            resp_header = {"type": "audio", "size": 0, "text": "", "silent": True}
            await websocket.send(build_frame(resp_header))
            return

        # 清理唤醒词
        clean_text = wake.clean_text(user_text)

        # 如果只有唤醒词没有实际内容，回个应答
        if not clean_text:
            reply_text = "我在呢，说吧~"
        else:
            # 3. LLM: 生成回复
            reply_text = await self.llm.chat(clean_text)

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
        await websocket.send(build_frame(resp_header, reply_audio))
        log.info(f"[Reply] {reply_text[:50]}{'...' if len(reply_text) > 50 else ''} ({len(reply_audio)} bytes)")

    async def start(self):
        """启动WebSocket服务器"""
        import websockets

        log.info(f"[Server] Starting on ws://{SERVER_HOST}:{SERVER_PORT}")
        log.info(f"[Config] STT={STT_ENGINE}, LLM={LLM_MODEL}, TTS=edge-tts/{TTS_VOICE}")
        log.info(f"[Config] WakeWord='{WAKE_WORD}' enabled={WAKE_WORD_ENABLED}")
        log.info(f"[Config] Whisper: {WHISPER_CLI_WSL}")
        log.info(f"[Config] Model: {WHISPER_MODEL_WSL}")

        async with websockets.serve(
            self.handle_client, SERVER_HOST, SERVER_PORT,
            ping_interval=30, ping_timeout=120,
            max_size=2**20,  # 1MB max frame
        ):
            log.info(f"[Server] Ready on port {SERVER_PORT}! Waiting for ESP32...")
            await asyncio.Future()


# ==================== 启动 ====================
async def main():
    print("=" * 50)
    print("  🤖 AI Robot Server v2.0 (GLM)")
    print("  电脑端 WebSocket 服务")
    print("=" * 50)
    print()
    if not LLM_API_KEY:
        print("  ⚠️  ZHIPU_API_KEY 未设置！请先运行:")
        print("     set ZHIPU_API_KEY=你的智谱key")
        print()
    print(f"  LLM:     {LLM_MODEL} (智谱GLM)")
    print(f"  STT:     whisper.cpp (local)")
    print(f"  TTS:     Edge TTS / {TTS_VOICE}")
    print(f"  WakeWord: '{WAKE_WORD}' ({'ON' if WAKE_WORD_ENABLED else 'OFF'})")
    print()
    print("请确保:")
    print("  1. 设置环境变量: set ZHIPU_API_KEY=你的key")
    print("  2. 电脑和ESP32在同一WiFi网络")
    print("  3. 防火墙放行 8770 端口")
    print("  4. ESP32固件里的 SERVER_URL 改成你电脑IP")
    print("  5. WSL Ubuntu 中 whisper.cpp 已编译")
    print()

    server = RobotServer()
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
