# 🤖 AI Robot 项目 v2.0

> 桌面AI机器人：ESP32-S3 硬件 + 电脑端 LLM 大脑

## v2.0 新特性

- 🎤 **VAD 自动录音** — 不再固定时长，自动检测说话起止
- 🔗 **断线重连** — WebSocket 断了自动重连，永不掉线
- 💬 **连续对话** — 多轮上下文，像真人聊天
- 🗣️ **唤醒词** — 说"小呆"唤醒（可关闭）
- 🧠 **GLM 大脑** — 智谱 GLM-5-Turbo，中文原生
- 📝 **Whisper Small** — 中文识别率大幅提升（tiny→small）
- 📷 **摄像头预留** — 代码框架已就绪，接线后启用
- ✨ **开机和弦** — C-E-G 三音提示启动完成

## 项目结构

```
ai-robot/
├── esp32-firmware/
│   └── main.py               # ESP32 MicroPython 固件 v2.0
├── server/
│   └── server.py             # 电脑端 WebSocket 服务 v2.0
├── whisper.cpp/              # Windows 端 whisper 模型缓存
├── docs/
│   └── wiring-diagram.svg    # 接线图
├── shopping-list.md          # 硬件购物清单
├── test-mic.py               # 麦克风测试脚本
└── README.md
```

## 快速开始

### 1. WSL 端 (whisper.cpp)

```bash
# 已安装到 ~/whisper.cpp (持久路径)
# whisper-cli: ~/whisper.cpp/build/bin/whisper-cli
# 模型:       ~/whisper.cpp/models/ggml-small.bin
```

### 2. PC 端 (server.py)

```bash
# 安装依赖
pip install openai websockets edge-tts numpy

# 还需要 ffmpeg
winget install ffmpeg

# 启动服务器
cd D:\GitHub\ai-robot\server
python server.py
```

### 3. ESP32 端 (firmware)

```bash
# 刷 MicroPython 固件 (ESP32-S3 SPIRAM)
esptool.py --chip esp32s3 -p COM3 write_flash -z 0x0 firmware/ESP32_GENERIC_S3-SPIRAM_OCT-*.bin

# 上传代码
mpremote cp esp32-firmware/main.py :main.py

# 配置 main.py 里的:
#   WIFI_SSID / WIFI_PASS / SERVER_URL
```

### 4. 运行！

1. 先启动 PC 端 `python server.py`
2. 给 ESP32 上电
3. 听到 C-E-G 和弦 → 就绪
4. 说话 → 蓝灯（listening）
5. 停止说话 → 黄灯（thinking）
6. 回复播放 → 绿灯（speaking）

## 通信协议

帧格式（双向一致）:
```
[4字节 header 长度][JSON header][音频 PCM 数据]
```

音频参数：16kHz / 16bit / Mono

## VAD 参数调优

在 `main.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VAD_CHUNK_MS` | 200 | 采样间隔 |
| `VAD_SILENCE_THRESHOLD` | 800 | 静音 RMS 阈值 |
| `VAD_SPEECH_THRESHOLD` | 1500 | 说话 RMS 阈值 |
| `VAD_SILENCE_FRAMES` | 15 | 静音几帧后认为说完 (3秒) |
| `VAD_MAX_SPEECH_FRAMES` | 100 | 最大录音时长 (20秒) |
| `VAD_MIN_SPEECH_FRAMES` | 3 | 最短说话时长 (0.6秒) |

## 技术栈

| 组件 | 技术 |
|------|------|
| 硬件主控 | ESP32-S3 (MicroPython) |
| 通信协议 | WebSocket over WiFi |
| 语音识别 | whisper.cpp small (本地) |
| 大脑 | 智谱 GLM-5-Turbo |
| 语音合成 | Edge TTS (免费) |
| 音频转码 | ffmpeg |

## TODO

- [ ] 摄像头接线 + 视觉理解（代码已预留）
- [ ] OLED 屏幕显示表情
- [ ] 3D 打印外壳设计
- [ ] VAD 参数环境自适应
- [ ] 唤醒词本地检测（无需先 STT）

---

*Updated: 2026-07-01*
