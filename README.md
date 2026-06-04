# 🤖 AI Robot 项目

> 桌面AI机器人：ESP32-S3 硬件 + 电脑端 LLM 大脑

## 项目结构

```
ai-robot/
├── shopping-list.md          # 硬件购物清单
├── esp32-firmware/
│   └── main.py               # ESP32 MicroPython 固件
├── server/
│   └── server.py             # 电脑端 WebSocket 服务
└── README.md                 # 本文件
```

## 快速开始

### 1. 买配件
见 `shopping-list.md`，约 ¥100

### 2. 刷 ESP32 固件
```bash
# 安装 esptool
pip install esptool

# 下载 MicroPython 固件 (ESP32-S3)
# https://micropython.org/download/ESP32_GENERIC_S3/

# 刷入固件
esptool.py --chip esp32s3 -p COM3 write_flash -z 0x1000 ESP32_GENERIC_S3-*.bin

# 上传代码
pip install mpremote
mpremote cp esp32-firmware/main.py :main.py
```

### 3. 配置并启动服务器
```bash
# 安装依赖
pip install websockets openai edge-tts numpy

# 还需要安装 ffmpeg (音频转码用)
# Windows: winget install ffmpeg
# 或从 https://ffmpeg.org/download.html 下载

# 编辑 server.py 里的 API_KEY
# 编辑 main.py 里的 WIFI_SSID / WIFI_PASS / SERVER_URL

# 启动服务器
python server/server.py
```

### 4. 给 ESP32 上电
ESP32 连接 WiFi → 连接电脑 WebSocket → 开始对话！

## 通信协议

帧格式（双向一致）:
```
[4字节 header 长度][JSON header][音频 PCM 数据]
```

Header 示例:
```json
{
  "type": "audio",
  "sample_rate": 16000,
  "bits": 16,
  "channels": 1,
  "size": 96000
}
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 硬件主控 | ESP32-S3 (MicroPython) |
| 通信协议 | WebSocket over WiFi |
| 语音识别 | whisper.cpp (本地) / Whisper API |
| 大脑 | 智谱 GLM-5-Turbo |
| 语音合成 | Edge TTS (免费) |
| 音频转码 | ffmpeg |

## 后续可加功能

- [ ] 摄像头画面传输 + 视觉理解
- [ ] VAD（语音活动检测，自动开始/停止录音）
- [ ] 唤醒词检测（"小呆小呆"）
- [ ] OLED 屏幕显示表情
- [ ] 3D 打印外壳设计
- [ ] 连续对话（打断功能）
