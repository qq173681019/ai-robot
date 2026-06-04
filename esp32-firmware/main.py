"""
AI Robot - ESP32-S3 固件 (MicroPython)
========================================
功能: 麦克风采集 → WebSocket发送 → 接收指令 → 喇叭播放 + LED + 按钮
依赖: MicroPython v1.23+, 需要刷入带 I2S 支持的固件

接线:
  INMP441:  SCK→GPIO4, WS→GPIO5, SD→GPIO6, L/R→GND
  MAX98357A: BCLK→GPIO7, LRC→GPIO8, DIN→GPIO9
  WS2812:  DIN→GPIO38
  按钮:    GPIO0 (用BOOT键复用)
"""

import machine
import network
import uasyncio as asyncio
import json
import struct
import time
from machine import Pin, I2S

# ==================== 配置 ====================
WIFI_SSID = "YOUR_WIFI_SSID"      # 改成你的WiFi名
WIFI_PASS = "YOUR_WIFI_PASSWORD"  # 改成你的WiFi密码
SERVER_URL = "ws://192.168.1.100:8765"  # 改成你电脑的IP

# 引脚定义
PIN_MIC_SCK = 4
PIN_MIC_WS  = 5
PIN_MIC_SD  = 6
PIN_SPK_BCLK = 7
PIN_SPK_LRC  = 8
PIN_SPK_DIN  = 9
PIN_LED_DIN  = 38
PIN_BUTTON   = 0

# 音频参数
SAMPLE_RATE = 16000
RECORD_DURATION_MS = 3000  # 每次录音3秒
CHANNELS = 1
BITS = 16

# ==================== WiFi ====================
class WiFi:
    def __init__(self):
        self.wlan = network.WLAN(network.STA_IF)

    def connect(self, ssid, password, timeout=15):
        self.wlan.active(True)
        if self.wlan.isconnected():
            print(f"[WiFi] Already connected: {self.wlan.ifconfig()[0]}")
            return True
        self.wlan.connect(ssid, password)
        start = time.time()
        while not self.wlan.isconnected():
            if time.time() - start > timeout:
                print("[WiFi] Connection timeout!")
                return False
            time.sleep(0.5)
        ip = self.wlan.ifconfig()[0]
        print(f"[WiFi] Connected! IP: {ip}")
        return True

    def is_connected(self):
        return self.wlan.isconnected()

# ==================== 麦克风 ====================
class Microphone:
    def __init__(self):
        self.i2s = I2S(
            0,
            sck=Pin(PIN_MIC_SCK),
            ws=Pin(PIN_MIC_WS),
            sd=Pin(PIN_MIC_SD),
            mode=I2S.RX,
            bits=BITS,
            format=I2S.MONO,
            rate=SAMPLE_RATE,
            ibuf=16000,
        )
        print("[Mic] INMP441 initialized")

    def read(self, duration_ms=RECORD_DURATION_MS):
        """录音指定时长，返回bytes"""
        num_samples = int(SAMPLE_RATE * duration_ms / 1000)
        num_bytes = num_samples * 2  # 16bit = 2 bytes per sample
        buf = bytearray(num_bytes)
        self.i2s.readinto(buf)
        print(f"[Mic] Recorded {duration_ms}ms ({len(buf)} bytes)")
        return bytes(buf)

    def deinit(self):
        self.i2s.deinit()

# ==================== 喇叭 ====================
class Speaker:
    def __init__(self):
        self.i2s = I2S(
            1,
            sck=Pin(PIN_SPK_BCLK),
            ws=Pin(PIN_SPK_LRC),
            sd=Pin(PIN_SPK_DIN),
            mode=I2S.TX,
            bits=BITS,
            format=I2S.MONO,
            rate=SAMPLE_RATE,
            ibuf=16000,
        )
        print("[Speaker] MAX98357A initialized")

    def play(self, audio_bytes):
        """播放音频数据"""
        self.i2s.write(audio_bytes)
        print(f"[Speaker] Played {len(audio_bytes)} bytes")

    def deinit(self):
        self.i2s.deinit()

# ==================== LED ====================
class LEDController:
    def __init__(self, pin=PIN_LED_DIN, num_leds=4):
        self.num_leds = num_leds
        self.pin = Pin(pin)
        self.pixels = [(0, 0, 0)] * num_leds
        # WS2812 用 machine.PWM 模拟或 neopixel 库
        try:
            from neopixel import NeoPixel
            self.np = NeoPixel(self.pin, num_leds)
            self.has_np = True
        except ImportError:
            self.has_np = False
            print("[LED] NeoPixel lib not found, LED disabled")

    def set_color(self, index, r, g, b):
        """设置单个LED颜色"""
        if not self.has_np or index >= self.num_leds:
            return
        self.np[index] = (r, g, b)
        self.np.write()

    def set_all(self, r, g, b):
        """设置所有LED"""
        if not self.has_np:
            return
        for i in range(self.num_leds):
            self.np[i] = (r, g, b)
        self.np.write()

    def off(self):
        self.set_all(0, 0, 0)

    # 预设表情
    def show_listening(self):
        """蓝色呼吸灯"""
        self.set_all(0, 100, 255)

    def show_speaking(self):
        """绿色"""
        self.set_all(0, 255, 100)

    def show_thinking(self):
        """黄色闪烁"""
        self.set_all(255, 200, 0)

    def show_error(self):
        """红色"""
        self.set_all(255, 0, 0)

    def show_idle(self):
        """微弱白色"""
        self.set_all(20, 20, 20)

# ==================== 按钮 ====================
class Button:
    def __init__(self, pin=PIN_BUTTON):
        self.pin = Pin(pin, Pin.IN, Pin.PULL_UP)
        self.last_press = 0
        self.callback = None

    def is_pressed(self):
        return self.pin.value() == 0

    def set_callback(self, cb):
        self.callback = cb
        # 使用轮询方式检测，ESP32中断有时不稳定
        self.pin.irq(trigger=Pin.IRQ_FALLING, handler=self._irq_handler)

    def _irq_handler(self, pin):
        now = time.ticks_ms()
        if time.ticks_diff(now, self.last_press) < 300:  # 去抖
            return
        self.last_press = now
        if self.callback:
            self.callback()

# ==================== WebSocket 通信 ====================
class RobotClient:
    def __init__(self, mic, speaker, led):
        self.mic = mic
        self.speaker = speaker
        self.led = led
        self.is_listening = False
        self.connected = False

    async def connect_and_run(self):
        """主循环: 连接服务器 → 录音 → 发送 → 接收 → 播放"""
        import usocket as socket

        # 解析服务器地址
        # ws://192.168.1.100:8765
        url = SERVER_URL.replace("ws://", "").replace("ws:", "")
        parts = url.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8765

        print(f"[WS] Connecting to {host}:{port}...")

        try:
            sock = socket.socket()
            sock.connect((host, port))
            self.connected = True
            print("[WS] Connected!")
            self.led.show_idle()
        except Exception as e:
            print(f"[WS] Connection failed: {e}")
            self.led.show_error()
            return

        # 简易 TCP 帧协议:
        # 发送: [4字节长度][JSON头][音频数据]
        # 接收: [4字节长度][JSON头][音频数据]

        try:
            while self.connected:
                # 等待触发（按钮按下或持续监听模式）
                self.led.show_idle()
                print("[Robot] Waiting for trigger...")

                # TODO: 这里可以改成按钮触发或VAD（语音活动检测）
                # 目前用简单定时循环测试
                await asyncio.sleep(1)

                # 录音
                self.led.show_listening()
                print("[Mic] Recording...")
                audio_data = self.mic.read(RECORD_DURATION_MS)

                # 发送音频到服务器
                self.led.show_thinking()
                header = json.dumps({
                    "type": "audio",
                    "sample_rate": SAMPLE_RATE,
                    "bits": BITS,
                    "channels": CHANNELS,
                    "duration_ms": RECORD_DURATION_MS,
                    "size": len(audio_data),
                })
                header_bytes = header.encode()

                # 帧格式: [4字节header长度][header][音频]
                frame = struct.pack(">I", len(header_bytes)) + header_bytes + audio_data
                sock.send(frame)
                print(f"[WS] Sent {len(frame)} bytes")

                # 接收服务器响应
                # 先读4字节header长度
                resp_len_bytes = self._recv_exact(sock, 4)
                if not resp_len_bytes:
                    print("[WS] Server disconnected")
                    break
                resp_header_len = struct.unpack(">I", resp_len_bytes)[0]

                # 读header
                resp_header_bytes = self._recv_exact(sock, resp_header_len)
                if not resp_header_bytes:
                    break
                resp_header = json.loads(resp_header_bytes.decode())

                # 读音频数据
                audio_size = resp_header.get("size", 0)
                if audio_size > 0:
                    resp_audio = self._recv_exact(sock, audio_size)
                    self.led.show_speaking()
                    self.speaker.play(resp_audio)

                # 处理动作指令
                action = resp_header.get("action", None)
                if action == "led":
                    r = resp_header.get("r", 0)
                    g = resp_header.get("g", 0)
                    b = resp_header.get("b", 0)
                    self.led.set_all(r, g, b)

                print("[Robot] Response played ✓")

        except Exception as e:
            print(f"[WS] Error: {e}")
            self.led.show_error()
        finally:
            sock.close()
            self.connected = False

    def _recv_exact(self, sock, n):
        """精确接收n个字节"""
        buf = bytearray(n)
        received = 0
        while received < n:
            chunk = sock.recv(n - received)
            if not chunk:
                return None
            buf[received:received+len(chunk)] = chunk
            received += len(chunk)
        return bytes(buf)

# ==================== 主程序 ====================
async def main():
    print("=" * 40)
    print("  🤖 AI Robot Firmware v1.0")
    print("  ESP32-S3 + INMP441 + MAX98357A")
    print("=" * 40)

    # 1. 连接WiFi
    wifi = WiFi()
    if not wifi.connect(WIFI_SSID, WIFI_PASS):
        print("[FATAL] WiFi connection failed, restarting in 5s...")
        time.sleep(5)
        machine.reset()

    # 2. 初始化硬件
    mic = Microphone()
    speaker = Speaker()
    led = LEDController()
    button = Button()

    # 3. 按钮回调
    def on_button_press():
        print("[Button] Pressed! Starting recording...")
        # 按钮按下时触发录音，在主循环里处理
        # TODO: 用事件通知主循环

    button.set_callback(on_button_press)

    # 4. 启动LED
    led.show_idle()

    # 5. 连接服务器并运行
    robot = RobotClient(mic, speaker, led)
    await robot.connect_and_run()

# 启动
try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("[Robot] Shutting down...")
except Exception as e:
    print(f"[FATAL] {e}")
    import sys
    sys.print_exception(e)
