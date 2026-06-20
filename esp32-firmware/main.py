"""
AI Robot - ESP32-S3 固件 (MicroPython)
========================================
功能: 麦克风采集 → WebSocket发送 → 接收指令 → 喇叭播放 + LED + 按钮
依赖: MicroPython v1.23+, 需要刷入带 I2S 支持的固件

接线:
  INMP441:  SCK→GPIO4, WS→GPIO5, SD→GPIO6, L/R→GND
  MAX98357A: BCLK→GPIO7, LRC→GPIO8, DIN→GPIO9
  WS2812:  DIN→GPIO38
  摄像头:  TX→GPIO14, RX→GPIO15
  按钮:    GPIO0 (用BOOT键复用)
"""

import machine
import network
import uasyncio as asyncio
import json
import struct
import time
import math
from machine import Pin, I2S

# ==================== 配置 ====================
WIFI_SSID = "Jerico"            # 改成你的WiFi名
WIFI_PASS = "gq850831"            # 改成你的WiFi密码
SERVER_URL = "ws://192.168.0.109:8765"  # 改成你电脑的IP (WLAN 192.168.0.109)

# 测试模式: "mic"=循环录音打印rms, "speaker"=播1kHz, "full"=完整机器人
TEST_MODE = "speaker"

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

            # WebSocket 升级握手 (Sec-WebSocket-Key 必须是 16 字节 base64)
            import ubinascii
            key_bytes = ubinascii.b2a_base64(b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10')[:-1]
            handshake = (
                b"GET / HTTP/1.1\r\n"
                b"Host: " + host.encode() + b":" + str(port).encode() + b"\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Key: " + key_bytes + b"\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"\r\n"
            )
            sock.send(handshake)
            resp = b""
            while b"\r\n\r\n" not in resp:
                chunk = sock.recv(1024)
                if not chunk:
                    raise Exception("WS handshake EOF")
                resp += chunk
            if b" 101 " not in resp.split(b"\r\n", 1)[0]:
                snippet = resp[:100].decode("utf-8", "replace")
                raise Exception("WS handshake failed: " + snippet)
            print("[WS] Handshake OK")
        except Exception as e:
            import sys
            sys.print_exception(e)
            print(f"[WS] Connection failed: {e}")
            self.led.show_error()
            return

        # 主循环: 录音 → 发送 → 接收 → 播放
        try:
            while self.connected:
                self.led.show_idle()
                print("[Robot] Waiting for trigger...")
                await asyncio.sleep(1)

                self.led.show_listening()
                print("[Mic] Recording...")
                audio_data = self.mic.read(RECORD_DURATION_MS)

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
                payload = struct.pack(">I", len(header_bytes)) + header_bytes + audio_data
                self._ws_send(sock, payload)
                print(f"[WS] Sent {len(payload)} bytes")

                resp = self._ws_recv(sock)
                if not resp:
                    print("[WS] Server disconnected")
                    break
                resp_header_len = struct.unpack(">I", resp[:4])[0]
                resp_header = json.loads(resp[4:4 + resp_header_len].decode())
                audio_size = resp_header.get("size", 0)
                if audio_size > 0:
                    start = 4 + resp_header_len
                    resp_audio = resp[start:start + audio_size]
                    self.led.show_speaking()
                    self.speaker.play(resp_audio)

                action = resp_header.get("action", None)
                if action == "led":
                    r = resp_header.get("r", 0)
                    g = resp_header.get("g", 0)
                    b = resp_header.get("b", 0)
                    self.led.set_all(r, g, b)

                print("[Robot] Response played")
        except Exception as e:
            import sys
            sys.print_exception(e)
            print(f"[WS] Loop error: {e}")
        finally:
            try:
                sock.close()
            except:
                pass

    def _ws_send(self, sock, payload: bytes):
        """包 WS 帧 (binary) 发送"""
        import os
        mask = os.urandom(4)
        L = len(payload)
        if L < 126:
            hdr = b'\x82' + bytes([0x80 | L])
        elif L < 65536:
            hdr = b'\x82' + bytes([0x80 | 126]) + struct.pack(">H", L)
        else:
            hdr = b'\x82' + bytes([0x80 | 127]) + struct.pack(">Q", L)
        out = bytearray(hdr)
        out.extend(mask)
        m = bytearray(L)
        for i in range(L):
            m[i] = payload[i] ^ mask[i % 4]
        out.extend(m)
        sock.send(bytes(out))

    def _ws_recv(self, sock) -> bytes:
        """收一帧 WS binary, 返回 payload"""
        hdr = self._recv_exact(sock, 2)
        if not hdr or len(hdr) < 2:
            return b""
        b1, b2 = hdr[0], hdr[1]
        opcode = b1 & 0x0F
        L = b2 & 0x7F
        if L == 126:
            ext = self._recv_exact(sock, 2)
            L = struct.unpack(">H", ext)[0]
        elif L == 127:
            ext = self._recv_exact(sock, 8)
            L = struct.unpack(">Q", ext)[0]
        payload = self._recv_exact(sock, L)
        return payload

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

    # 2b. 测试模式分支 (P1 验证, 不接服务器不接喇叭)
    if TEST_MODE == "mic":
        print("[TEST_MODE=mic] 5 cycles, record 1s + print rms.")
        for i in range(5):
            audio = mic.read(1000)
            samples = struct.unpack(f'<{int(SAMPLE_RATE*1)}h', audio)
            rms = int((sum(s*s for s in samples) / len(samples)) ** 0.5)
            mx, mn = max(samples), min(samples)
            print(f"  [{i+1}/5] rms={rms:5d}  max={mx:6d}  min={mn:6d}")
            time.sleep(0.1)
        print("[TEST_MODE=mic] done.")
        import sys
        sys.exit(0)

    if TEST_MODE == "speaker":
        print("[TEST_MODE=speaker] LONG 3s 1kHz on GPIO7/8/9")
        speaker = Speaker()
        # 1 个 3 秒长音
        freq = 1000
        amp = 32000  # 满量程
        n = SAMPLE_RATE * 3  # 3 秒
        wave = bytearray(n * 2)
        for i in range(n):
            s = int(amp * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
            wave[2*i] = s & 0xFF
            wave[2*i+1] = (s >> 8) & 0xFF
        print(f"  play {freq}Hz 3s, amp={amp}, {len(wave)} bytes")
        speaker.play(bytes(wave))
        print("  done. listen for 1kHz tone 3 seconds.")
        speaker.deinit()
        import sys
        sys.exit(0)

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
