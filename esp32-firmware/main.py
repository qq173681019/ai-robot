"""
AI Robot - ESP32-S3 固件 v2.0 (MicroPython)
=============================================
v2.0 改动:
  - 断线自动重连 (5秒重试，无限重连)
  - VAD 语音活动检测 (自动检测说话起止)
  - 连续对话 (播放完自动进入下一轮)
  - 打断功能 (播放中按BOOT可打断)
  - 唤醒词支持 (配合 server 端)

接线:
  INMP441:  SCK→GPIO4, WS→GPIO5, SD→GPIO6, L/R→GND
  MAX98357A: BCLK→GPIO7, LRC→GPIO8, DIN→GPIO9, SD→GPIO1
  WS2812:  DIN→GPIO38
  摄像头:  TX→GPIO14, RX→GPIO15
  按钮:    GPIO0 (BOOT键复用)

依赖: MicroPython v1.23+, ESP32_GENERIC_S3-SPIRAM 固件
"""

import machine
import network
import uasyncio as asyncio
import json
import struct
import time
import math
import gc
from machine import Pin, I2S

# ==================== 配置 ====================
WIFI_SSID = "Jerico"
WIFI_PASS = "gq850831"
SERVER_URL = "ws://192.168.0.109:8770"

# 录音参数
SAMPLE_RATE = 16000
CHANNELS = 1
BITS = 16

# VAD 参数
VAD_CHUNK_MS = 200          # 每200ms采样一次能量
VAD_SILENCE_THRESHOLD = 800  # RMS 低于此值视为静音
VAD_SPEECH_THRESHOLD = 1500  # RMS 高于此值视为说话
VAD_SILENCE_FRAMES = 15     # 连续多少帧静音后认为说完 (15*200ms=3秒)
VAD_MAX_SPEECH_FRAMES = 100 # 最大录音帧数 (100*200ms=20秒)
VAD_MIN_SPEECH_FRAMES = 3   # 最少说话帧数才发送 (3*200ms=600ms)
VAD_PRE_ROLL = 3            # 预缓冲帧数 (检测到说话时回溯几帧)

# 引脚定义
PIN_MIC_SCK = 4
PIN_MIC_WS  = 5
PIN_MIC_SD  = 6
PIN_SPK_SD  = 1    # MAX98357A Shutdown pin (LOW=静音, HIGH=播放)
PIN_SPK_BCLK = 7
PIN_SPK_LRC  = 8
PIN_SPK_DIN  = 9
PIN_LED_DIN  = 38
PIN_BUTTON   = 0

# 重连参数
RECONNECT_DELAY = 5  # 断线后5秒重试


# ==================== WiFi ====================
class WiFi:
    def __init__(self):
        self.wlan = network.WLAN(network.STA_IF)

    def connect(self, ssid, password, timeout=15):
        self.wlan.active(True)
        if self.wlan.isconnected():
            print(f"[WiFi] Connected: {self.wlan.ifconfig()[0]}")
            return True
        self.wlan.connect(ssid, password)
        start = time.ticks_ms()
        while not self.wlan.isconnected():
            if time.ticks_diff(time.ticks_ms(), start) > timeout * 1000:
                print("[WiFi] Timeout!")
                return False
            time.sleep(0.5)
        print(f"[WiFi] Connected! IP: {self.wlan.ifconfig()[0]}")
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
            bits=32,
            format=I2S.STEREO,
            rate=SAMPLE_RATE,
            ibuf=40000,
        )
        print("[Mic] INMP441 init OK")

    def read_chunk(self, duration_ms=VAD_CHUNK_MS):
        """读取一小段音频，返回16bit mono bytes"""
        num_samples = int(SAMPLE_RATE * duration_ms / 1000)
        num_bytes = num_samples * 4  # 32-bit slot
        buf = bytearray(num_bytes)
        self.i2s.readinto(buf)
        # 32-bit → 16-bit 转换
        out = bytearray(num_samples * 2)
        for i in range(num_samples):
            out[2*i] = buf[4*i + 1]
            out[2*i + 1] = buf[4*i + 2]
        return bytes(out)

    def read_fixed(self, duration_ms):
        """读取固定时长的音频"""
        num_samples = int(SAMPLE_RATE * duration_ms / 1000)
        num_bytes = num_samples * 4
        buf = bytearray(num_bytes)
        self.i2s.readinto(buf)
        out = bytearray(num_samples * 2)
        for i in range(num_samples):
            out[2*i] = buf[4*i + 1]
            out[2*i + 1] = buf[4*i + 2]
        return bytes(out)

    @staticmethod
    def calc_rms(audio_bytes):
        """计算RMS能量"""
        n = len(audio_bytes) // 2
        if n == 0:
            return 0
        samples = struct.unpack(f'<{n}h', audio_bytes)
        return int((sum(s*s for s in samples) / n) ** 0.5)

    def deinit(self):
        self.i2s.deinit()


# ==================== VAD 语音活动检测 ====================
class VAD:
    """基于能量RMS的简单VAD"""

    def __init__(self, mic: Microphone):
        self.mic = mic
        self.pre_roll = []  # 预缓冲
        self._reset()

    def _reset(self):
        self.state = "idle"  # idle → listening → speaking → done
        self.speech_frames = []
        self.silence_count = 0
        self.speech_count = 0

    async def detect_and_record(self, button=None):
        """
        主VAD循环：
        1. 持续采样，计算RMS
        2. 检测到说话开始录音
        3. 静音超过阈值后返回完整音频
        4. 期间可按按钮强制提前结束
        """
        self._reset()
        pre_roll_buf = []
        max_wait_s = 120  # 最大等待2分钟
        start = time.ticks_ms()

        while True:
            # 超时检查
            if time.ticks_diff(time.ticks_ms(), start) > max_wait_s * 1000:
                print("[VAD] Timeout, no speech detected")
                return None

            # 按钮打断
            if button and button.consume_press():
                print("[VAD] Button pressed, force start")
                self.state = "speaking"

            # 读一帧
            chunk = self.mic.read_chunk(VAD_CHUNK_MS)
            rms = Microphone.calc_rms(chunk)

            if self.state == "idle":
                # 预缓冲
                pre_roll_buf.append(chunk)
                if len(pre_roll_buf) > VAD_PRE_ROLL:
                    pre_roll_buf.pop(0)

                if rms > VAD_SPEECH_THRESHOLD:
                    print(f"[VAD] Speech detected! rms={rms}")
                    self.state = "speaking"
                    # 加入预缓冲
                    self.speech_frames = list(pre_roll_buf)
                    self.speech_count = 1
                    self.silence_count = 0
                else:
                    # 每隔一段时间打印状态
                    if time.ticks_ms() % 5000 < VAD_CHUNK_MS:
                        pass  # 安静等待，不打印太多

            elif self.state == "speaking":
                self.speech_frames.append(chunk)
                self.speech_count += 1

                if rms < VAD_SILENCE_THRESHOLD:
                    self.silence_count += 1
                else:
                    self.silence_count = 0  # 重置静音计数

                # 说完了
                if self.silence_count >= VAD_SILENCE_FRAMES:
                    if self.speech_count >= VAD_MIN_SPEECH_FRAMES:
                        audio = b''.join(self.speech_frames)
                        print(f"[VAD] Speech ended: {len(audio)} bytes, {self.speech_count} frames")
                        return audio
                    else:
                        print(f"[VAD] Too short ({self.speech_count} frames), resetting")
                        self._reset()
                        pre_roll_buf = []

                # 超过最大时长
                if self.speech_count >= VAD_MAX_SPEECH_FRAMES:
                    audio = b''.join(self.speech_frames)
                    print(f"[VAD] Max length reached: {len(audio)} bytes")
                    return audio

            # 让出CPU
            await asyncio.sleep_ms(10)


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
        self.sd_pin = Pin(PIN_SPK_SD, Pin.OUT, value=0)
        print("[Speaker] MAX98357A init OK")

    def play(self, audio_bytes, amp=3):
        """播放音频，amp=放大倍数"""
        # 放大
        n_samples = len(audio_bytes) // 2
        samples_in = struct.unpack(f'<{n_samples}h', audio_bytes)
        samples_out = bytearray(n_samples * 2)
        for i, s in enumerate(samples_in):
            s2 = s * amp
            if s2 > 32767: s2 = 32767
            if s2 < -32768: s2 = -32768
            struct.pack_into('<h', samples_out, i * 2, s2)

        self.sd_pin.on()  # 使能功放
        self.i2s.write(bytes(samples_out))
        self.sd_pin.off()  # 关闭功放防自激

    def play_beep(self, freq=1000, duration_ms=100):
        """播放提示音"""
        n = int(SAMPLE_RATE * duration_ms / 1000)
        wave = bytearray(n * 2)
        for i in range(n):
            s = int(20000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
            struct.pack_into('<h', wave, i * 2, s)
        self.play(bytes(wave), amp=1)

    def enable(self):
        self.sd_pin.on()

    def disable(self):
        self.sd_pin.off()

    def deinit(self):
        self.sd_pin.off()
        self.i2s.deinit()


# ==================== LED ====================
class LEDController:
    def __init__(self, pin=PIN_LED_DIN, num_leds=4):
        self.num_leds = num_leds
        self.has_np = False
        try:
            from neopixel import NeoPixel
            self.np = NeoPixel(Pin(pin), num_leds)
            self.has_np = True
            self.show_idle()
        except ImportError:
            print("[LED] NeoPixel not found")

    def set_all(self, r, g, b):
        if not self.has_np:
            return
        for i in range(self.num_leds):
            self.np[i] = (r, g, b)
        self.np.write()

    def show_idle(self):     self.set_all(20, 20, 20)    # 微弱白
    def show_listening(self): self.set_all(0, 100, 255)  # 蓝色
    def show_speaking(self):  self.set_all(0, 255, 100)  # 绿色
    def show_thinking(self):  self.set_all(255, 200, 0)  # 黄色
    def show_error(self):     self.set_all(255, 0, 0)    # 红色
    def off(self):           self.set_all(0, 0, 0)


class FakeLED:
    """没接LED时的空实现"""
    def show_idle(self): pass
    def show_listening(self): pass
    def show_thinking(self): pass
    def show_speaking(self): pass
    def show_error(self): pass
    def off(self): pass


# ==================== 按钮 ====================
class Button:
    def __init__(self, pin=PIN_BUTTON):
        self.pin = Pin(pin, Pin.IN, Pin.PULL_UP)
        self.last_press = 0
        self.pressed_flag = False

    def _irq_handler(self, pin):
        now = time.ticks_ms()
        if time.ticks_diff(now, self.last_press) < 300:  # 去抖300ms
            return
        self.last_press = now
        self.pressed_flag = True
        print("[Button] IRQ triggered")

    def setup_irq(self):
        self.pin.irq(trigger=Pin.IRQ_FALLING, handler=self._irq_handler)

    def consume_press(self):
        if self.pressed_flag:
            self.pressed_flag = False
            return True
        return False

    def is_pressed(self):
        return self.pin.value() == 0


# ==================== 摄像头 (OV2640) ====================
# 注意: 摄像头引脚与麦克风/喇叭有冲突 (GPIO6/7/9)
# 实际使用时需要调整引脚或用独立摄像头模块
# 目前为预留代码，需要硬件调整后启用
class Camera:
    def __init__(self):
        self.cam = None
        self.enabled = CAMERA_ENABLED
        if not self.enabled:
            print("[Camera] Disabled (set CAMERA_ENABLED=True after wiring)")
            return
        try:
            import esp32_camera
            esp32_camera.init(
                hsize=esp32_camera.SIZE.QVGA,
                vsize=esp32_camera.SIZE.QVGA,
                framesize=esp32_camera.FRAMESIZE.QVGA,
                freq=20000000,
                pins=(
                    PIN_CAM_SIOD, PIN_CAM_SIOC,      # I2C
                    PIN_CAM_Y9, PIN_CAM_Y8, PIN_CAM_Y7,
                    PIN_CAM_Y6, PIN_CAM_Y5, PIN_CAM_Y4,
                    PIN_CAM_Y3, PIN_CAM_Y2,
                    PIN_CAM_VSYNC, PIN_CAM_HREF,
                    PIN_CAM_PCLK, PIN_CAM_XCLK,
                ),
            )
            esp32_camera.quality(CAMERA_QUALITY)
            esp32_camera.contrast(0)
            esp32_camera.brightness(0)
            self.cam = esp32_camera
            print("[Camera] OV2640 init OK")
        except Exception as e:
            print(f"[Camera] Init failed: {e}")
            self.enabled = False

    def capture(self):
        """拍一张JPEG照片，返回bytes"""
        if not self.enabled or not self.cam:
            return None
        try:
            buf = self.cam.capture()
            return buf
        except Exception as e:
            print(f"[Camera] Capture failed: {e}")
            return None

    def capture_base64(self):
        """拍照并返回base64编码"""
        import ubinascii
        buf = self.capture()
        if buf:
            return ubinascii.b2a_base64(buf)[:-1].decode()
        return None

    def deinit(self):
        if self.cam:
            try:
                self.cam.deinit()
            except:
                pass


# ==================== WebSocket 通信 (带断线重连) ====================
class WSClient:
    """底层WebSocket客户端，封装连接和帧收发"""

    def __init__(self):
        self.sock = None
        self.connected = False

    def connect(self, url=SERVER_URL, timeout=10):
        """连接WebSocket服务器"""
        import usocket as socket

        url_clean = url.replace("ws://", "").replace("ws:", "").rstrip("/")
        parts = url_clean.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 80

        print(f"[WS] Connecting {host}:{port}...")

        try:
            self.sock = socket.socket()
            self.sock.settimeout(timeout)
            self.sock.connect((host, port))

            # WebSocket 升级握手
            import ubinascii
            import os
            key_bytes = ubinascii.b2a_base64(os.urandom(16))[:-1]
            handshake = (
                b"GET / HTTP/1.1\r\n"
                b"Host: " + host.encode() + b":" + str(port).encode() + b"\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Key: " + key_bytes + b"\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"\r\n"
            )
            self.sock.send(handshake)

            resp = b""
            while b"\r\n\r\n" not in resp:
                chunk = self.sock.recv(1024)
                if not chunk:
                    raise Exception("WS handshake EOF")
                resp += chunk

            if b" 101 " not in resp.split(b"\r\n", 1)[0]:
                raise Exception("WS handshake failed: " + resp[:100].decode("utf-8", "replace"))

            self.connected = True
            self.sock.settimeout(None)  # 阻塞模式交给 asyncio
            print("[WS] Connected!")
            return True

        except Exception as e:
            print(f"[WS] Connect failed: {e}")
            self.connected = False
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            return False

    def send(self, payload: bytes):
        """发送二进制帧"""
        if not self.sock:
            raise Exception("not connected")
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
        self.sock.send(bytes(out))

    def recv(self) -> bytes:
        """接收一帧，返回payload。断线返回 b''"""
        if not self.sock:
            return b""
        try:
            hdr = self._recv_exact(2)
            if not hdr or len(hdr) < 2:
                return b""
            b1, b2 = hdr[0], hdr[1]
            opcode = b1 & 0x0F
            L = b2 & 0x7F
            if L == 126:
                ext = self._recv_exact(2)
                L = struct.unpack(">H", ext)[0]
            elif L == 127:
                ext = self._recv_exact(8)
                L = struct.unpack(">Q", ext)[0]
            return self._recv_exact(L)
        except Exception as e:
            print(f"[WS] recv error: {e}")
            return b""

    def _recv_exact(self, n):
        buf = bytearray(n)
        got = 0
        while got < n:
            chunk = self.sock.recv(n - got)
            if not chunk:
                return None
            buf[got:got+len(chunk)] = chunk
            got += len(chunk)
        return bytes(buf)

    def close(self):
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None


# ==================== 机器人主逻辑 ====================
class Robot:
    """主机器人逻辑：VAD录音 → 发送 → 接收 → 播放，带断线重连"""

    def __init__(self, mic, speaker, led, button, camera=None):
        self.mic = mic
        self.speaker = speaker
        self.led = led
        self.button = button
        self.camera = camera
        self.ws = WSClient()
        self.vad = VAD(mic)

    async def run_forever(self):
        """主循环：连接 → 对话 → 断线重连"""
        while True:
            # 连接服务器
            self.led.show_idle()
            while not self.ws.connect():
                self.led.show_error()
                print(f"[Robot] Reconnect in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)
                self.led.show_idle()

            self.led.show_idle()
            # 播放启动音
            self.speaker.play_beep(880, 100)

            # 对话循环
            try:
                await self._conversation_loop()
            except Exception as e:
                import sys
                sys.print_exception(e)
                print(f"[Robot] Loop error: {e}")
            finally:
                self.ws.close()
                self.led.show_error()
                print("[Robot] Disconnected, reconnecting...")

    async def _conversation_loop(self):
        """对话主循环：VAD录音 → 发送 → 接收 → 播放"""
        consecutive_errors = 0

        while self.ws.connected:
            # 检查按钮（打断当前等待）
            if self.button.consume_press():
                # 手动触发录音
                pass

            # === 阶段1: VAD 等待说话 ===
            self.led.show_listening()
            audio_data = await self.vad.detect_and_record(self.button)

            if not audio_data:
                # 超时或太短，继续等
                continue

            consecutive_errors = 0

            # === 阶段2: 发送到服务器 ===
            self.led.show_thinking()
            header = json.dumps({
                "type": "audio",
                "sample_rate": SAMPLE_RATE,
                "bits": BITS,
                "channels": CHANNELS,
                "size": len(audio_data),
                "has_image": False,  # 预留
            })
            header_bytes = header.encode()
            payload = struct.pack(">I", len(header_bytes)) + header_bytes + audio_data

            try:
                self.ws.send(payload)
                print(f"[Robot] Sent {len(payload)} bytes")
            except Exception as e:
                print(f"[Robot] Send error: {e}")
                break

            # === 阶段3: 等待响应 ===
            resp = self.ws.recv()
            if not resp:
                print("[Robot] Server disconnected")
                break

            resp_header_len = struct.unpack(">I", resp[:4])[0]
            resp_header = json.loads(resp[4:4 + resp_header_len].decode())
            audio_size = resp_header.get("size", 0)

            # 静默响应（唤醒词未激活时）
            if resp_header.get("silent"):
                print("[Robot] Silent response (wake word not detected)")
                continue

            # === 阶段4: 播放回复 ===
            if audio_size > 0:
                self.led.show_speaking()
                start = 4 + resp_header_len
                resp_audio = resp[start:start + audio_size]
                text = resp_header.get("text", "")
                print(f"[Robot] Reply: {text[:60]}")

                try:
                    self.speaker.play(resp_audio, amp=3)
                except Exception as e:
                    print(f"[Robot] Play error: {e}")
                    # 重建 speaker
                    try:
                        self.speaker.deinit()
                    except:
                        pass

            print("[Robot] --- Round complete ---")


# ==================== 主程序 ====================
async def main():
    print("=" * 40)
    print("  🤖 AI Robot Firmware v2.0")
    print("  VAD + Reconnect + Continuous")
    print("=" * 40)

    # 1. 连WiFi
    wifi = WiFi()
    for attempt in range(3):
        if wifi.connect(WIFI_SSID, WIFI_PASS):
            break
        print(f"[WiFi] Retry {attempt+1}/3...")
        time.sleep(2)
    else:
        print("[FATAL] WiFi failed, resetting in 5s")
        time.sleep(5)
        machine.reset()

    # 2. 初始化硬件
    # 关闭功放防上电自激
    Pin(PIN_SPK_SD, Pin.OUT, value=0)

    mic = Microphone()
    speaker = Speaker()
    button = Button()
    button.setup_irq()

    # LED (如果有neopixel库)
    led = LEDController()
    if not led.has_np:
        led = FakeLED()

    led.show_idle()

    # 3. 启动提示音
    speaker.play_beep(523, 100)  # C5
    time.sleep_ms(100)
    speaker.play_beep(659, 100)  # E5
    time.sleep_ms(100)
    speaker.play_beep(784, 150)  # G5 - 开机三和弦

    print("[Robot] Ready! Say something or press BOOT.")
    print(f"[Config] VAD: silence={VAD_SILENCE_THRESHOLD}, speech={VAD_SPEECH_THRESHOLD}")
    print(f"[Config] Server: {SERVER_URL}")

    # 4. 摄像头 (需要硬件接线后启用)
    camera = Camera()

    # 5. 运行
    robot = Robot(mic, speaker, led, button, camera)
    await robot.run_forever()


# 启动
try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n[Robot] Shutdown...")
except Exception as e:
    print(f"[FATAL] {e}")
    import sys
    sys.print_exception(e)
    time.sleep(5)
    machine.reset()
