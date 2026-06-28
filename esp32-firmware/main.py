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
SERVER_URL = "ws://192.168.0.109:8770"  # 改成你电脑的IP (WLAN 192.168.0.109), 8766/8765 都被 WSL NAT 残留绑, 用 8770

# 测试模式: "mic"=循环录音打印rms, "speaker"=播1kHz, "loopback"=mic→喇叭, "full"=完整机器人, "none"=什么都不做(只打印提示)
TEST_MODE = "full"  # 全栈模式: 录音→STT→LLM→TTS→播放
AUTO_RECORD_DELAY_S = 0  # 0 = 不等, boot 完立刻开始录 (下面 RECORD_DURATION_MS 控制录音时长)
RECORD_DURATION_MS = 10000  # 录音窗口 10 秒, 给你足够时间说话

# 引脚定义
PIN_MIC_SCK = 4
PIN_MIC_WS  = 5
PIN_MIC_SD  = 6
PIN_SPK_SD  = 1   # SD = 功放使能控制 (Shutdown)
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
        # INMP441 输出 24-bit 数据填入 32-bit slot, MicroPython I2S 必须用 bits=32 才能正确接收
        self.i2s = I2S(
            0,
            sck=Pin(PIN_MIC_SCK),
            ws=Pin(PIN_MIC_WS),
            sd=Pin(PIN_MIC_SD),
            mode=I2S.RX,
            bits=32,
            format=I2S.STEREO,  # INMP441 L/R=GND 选左声道, MONO/STEREO 都行
            rate=SAMPLE_RATE,
            ibuf=40000,
        )
        print("[Mic] INMP441 initialized (bits=32, format=STEREO)")

    def read(self, duration_ms=RECORD_DURATION_MS):
        """录音指定时长，返回bytes (16bit mono)"""
        num_samples = int(SAMPLE_RATE * duration_ms / 1000)
        # I2S 32-bit slot 模式下, 每样本 4 字节 (高 24 位是 mic 数据, 低 8 位是 0)
        num_bytes = num_samples * 4
        buf = bytearray(num_bytes)
        self.i2s.readinto(buf)
        # 把 32-bit (实际 24-bit 左对齐) 转换成 16-bit PCM
        # 每 4 字节取前 2 字节 (高 16 位 = mic 的有效数据)
        out = bytearray(num_samples * 2)
        for i in range(num_samples):
            # 小端序: [byte0, byte1, byte2, byte3] -> 取 [byte1, byte2] 作为 16-bit 样本
            # INMP441 24-bit 左对齐: 在 32-bit slot 中, 数据在 [byte1, byte2, byte3, 0]
            # 取中 16 位 [byte1, byte2]
            out[2*i] = buf[4*i + 1]
            out[2*i + 1] = buf[4*i + 2]
        print(f"[Mic] Recorded {duration_ms}ms ({len(out)} bytes after conversion)")
        return bytes(out)

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
        self.pressed_flag = False  # ISR 设置, 主循环 consume

    def is_pressed(self):
        return self.pin.value() == 0

    def set_callback(self, cb):
        self.callback = cb
        # 使用轮询方式检测，ESP32中断有时不稳定
        self.pin.irq(trigger=Pin.IRQ_FALLING, handler=self._irq_handler)

    def _irq_handler(self, pin):
        # ISR 里只设置 volatile 标志, 不调 Python 回调
        # (MicroPython 在 ISR 里调 asyncio.Event.set() 不可靠, 会丢失事件)
        now = time.ticks_ms()
        if time.ticks_diff(now, self.last_press) < 300:  # 去抖
            return
        self.last_press = now
        self.pressed_flag = True

    def consume_press(self):
        """主循环调用: 返回 True 表示有按钮按下 (一次性)"""
        if self.pressed_flag:
            self.pressed_flag = False
            return True
        return False

# ==================== WebSocket 通信 ====================
class RobotClient:
    def __init__(self, mic, speaker, led):
        self.mic = mic
        self.speaker = speaker
        self.led = led
        self.is_listening = False
        self.connected = False

    async def connect_and_run(self, button=None):
        """主循环: 连接服务器 → 等按钮 → 录音 → 发送 → 接收 → 播放"""
        import usocket as socket
        import struct as _struct  # 强制 local, 避开 MicroPython async 编译器的 global/local 错判

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

        # 主循环: 等提示音(3秒) → 录音 → 噪音门判断 → 发送 → 接收 → 播放
        # 噪音/太安静就 continue 跳过, 不发给 server
        try:
            while True:
                self.led.show_idle()
                if button is not None:
                    if AUTO_RECORD_DELAY_S > 0:
                        print(f"[Robot] Press BOOT or wait {AUTO_RECORD_DELAY_S}s...")
                        triggered = False
                        for _tick in range(AUTO_RECORD_DELAY_S * 20):
                            if button.consume_press():
                                print("[Robot] Button pressed!")
                                triggered = True
                                break
                            await asyncio.sleep_ms(50)
                        if not triggered:
                            print(f"[Robot] Auto-record after {AUTO_RECORD_DELAY_S}s")
                    elif button is not None:
                        # BOOT 坏了, 不等按钮, 等 3 秒让你走到 mic 前
                        print(f"[Robot] Recording in 3s (BOOT 硬件坏, 跳过按钮等待)...")
                        await asyncio.sleep(3)
                        print(f"[Robot] Auto-record now")
                    else:
                        print("[Robot] Press BOOT to talk...")
                        while not button.consume_press():
                            await asyncio.sleep_ms(50)
                        print("[Robot] Button pressed!")
                else:
                    await asyncio.sleep(1)

            self.led.show_listening()
            print("[Mic] Recording...")
            # 播一个 beep 作为录音开始提示, 你听到 beep 就开始说话
            try:
                _sd_beep = Pin(PIN_SPK_SD, Pin.OUT); _sd_beep.on()
                _beep = bytearray()
                for _i in range(1600):  # 1600 样本 = 100ms at 16kHz
                    _s = int(20000 * math.sin(2 * math.pi * 1000 * _i / 16000))
                    _beep.extend(_struct.pack('<h', _s))
                self.speaker.play(bytes(_beep))
                _sd_beep.off()
                print("[Beep] 录音开始提示音 (1kHz 100ms)")
            except Exception as _e:
                print(f"[Beep] err: {_e}")
            audio_data = self.mic.read(RECORD_DURATION_MS)
            # 打印 RMS 判断 mic 录音有没有声音
            _samples_for_rms = _struct.unpack(f'<{len(audio_data)//2}h', audio_data)
            _rms = int((sum(s*s for s in _samples_for_rms) / len(_samples_for_rms)) ** 0.5)
            _mx = max(_samples_for_rms); _mn = min(_samples_for_rms)
            print(f"[Mic] rms={_rms} peak_max={_mx} peak_min={_mn}")

            # 噪音门: RMS < 阈值 = 静音/噪音, 跳过不上传, 避免 LLM 胡说八道循环
            if _rms < 500:
                print(f"[Mic] 太安静 (rms={_rms} < 500), 跳过本次录音, 等你说")
                await asyncio.sleep(1)  # 短暂延迟避免紧接重录
                continue  # 直接进下一轮主循环, 安静等你

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
            payload = _struct.pack(">I", len(header_bytes)) + header_bytes + audio_data
            self._ws_send(sock, payload)
            print(f"[WS] Sent {len(payload)} bytes")

            resp = self._ws_recv(sock)
            if not resp:
                print("[WS] Server disconnected")
                return  # 整个连接失败直接退出, 不再循环
            resp_header_len = _struct.unpack(">I", resp[:4])[0]
            resp_header = json.loads(resp[4:4 + resp_header_len].decode())
            audio_size = resp_header.get("size", 0)
            if audio_size > 0:
                start = 4 + resp_header_len
                resp_audio = resp[start:start + audio_size]
                try:
                    self.led.show_speaking()
                except: pass
                try:
                    print(f"[Robot] Playing {audio_size} bytes audio...")
                    # 拉高 SD pin 使能功放 (整个 main() 流程 SD 是 LOW, 必须显式拉高才能发声)
                    _sd = Pin(PIN_SPK_SD, Pin.OUT)
                    _sd.on()
                    # 放大 3x 防止 AI 回复声音小
                    amp = 3
                    n_samples = audio_size // 2
                    samples_in = _struct.unpack(f'<{n_samples}h', resp_audio)
                    samples_out = []
                    for s in samples_in:
                        s2 = s * amp
                        if s2 > 32767: s2 = 32767
                        if s2 < -32768: s2 = -32768
                        samples_out.append(s2)
                    resp_audio_amp = struct.pack(f'<{n_samples}h', *samples_out)
                    self.speaker.play(resp_audio_amp)
                    _sd.off()  # 播放完拉低, 防自激
                    print(f"[Robot] Play done (3x amp)")
                except Exception as play_err:
                    print(f"[Robot] play error: {play_err}")
                    # 重建 I2S speaker
                    try:
                        self.speaker.deinit()
                    except: pass
                    self.speaker = Speaker()

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

    # 1.5 立刻拉低 SD pin 关闭功放, 防止上电自激
    try:
        Pin(PIN_SPK_SD, Pin.OUT, value=0)
        print(f"[Init] SD pin (GPIO{PIN_SPK_SD}) forced LOW (amp disabled)")
    except Exception as e:
        print(f"[Init] SD pin init failed: {e}")

    # 2. 初始化硬件
    if TEST_MODE == "none":
        print("[TEST_MODE=none] idle, no test running. set TEST_MODE=mic/speaker/loopback/full then re-run.")
        import sys
        sys.exit(0)
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
        # SD pin 控制: 拉高使能功放
        sd_pin = Pin(PIN_SPK_SD, Pin.OUT)
        sd_pin.on()  # 使能 MAX98357A
        print(f"  SD (GPIO{PIN_SPK_SD}) = HIGH, amp enabled")
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
        sd_pin.off()  # 关闭功放
        print(f"  SD (GPIO{PIN_SPK_SD}) = LOW, amp disabled")
        time.sleep(0.3)  # 让声波播完
        machine.reset()  # 软重启, 释放所有资源

    if TEST_MODE == "loopback":
        print("[TEST_MODE=loopback] mic 1s -> speaker 1s (3 cycles) 1x amp")
        sd_pin = Pin(PIN_SPK_SD, Pin.OUT)
        sd_pin.on()
        print(f"  SD (GPIO{PIN_SPK_SD}) = HIGH, amp enabled")
        speaker = Speaker()
        AMP_FACTOR = 1  # 不放大, 避免啸叫
        try:
            for i in range(3):
                print(f"  [{i+1}/3] record 1s ... (say something!)")
                audio = mic.read(1000)
                samples_in = struct.unpack(f'<{int(SAMPLE_RATE*1)}h', audio)
                if AMP_FACTOR != 1:
                    samples_out = []
                    for s in samples_in:
                        s2 = s * AMP_FACTOR
                        if s2 > 32767: s2 = 32767
                        if s2 < -32768: s2 = -32768
                        samples_out.append(s2)
                    audio_amp = struct.pack(f'<{int(SAMPLE_RATE*1)}h', *samples_out)
                else:
                    audio_amp = audio
                rms = int((sum(s*s for s in samples_in) / len(samples_in)) ** 0.5)
                print(f"  recorded rms={rms}, playing back ...")
                speaker.play(audio_amp)
                print(f"  done cycle {i+1}")
                time.sleep(0.3)
            print("  LOOPBACK done.")
        except Exception as e:
            print(f"  ERR: {e}")
        finally:
            try:
                speaker.deinit()
            except:
                pass
            sd_pin.off()
            print(f"  SD (GPIO{PIN_SPK_SD}) = LOW, amp disabled")
            time.sleep(0.3)
            machine.reset()

    speaker = Speaker()
    # led = LEDController()  # 暂时禁用, 等接 WS2812 再开
    button = Button()

    # 3. 按钮 — 主循环直接轮询 button.consume_press(), 不需要 event
    def on_button_press():
        print("[Button] Pressed! (ISR set flag)")
    button.set_callback(on_button_press)

    # 4. 启动LED (暂时禁用)
    # led.show_idle()

    # 5. 连接服务器并运行
    class FakeLED:
        def show_idle(self): pass
        def show_listening(self): pass
        def show_thinking(self): pass
        def show_speaking(self): pass
        def show_error(self): pass
        def off(self): pass
    led = FakeLED()
    robot = RobotClient(mic, speaker, led)
    # while True 循环, 噪音门跳过, 不会退出
    await robot.connect_and_run(button)

# 启动
try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("[Robot] Shutting down...")
except Exception as e:
    print(f"[FATAL] {e}")
    import sys
    sys.print_exception(e)
