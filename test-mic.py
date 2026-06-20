"""
INMP441 麦克风最小测试
用法: mpremote connect COM11 run test-mic.py
预期: 看到 3 秒录音 + 字节数 + 最大最小值
"""
from machine import Pin, I2S
import time

PIN_MIC_SCK = 4
PIN_MIC_WS  = 5
PIN_MIC_SD  = 6
SAMPLE_RATE = 16000
BITS = 16

print("[1/4] Init I2S RX on GPIO 4/5/6 ...")
i2s = I2S(
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
print("    OK")

print("[2/4] Recording 3 seconds ...")
duration_ms = 3000
num_samples = int(SAMPLE_RATE * duration_ms / 1000)
num_bytes = num_samples * 2
buf = bytearray(num_bytes)
t0 = time.ticks_ms()
i2s.readinto(buf)
t1 = time.ticks_ms()
print(f"    {len(buf)} bytes in {time.ticks_diff(t1, t0)} ms")

print("[3/4] Analyzing audio ...")
import struct
samples = struct.unpack(f'<{num_samples}h', buf)
max_v = max(samples)
min_v = min(samples)
nonzero = sum(1 for s in samples if abs(s) > 100)
rms = (sum(s*s for s in samples) / num_samples) ** 0.5

print(f"    max={max_v} min={min_v} rms={int(rms)} nonzero={nonzero}")

print("[4/4] Verdict")
if nonzero < 10:
    print("    ❌ 静音 — 麦克风没接好 / L/R 错 / 没供电")
elif rms < 100:
    print("    ⚠️  弱信号 — 试着对麦克风吹气")
else:
    print("    ✅ 麦克风在工作! (rms 越大越响)")

i2s.deinit()
print("DONE")
