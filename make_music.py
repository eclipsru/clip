#!/usr/bin/env python3
"""Original 128-BPM electronic soundtrack for the NIGHT DRIVE clip.

Everything is synthesized from scratch with numpy/scipy, so the result is
royalty-free and the beat grid (16 bars x 1.875 s = exactly 30 s) can be used
to place every cut, flash and zoom punch in the video.

    BPM 128 | beat 0.46875 s | bar 1.875 s | 16 bars = 30.000 s
    0.000  intro        (bars 0-1)
    3.750  build        (bars 2-3)
    7.500  DROP 1       (bars 4-7)
   15.000  breakdown    (bars 8-9)
   18.750  DROP 2       (bars 10-13)
   26.250  outro        (bars 14-15)
"""

import os
import wave

import numpy as np
from scipy.signal import butter, sawtooth, sosfilt, fftconvolve

SR = 48000
BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
DURATION = 16 * BAR
N = int(round(SR * DURATION))
OUT = os.path.join(os.path.dirname(__file__), "work", "music.wav")

rng = np.random.default_rng(20261103)
LEFT = np.zeros(N, dtype=np.float32)
RIGHT = np.zeros(N, dtype=np.float32)
REVERB_L = np.zeros(N, dtype=np.float32)
REVERB_R = np.zeros(N, dtype=np.float32)
KICK_TIMES = []


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def times(dur):
    return np.arange(int(dur * SR), dtype=np.float32) / SR


def lowpass(x, hz, order=2):
    return sosfilt(butter(order, min(hz, SR * 0.49), btype="low", fs=SR, output="sos"), x).astype(np.float32)


def highpass(x, hz, order=2):
    return sosfilt(butter(order, max(hz, 20), btype="high", fs=SR, output="sos"), x).astype(np.float32)


def bandpass(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.49)], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def white(dur):
    return rng.normal(0, 1, int(dur * SR)).astype(np.float32)


def supersaw(freq, dur, voices=5, spread=0.09, cutoff=3500):
    """Detuned analogue-style saw stack, softened by a low-pass."""
    t = times(dur)
    out = np.zeros_like(t)
    for v in range(voices):
        det = (v - (voices - 1) / 2) / max(voices - 1, 1) * spread
        f = freq * (1 + det / 100.0)
        out += sawtooth(2 * np.pi * f * t + rng.uniform(0, 6.28)) / voices
    return lowpass(out, cutoff) * 0.8


def place(dest_l, dest_r, start, sig, gain=1.0, pan=0.0):
    i = int(round(start * SR))
    if i >= N or i < 0:
        return
    j = min(N, i + len(sig))
    s = sig[: j - i].astype(np.float32)
    pl = np.cos((pan + 1) * np.pi / 4)
    pr = np.sin((pan + 1) * np.pi / 4)
    dest_l[i:j] += s * gain * pl
    dest_r[i:j] += s * gain * pr


# ---------------------------------------------------------------- instruments
def kick(dur=0.55):
    t = times(dur)
    f = 48 + 125 * np.exp(-t * 30)
    phase = 2 * np.pi * (48 * t + 125 * (1 - np.exp(-30 * t)) / 30)
    body = np.sin(phase) * np.exp(-t * 7.5)
    click = white(dur) * np.exp(-t * 260) * 0.35
    sub = np.sin(2 * np.pi * 47 * t) * np.exp(-t * 9) * 0.35
    return np.tanh(2.0 * (body + click + sub)) * 0.92


def clap(dur=0.42):
    t = times(dur)
    n = bandpass(white(dur), 900, 6500)
    bursts = np.zeros_like(t)
    for d, a in [(0.0, 1.0), (0.011, 0.75), (0.023, 0.62), (0.038, 0.4)]:
        tt = np.clip(t - d, 0, None)
        bursts += a * np.exp(-tt * 110) * (t >= d)
    tail = np.exp(-t * 17) * 0.32
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 26) * 0.3
    return (n * (bursts + tail) + tone) * 0.6


def hat(dur=0.16, open_=False):
    t = times(dur)
    n = highpass(white(dur), 6800)
    env = np.exp(-t * (11 if open_ else 62))
    return n * env * (0.42 if open_ else 0.34)


def bass_note(freq, dur, bright=1.0):
    t = times(dur)
    glide = 1 + 0.015 * np.exp(-t * 35)
    saw = sawtooth(2 * np.pi * freq * glide * t)
    tone = lowpass(saw, 260 + 420 * bright)
    sub = np.sin(2 * np.pi * freq * t) * 0.85
    env = (1 - np.exp(-t * 180)) * np.exp(-t * (2.2 / max(dur, 0.1)))
    return np.tanh(1.5 * (0.55 * tone + 0.75 * sub) * env) * 0.62


def pluck(freq, dur, cutoff=2600):
    t = times(dur)
    s = supersaw(freq, dur, voices=3, spread=0.14, cutoff=cutoff)
    sq = np.sign(np.sin(2 * np.pi * freq * t)) * 0.18
    env = (1 - np.exp(-t * 500)) * np.exp(-t * (7.5 / max(dur, 0.06)))
    return (s + sq) * env * 0.55


def pad_chord(notes, dur, cutoff=1250):
    t = times(dur)
    out = np.zeros_like(t)
    for m in notes:
        out += supersaw(mtof(m), dur, voices=3, spread=0.22, cutoff=cutoff) / len(notes)
    attack = 1 - np.exp(-t * 5)
    release = np.clip((dur - t) / 0.55, 0, 1)
    return out * attack * release * 0.38


def stab(notes, dur, cutoff=4200):
    t = times(dur)
    out = np.zeros_like(t)
    for m in notes:
        out += supersaw(mtof(m), dur, voices=5, spread=0.18, cutoff=cutoff) / len(notes)
    env = (1 - np.exp(-t * 350)) * np.exp(-t * 8)
    return out * env * 0.4


def impact(dur=1.6):
    t = times(dur)
    f = 62 * np.exp(-t * 2.2) + 34
    boom = np.sin(2 * np.pi * f * t) * np.exp(-t * 2.6)
    noise = lowpass(white(dur), 2500) * np.exp(-t * 22) * 0.5
    return np.tanh(2.2 * (boom + noise)) * 0.85


def riser(dur, reverse=False):
    t = times(dur)
    n = white(dur)
    # rising band-pass sweep, implemented as a crossfade between 4 static bands
    out = np.zeros_like(t)
    bands = [(200, 700), (600, 1800), (1500, 4500), (3500, 12000)]
    seg = dur / len(bands)
    for k, (lo, hi) in enumerate(bands):
        b = bandpass(n, lo, hi)
        w = np.clip(1 - np.abs(t - (k + 0.5) * seg) / (seg * 0.95), 0, 1)
        out += b * w
    ramp = (t / dur) ** 1.7 if reverse else (t / dur) ** 1.4
    sweep = np.sin(2 * np.pi * (180 * t + 900 * t * t / dur)) * 0.25 * ramp
    out = out * (0.28 + 0.55 * ramp) + sweep
    return out * 0.55


def whoosh(dur=0.55):
    t = times(dur)
    n = white(dur)
    out = np.zeros_like(t)
    bands = [(250, 900), (700, 2200), (1800, 5200), (3000, 9000)]
    seg = dur / len(bands)
    for k, (lo, hi) in enumerate(bands):
        w = np.clip(1 - np.abs(t - (k + 0.5) * seg) / (seg * 0.9), 0, 1)
        out += bandpass(n, lo, hi) * w
    return out * np.sin(np.pi * t / dur) ** 1.5 * 0.5


# ---------------------------------------------------------------- arrangement
# A minor: Am - F - C - G, one chord per bar
CHORDS = [
    ([57, 60, 64], 45),  # Am / A2
    ([53, 57, 60], 41),  # F  / F2
    ([60, 64, 67], 48),  # C  / C3
    ([55, 59, 62], 43),  # G  / G2
]
# A-minor pentatonic lead motif (eighth-note grid)
LEAD = [69, 72, 74, 72, 76, 74, 72, 69, 67, 69, 72, 74, 76, 79, 76, 74]

KICK = kick()
CLAP = clap()
HAT = hat()
OPEN_HAT = hat(0.32, open_=True)


def is_drop(bar):
    return 4 <= bar < 8 or 10 <= bar < 14


for b in range(16):
    chord, bass_root = CHORDS[b % 4]
    bar_t = b * BAR

    # ----- pad (whole bar, with reverb send)
    pad = pad_chord(chord, BAR * 1.04, cutoff=950 if b < 4 else 1450)
    pad_gain = 0.44 if b < 2 else (0.3 if is_drop(b) else 0.62)
    place(LEFT, RIGHT, bar_t, pad, pad_gain, pan=0.0)
    place(REVERB_L, REVERB_R, bar_t, pad, 0.5, pan=0.15)

    # ----- kick
    for beat in range(4):
        t = bar_t + beat * BEAT
        if b < 2:
            if beat not in (0, 2):
                continue
            g = 0.62
        elif b < 4:
            g = 0.74 + 0.05 * b
        elif is_drop(b):
            g = 1.0
        elif b < 10:
            if beat != 0:
                continue
            g = 0.55
        else:
            if beat != 0:
                continue
            g = 0.7
        place(LEFT, RIGHT, t, KICK, g)
        KICK_TIMES.append(t)

    if is_drop(b) and b in (7, 11, 13):
        extra = bar_t + 3.5 * BEAT
        place(LEFT, RIGHT, extra, KICK, 0.62)
        KICK_TIMES.append(extra)

    # ----- clap / snare
    if b >= 2 and not (8 <= b < 10):
        for beat in (1, 3):
            g = 0.85 if is_drop(b) else 0.6
            place(LEFT, RIGHT, bar_t + beat * BEAT, CLAP, g, pan=0.03)
            place(REVERB_L, REVERB_R, bar_t + beat * BEAT, CLAP, 0.32)

    # snare roll into both drops
    if b in (3, 9):
        for step in range(16):
            t = bar_t + 2 * BEAT + step * (BEAT / 4) * (1 - step / 40)
            if t >= (b + 1) * BAR:
                break
            place(LEFT, RIGHT, t, CLAP, 0.12 + 0.4 * (step / 16) ** 2, pan=-0.1 + 0.2 * (step % 3) / 2)

    # ----- hats
    hat_steps = 4 if 8 <= b < 10 else (8 if b < 4 or b >= 14 else 16)
    for step in range(hat_steps):
        t = bar_t + step * BAR / hat_steps
        if b == 0 and step % 2:
            continue
        vel = 0.28 + 0.22 * (step % 4 == 0) + 0.12 * rng.random()
        if is_drop(b):
            vel *= 1.25
        if 8 <= b < 10:
            vel *= 0.35
        pan = 0.18 if step % 2 else -0.18
        place(LEFT, RIGHT, t, OPEN_HAT if (step % 8 == 6 and is_drop(b)) else HAT, vel, pan)

    # ----- bass
    if b >= 2 and not (8 <= b < 10):
        pattern = [(0, 0.55, 0), (3, 0.22, 0), (6, 0.25, 7),
                   (8, 0.42, 0), (11, 0.22, 12), (14, 0.26, 7)]
        for step, length, interval in pattern:
            t = bar_t + step * BEAT / 4
            dur = max(length, 0.14) * BEAT
            f = mtof(bass_root + interval)
            g = 0.85 if is_drop(b) else 0.55
            place(LEFT, RIGHT, t, bass_note(f, dur, bright=1.0 if is_drop(b) else 0.55), g)

    # ----- arpeggio (16ths), quiet in intro / breakdown
    arp_pat = [0, 1, 2, 1, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 2]
    for step in range(16):
        if b == 0 and step % 4:
            continue
        t = bar_t + step * BEAT / 4
        m = chord[arp_pat[step]] + (12 if step in (6, 13) else 0)
        g = 0.16 if b < 2 else (0.34 if is_drop(b) else (0.45 if 8 <= b < 10 else 0.24))
        p = pluck(mtof(m), 0.20, cutoff=1600 if b < 4 else 3200)
        place(LEFT, RIGHT, t, p, g, pan=0.35 if step % 2 else -0.35)
        place(REVERB_L, REVERB_R, t, p, 0.22, pan=-0.3 if step % 2 else 0.3)

    # ----- lead stabs on the drops
    if is_drop(b):
        for step in (0, 6, 10, 14):
            t = bar_t + step * BEAT / 4
            place(LEFT, RIGHT, t, stab(chord, 0.26), 0.42 if step == 0 else 0.3)
        if b in (5, 7, 11, 13):
            for k, m in enumerate(LEAD):
                t = bar_t + k * BEAT / 2
                place(LEFT, RIGHT, t, pluck(mtof(m), 0.22, 4200), 0.25, pan=0.12)

# ----- risers, reverse cymbals, impacts, whooshes
place(LEFT, RIGHT, 5.625, riser(1.875), 0.62, pan=0.0)
place(REVERB_L, REVERB_R, 5.625, riser(1.875), 0.28)
place(LEFT, RIGHT, 6.5625, whoosh(0.9), 0.5)
place(LEFT, RIGHT, 7.5, impact(), 1.0)
KICK_TIMES.append(7.5)

place(LEFT, RIGHT, 17.8125, riser(0.9375, reverse=True), 0.7)
place(LEFT, RIGHT, 18.28125, whoosh(0.45), 0.55)
place(LEFT, RIGHT, 18.75, impact(), 1.05)
KICK_TIMES.append(18.75)

place(LEFT, RIGHT, 26.25, impact(2.2), 0.9)
KICK_TIMES.append(26.25)
place(LEFT, RIGHT, 26.25, whoosh(1.1), 0.35)
place(LEFT, RIGHT, 15.0, whoosh(0.8), 0.3)

# ----- simple algorithmic reverb (short, bright, stereo)
ir_len = int(1.7 * SR)
ir_t = np.arange(ir_len) / SR
ir = rng.normal(0, 1, ir_len).astype(np.float32) * np.exp(-ir_t * 5.2)
for d, a in [(0.019, 0.7), (0.037, 0.5), (0.071, 0.35), (0.113, 0.22)]:
    ir[int(d * SR)] += a
ir *= 0.55
rev_l = fftconvolve(REVERB_L, ir)[:N].astype(np.float32)
rev_r = fftconvolve(REVERB_R, ir)[:N].astype(np.float32)
rev_l = highpass(lowpass(rev_l, 6500), 180)
rev_r = highpass(lowpass(rev_r, 6500), 180)

L = LEFT + rev_l * 0.42
R = RIGHT + rev_r * 0.42

# ----- arp echo (dotted-eighth feedback delay) applied to the whole mix, subtle
delay = int(round(BEAT * 0.75 * SR))
echo_l = np.zeros(N, dtype=np.float32)
echo_r = np.zeros(N, dtype=np.float32)
for tap, g in [(1, 0.18), (2, 0.09), (3, 0.045)]:
    d = tap * delay
    echo_l[d:] += L[:-d] * g
    echo_r[d:] += R[:-d] * g
L += echo_r * 0.5   # ping-pong
R += echo_l * 0.5

# ----- sidechain pumping: duck everything except drums on each kick
duck = np.ones(N, dtype=np.float32)
env = np.exp(-np.arange(int(0.42 * SR)) / SR / 0.11)
for t in KICK_TIMES:
    i = int(round(t * SR))
    j = min(N, i + len(env))
    seg = 1 - 0.52 * env[: j - i]
    duck[i:j] = np.minimum(duck[i:j], seg)
# drums are already in the mix: apply a gentler, global pump instead of full duck
pump = 1 - (1 - duck) * 0.42
L *= pump
R *= pump

# ----- arrangement dynamics: quiet intro/breakdown, big drops
arr = np.ones(N, dtype=np.float32)
BAR_GAINS = [0.80, 0.86, 0.92, 0.98,
             1.06, 1.06, 1.06, 1.06,
             0.62, 0.72,
             1.10, 1.10, 1.10, 1.10,
             0.92, 0.76]
for b, g in enumerate(BAR_GAINS):
    i, j = int(b * BAR * SR), min(N, int((b + 1) * BAR * SR))
    arr[i:j] = g
k = int(0.12 * SR)
arr = np.convolve(arr, np.ones(k, dtype=np.float32) / k, mode="same").astype(np.float32)
L *= arr
R *= arr

# ----- master: soft clip, limiter, fades, slight stereo widening
mid = (L + R) * 0.5
side = (L - R) * 0.5 * 1.14
L, R = mid + side, mid - side
L = np.tanh(L * 1.12).astype(np.float32)
R = np.tanh(R * 1.12).astype(np.float32)
peak = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))), 1e-6)
gain = 0.94 / peak
L *= gain
R *= gain
fade_in = int(0.06 * SR)
fade_out = int(1.1 * SR)
L[:fade_in] *= np.linspace(0, 1, fade_in)
R[:fade_in] *= np.linspace(0, 1, fade_in)
L[-fade_out:] *= np.linspace(1, 0, fade_out) ** 1.4
R[-fade_out:] *= np.linspace(1, 0, fade_out) ** 1.4

# ----- optional low engine/wind bed extracted from the source footage
amb_path = os.path.join(os.path.dirname(__file__), "work", "ambience.wav")
if os.path.exists(amb_path):
    with wave.open(amb_path, "rb") as w:
        nframes = w.getnframes()
        raw = np.frombuffer(w.readframes(nframes), dtype="<i2")
        if w.getnchannels() == 2:
            raw = raw.reshape(-1, 2).mean(axis=1)
        amb = raw.astype(np.float32) / 32768.0
    if len(amb) < N:
        amb = np.tile(amb, int(np.ceil(N / len(amb))))
    amb = amb[:N]
    amb = lowpass(highpass(amb, 70), 1100) * 0.055
    bed = np.ones(N, dtype=np.float32)
    bed[0 : int(3.7 * SR)] = 0.75
    bed[int(3.7 * SR) : int(7.5 * SR)] = np.linspace(0.75, 0.25, int(3.8 * SR))
    bed[int(7.5 * SR) : int(15 * SR)] = 0.22
    bed[int(15 * SR) : int(18.75 * SR)] = np.linspace(0.45, 0.25, int(3.75 * SR))
    bed[int(18.75 * SR) : int(26 * SR)] = 0.2
    bed[int(26 * SR) :] = np.linspace(0.5, 0.0, N - int(26 * SR))
    L += amb * bed
    R += amb * bed * 0.95

# Leave headroom for the AAC encoder: sample peaks <= -1.9 dBFS so that
# inter-sample true peaks stay safely below 0 dBTP after lossy encoding.
peak2 = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))), 1e-6)
if peak2 > 0.68:
    L *= 0.68 / peak2
    R *= 0.68 / peak2

# ----- write 16-bit stereo wav
os.makedirs(os.path.dirname(OUT), exist_ok=True)
stereo = np.empty(N * 2, dtype="<i2")
stereo[0::2] = np.clip(L, -1, 1) * 32767
stereo[1::2] = np.clip(R, -1, 1) * 32767
with wave.open(OUT, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(stereo.tobytes())

print(f"wrote {OUT}: {DURATION:.3f}s, {len(KICK_TIMES)} kicks, peak {peak * gain:.3f}")
