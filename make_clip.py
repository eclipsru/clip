#!/usr/bin/env python3
"""NIGHT DRIVE — 30-second vertical music clip made from the 3 source videos.

Pipeline
    1. make_music.py -> work/music.wav (original 128 BPM soundtrack)
    2. 25 beat-synced shots are rendered with their own grade / camera move /
       glitch / bloom into work/shots/
    3. all shots are joined with xfade transitions, then flashes, light leaks,
       typography and a progress bar are added on the master, and the music is
       muxed in.

Everything is deterministic and reproducible:
    pip install numpy scipy pillow imageio-ffmpeg
    python3 make_music.py && python3 make_clip.py
"""

import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "work")
SHOTS = os.path.join(WORK, "shots")
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "night_drive.mp4")

FPS = 30
BPM = 128.0
BEAT = 60.0 / BPM          # 0.46875 s
BAR = 4 * BEAT             # 1.875 s
DURATION = 30.0            # 16 bars

SOURCES = [
    "19170840349292.mp4",   # 0: wide wet lot, lamp reflections
    "19170851359340.mp4",   # 1: close car / tracking shots
    "19170882882156.mp4",   # 2: car approaching, hero passes
]

FONT_BOLD = os.path.join(ASSETS, "Montserrat-Bold.ttf")
FONT_MED = os.path.join(ASSETS, "Montserrat-Medium.ttf")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("install imageio-ffmpeg or ffmpeg")
        return exe


FF = ffmpeg_exe()


# ---------------------------------------------------------------- shot list
# Each boundary is an exact beat of the 128 BPM grid. "in" is the source
# in-point, "look"/"move" are defined below, "speed" > 1 = faster.
CUTS = [
    0.0, 1.875, 3.75, 5.625, 7.5,
    8.4375, 9.375, 10.3125, 11.25, 12.1875, 13.125, 14.0625, 15.0,
    16.875, 17.8125, 18.75,
    19.6875, 20.625, 21.5625, 22.5, 23.4375, 24.375, 25.3125, 26.25,
    28.125, 30.0,
]

# (source, in-point, look, move, speed, glitch, hflip)
SHOT_DEFS = [
    (0, 0.10, "intro",  "drift",  0.90, False, False),   # 00 opening lamp / wet asphalt
    (2, 1.20, "intro",  "drift",  0.90, False, False),   # 01 distant car, red building
    (1, 0.55, "build",  "punch",  1.00, False, False),   # 02 blue car close-up
    (2, 4.80, "build",  "whip",   1.12, True,  False),   # 03 car approaching head-on
    (2, 9.40, "drop",   "punch",  1.00, True,  False),   # 04 DROP 1 — front pass
    (1, 23.20, "drop",  "punch",  1.00, False, True),    # 05 tracking, flipped
    (1, 24.70, "drop",  "punch",  1.00, False, False),   # 06 front / side
    (2, 10.30, "drop",  "whip",   1.00, True,  False),   # 07 turning car
    (1, 6.10, "drop",   "punch",  1.00, False, True),    # 08 headlight flare
    (2, 14.00, "drop",  "hero",   0.92, False, False),   # 09 slow hero turn
    (1, 13.60, "drop",  "drift",  1.00, False, False),   # 10 wide, reflections
    (1, 31.50, "drop",  "punch",  1.00, False, False),   # 11 side tracking
    (2, 1.50, "break",  "drift",  0.85, False, False),   # 12 BREAKDOWN — lamp on wet ground
    (1, 30.50, "build", "punch",  1.00, False, False),   # 13 build
    (2, 15.60, "build", "whip",   1.10, True,  False),   # 14 riser, close pass
    (2, 16.20, "drop",  "punch",  1.00, True,  False),   # 15 DROP 2 — wheel / driver
    (1, 2.00, "drop",   "punch",  1.00, False, True),    # 16 driver side
    (1, 5.00, "drop",   "punch",  1.00, False, False),   # 17 headlights
    (1, 25.50, "drop",  "hero",   0.92, False, True),    # 18 slow side pass
    (2, 11.00, "drop",  "punch",  1.00, False, False),   # 19 car to camera
    (1, 32.00, "drop",  "punch",  1.00, False, False),   # 20 tracking
    (2, 12.50, "drop",  "hero",   0.86, False, False),   # 21 slow-mo hero
    (1, 15.00, "drop",  "punch",  1.00, False, False),   # 22 last hit
    (0, 31.00, "outro", "drift",  0.95, False, False),   # 23 wide lot, outro
    (2, 1.20, "outro",  "drift",  0.88, False, False),   # 24 reflection + end card
]

assert len(SHOT_DEFS) == len(CUTS) - 1

# transitions between shot i-1 and shot i (index 1..24)
FAST = ["fade", "hblur", "pixelize", "slideleft", "smoothleft", "wipeup",
        "hlslice", "fadeblack", "radial", "smoothright", "zoomin", "fade"]
SPECIAL = {
    1:  ("fade",       0.36),
    2:  ("hblur",      0.24),
    3:  ("wipeleft",   0.18),
    4:  ("fadewhite",  0.14),
    12: ("fade",       0.40),
    13: ("hblur",      0.20),
    16: ("fadewhite",  0.14),
    23: ("fade",       0.38),
    24: ("fade",       0.50),
}
TRANSITIONS = {}
for i in range(1, len(CUTS) - 1):
    if i in SPECIAL:
        TRANSITIONS[i] = SPECIAL[i]
    else:
        TRANSITIONS[i] = (FAST[(i - 1) % len(FAST)], 0.10 if i % 3 else 0.14)


# ---------------------------------------------------------------- looks
# Cinematic night grade: lifted shadows, cool teal shadows / warm highlights,
# soft bloom from headlights, fine grain and a gentle vignette.
LOOKS = {
    "intro": dict(gamma=1.17, contrast=1.06, bright=0.034, sat=1.08,
                  cb="rs=-0.045:gs=0.022:bs=0.038:rh=0.040:gh=0.010:bh=-0.028",
                  vib=0.14, bloom=0.36, curve="0/0.02 0.18/0.23 0.5/0.55 0.85/0.90 1/0.98",
                  vig="PI/6.2", grain=3, sharp=0.30),
    "build": dict(gamma=1.15, contrast=1.11, bright=0.030, sat=1.15,
                  cb="rs=-0.040:gs=0.025:bs=0.035:rh=0.050:gh=0.012:bh=-0.030",
                  vib=0.20, bloom=0.30, curve="0/0.02 0.18/0.22 0.5/0.55 0.85/0.91 1/0.98",
                  vig="PI/5.8", grain=3, sharp=0.38),
    "drop": dict(gamma=1.16, contrast=1.13, bright=0.032, sat=1.18,
                 cb="rs=-0.045:gs=0.025:bs=0.040:rh=0.055:gh=0.015:bh=-0.035",
                 vib=0.24, bloom=0.32, curve="0/0.02 0.18/0.21 0.5/0.55 0.85/0.92 1/1.00",
                 vig="PI/5.6", grain=4, sharp=0.45),
    "break": dict(gamma=1.20, contrast=1.02, bright=0.038, sat=0.96,
                  cb="rs=-0.030:gs=0.018:bs=0.030:rh=0.030:gh=0.010:bh=-0.020",
                  vib=0.08, bloom=0.44, curve="0/0.03 0.18/0.24 0.5/0.56 0.85/0.90 1/0.97",
                  vig="PI/7", grain=3, sharp=0.22),
    "outro": dict(gamma=1.18, contrast=1.04, bright=0.034, sat=1.02,
                  cb="rs=-0.025:gs=0.015:bs=0.030:rh=0.055:gh=0.018:bh=-0.025",
                  vib=0.12, bloom=0.38, curve="0/0.02 0.18/0.23 0.5/0.55 0.85/0.90 1/0.97",
                  vig="PI/6.5", grain=3, sharp=0.28),
}


def move_expr(kind, phase):
    """zoompan expressions. `time` starts at the (beat-aligned) shot in-point."""
    b = BEAT
    p = phase
    if kind == "drift":
        z = f"1.005+0.00042*on+0.010*sin(time*0.9)"
        x = "iw/2-(iw/zoom/2)+16*sin(time*0.65)"
        y = "ih/2-(ih/zoom/2)+20*sin(time*0.50)"
    elif kind == "punch":
        z = f"1.020+0.00040*on+0.045*exp(-mod(time+{p:.4f},{b})*11)"
        x = "iw/2-(iw/zoom/2)+7*sin(time*17)*exp(-mod(time+" + f"{p:.4f},{b})*7)"
        y = "ih/2-(ih/zoom/2)+8*cos(time*15)*exp(-mod(time+" + f"{p:.4f},{b})*7)"
    elif kind == "whip":
        z = f"1.025+0.00055*on+0.055*sin(2*PI*time/(2*{b}))"
        x = "iw/2-(iw/zoom/2)+26*sin(time*1.7)"
        y = "ih/2-(ih/zoom/2)+14*cos(time*1.2)"
    else:  # hero — very slow, almost locked-off
        z = "1.055+0.00022*on+0.012*sin(time*1.1)"
        x = "iw/2-(iw/zoom/2)+10*sin(time*0.5)"
        y = "ih/2-(ih/zoom/2)+12*cos(time*0.4)"
    return z, x, y


def glitch_enable(phase):
    # two or three 2-3 frame RGB-split / noise bursts inside the shot
    a = 0.02 + phase
    return (f"between(t,{a:.3f},{a+0.10:.3f})+"
            f"between(t,{a+0.47:.3f},{a+0.55:.3f})+"
            f"between(t,{a+0.94:.3f},{a+0.99:.3f})")


def shot_filter(idx, D, look, move, speed, glitch, flip, phase):
    L = LOOKS[look]
    z, x, y = move_expr(move, phase)
    hflip = "hflip," if flip else ""
    gl = ""
    if glitch:
        en = glitch_enable(phase)
        gl = (f"format=rgb24,rgbashift=rh=16:bh=-14:rv=2:bv=-2:enable='{en}',"
              f"noise=alls=6:allf=t:enable='{en}',format=yuv420p,")
    chain = (
        f"[0:v]setpts=(PTS-STARTPTS)/{speed},fps={FPS},"
        f"trim=duration={D:.6f},setpts=PTS-STARTPTS,{hflip}"
        f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s=720x1280:fps={FPS},"
        f"scale=1080:1920:flags=lanczos,"
        f"eq=gamma={L['gamma']}:contrast={L['contrast']}:brightness={L['bright']}:"
        f"saturation={L['sat']}:gamma_weight=0.75,"
        f"curves=master='{L['curve']}',"
        f"colorbalance={L['cb']},vibrance=intensity={L['vib']},"
        f"split[main][gl];"
        f"[gl]scale=540:960:flags=lanczos,gblur=sigma=10,"
        f"curves=master='0/0 0.45/0 0.75/0.25 1/1',"
        f"colorbalance=rh=0.08:gh=0.02:bh=-0.08,eq=saturation=1.08,"
        f"scale=1080:1920:flags=lanczos,format=yuv420p[glow];"
        f"[main]format=yuv420p[m2];"
        f"[m2][glow]blend=all_mode=screen:all_opacity={L['bloom']},"
        f"unsharp=5:5:{L['sharp']},"
        f"vignette={L['vig']}:eval=frame,noise=alls={L['grain']}:allf=t+u,"
        f"{gl}format=yuv420p[v]"
    )
    return chain


def durations():
    """Work in integer frames so xfade offsets never drift against the beat grid.

    128 BPM at 30 fps is 14.0625 frames per beat, so cuts are quantised to the
    nearest frame (<= 17 ms error, imperceptible) and every transition length
    and offset is an exact integer number of frames.
    """
    fc = [int(round(t * FPS)) for t in CUTS]
    e_frames = [fc[i + 1] - fc[i] for i in range(len(SHOT_DEFS))]
    TF = {0: 0, len(SHOT_DEFS): 0}
    for i, (_, dur) in TRANSITIONS.items():
        TF[i] = max(2, int(round(dur * FPS)))
    D_frames = []
    for i in range(len(SHOT_DEFS)):
        D_frames.append(e_frames[i] + (TF[i] + TF[i + 1] + 1) // 2)
    D = [d / FPS for d in D_frames]
    T = {i: TF[i] / FPS for i in TF}
    return D, D_frames, TF, T


def render_shots(D, D_frames, TF, T):
    os.makedirs(SHOTS, exist_ok=True)
    for i, (src, tin, look, move, speed, glitch, flip) in enumerate(SHOT_DEFS):
        path = os.path.join(SHOTS, f"shot_{i:02d}.mp4")
        if (os.path.exists(path) and os.path.getsize(path) > 10000
                and not os.environ.get("FORCE_SHOTS")):
            print(f"[shot {i:02d}] cached", flush=True)
            continue
        phase = T[i] / 2
        fg = shot_filter(i, D[i], look, move, speed, glitch, flip, phase)
        src_dur = (D_frames[i] + 5) / FPS * speed + 0.5
        cmd = [
            FF, "-hide_banner", "-loglevel", "error",
            "-ss", f"{tin:.4f}", "-t", f"{src_dur:.4f}",
            "-i", os.path.join(ROOT, SOURCES[src]),
            "-filter_complex", fg, "-map", "[v]", "-an",
            "-frames:v", str(D_frames[i] + 5),
            "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "15", "-pix_fmt", "yuv420p", "-g", "60",
            "-y", path,
        ]
        print(f"[shot {i:02d}/{len(SHOT_DEFS)-1}] {SOURCES[src]} @ {tin:5.2f}s "
              f"-> {D[i]:.3f}s  {look}/{move}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:])
            raise SystemExit(f"shot {i} failed")
    print("all shots rendered", flush=True)


# ---------------------------------------------------------------- overlays
def make_leaks():
    h, w = 1920, 1080
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def leak(path, cx, cy, radius, color, angle=0.0):
        x = (xx - cx) * np.cos(angle) - (yy - cy) * np.sin(angle)
        y = (xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        d = np.sqrt((x / (radius * 0.75)) ** 2 + (y / radius) ** 2)
        core = np.clip(1 - d, 0, 1) ** 2.2
        streak = np.clip(1 - np.abs(y) / (radius * 0.10), 0, 1) * np.exp(-np.abs(x) / (radius * 1.4))
        a = np.clip(core * 0.85 + streak * 0.30, 0, 1) * 0.62
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = (a * 255).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(path)

    leak(os.path.join(WORK, "leak_warm.png"), -120, 260, 1250, (255, 148, 62), 0.35)
    leak(os.path.join(WORK, "leak_cool.png"), 1180, 1450, 1150, (70, 190, 255), -0.3)


# ---------------------------------------------------------------- master
def master_filter(D, D_frames, TF, n_shots):
    lines = []
    # normalise each shot stream
    for i in range(n_shots):
        lines.append(
            f"[{i}:v]settb=AVTB,fps={FPS},format=yuv420p,"
            f"trim=duration={D[i]:.6f},setpts=PTS-STARTPTS[s{i}]"
        )

    cum = D_frames[0]
    acc = "s0"
    for i in range(1, n_shots):
        name, _ = TRANSITIONS[i]
        dur_f = TF[i]
        offset_f = cum - dur_f
        out = "vfinal" if i == n_shots - 1 else f"x{i}"
        lines.append(
            f"[{acc}][s{i}]xfade=transition={name}:"
            f"duration={dur_f/FPS:.6f}:offset={offset_f/FPS:.6f}[{out}]"
        )
        cum = cum - dur_f + D_frames[i]
        acc = out

    # ---- master overlays -------------------------------------------------
    # subtle beat flashes on the two drops + final impact
    flashes = "+".join(
        f"between(t,{t:.4f},{t+0.085:.4f})"
        for t in (7.5, 8.4375, 9.375, 10.3125, 11.25, 12.1875, 13.125,
                  18.75, 19.6875, 20.625, 21.5625, 22.5, 23.4375, 24.375, 26.25)
    )
    big_flashes = "+".join(
        f"between(t,{t:.4f},{t+0.10:.4f})" for t in (7.5, 18.75, 26.25)
    )
    master_fx = [
        # thin progress rail
        f"drawbox=x=0:y=ih-7:w=iw:h=3:color=white@0.13:t=fill",
        f"drawbox=x=0:y=ih-7:w='iw*t/{DURATION}':h=3:color=0x8CE8FF@0.78:t=fill",
        # beat-synced exposure pulses
        f"eq=brightness=0.075:enable='{flashes}'",
        # white impact frames (drawn after eq so they stay clean)
        f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.80:t=fill:enable='{big_flashes}'",
    ]
    base = f"[{acc}]" + ",".join(master_fx)

    # light leaks (two looped RGBA images)
    leak_windows_warm = "+".join(
        f"between(t,{t:.3f},{t+0.55:.3f})" for t in (3.55, 7.35, 14.85, 18.6, 26.1)
    )
    leak_windows_cool = "+".join(
        f"between(t,{t:.3f},{t+0.5:.3f})" for t in (1.7, 5.5, 9.3, 16.7, 22.4, 28.0)
    )
    lines.append(
        f"[W:v]scale=1080:1920,format=rgba,colorchannelmixer=aa=0.85[lw];"
        f"[C:v]scale=1080:1920,format=rgba,colorchannelmixer=aa=0.70[lc]"
    )
    lines.append(base + f"[m0]")
    lines.append(
        f"[m0][lw]overlay=0:0:enable='{leak_windows_warm}':eof_action=pass[m1];"
        f"[m1][lc]overlay=0:0:enable='{leak_windows_cool}':eof_action=pass[m2]"
    )

    # fade the picture to black for the end card
    lines.append(f"[m2]fade=t=out:st=29.05:d=0.95[m3]")

    # ---- typography ------------------------------------------------------
    main_alpha = (
        "if(lt(t,0.30),0,"
        "if(lt(t,0.72),(t-0.30)/0.42,"
        "if(lt(t,1.42),1,if(lt(t,1.82),(1.82-t)/0.40,0))))"
    )
    sub_alpha = (
        "if(lt(t,0.55),0,"
        "if(lt(t,0.95),(t-0.55)/0.40,"
        "if(lt(t,1.42),1,if(lt(t,1.82),(1.82-t)/0.40,0))))"
    )
    end_alpha = (
        "if(lt(t,28.35),0,"
        "if(lt(t,28.80),(t-28.35)/0.45,"
        "if(lt(t,29.35),1,if(lt(t,29.85),(29.85-t)/0.50,0))))"
    )
    title = "N I G H T   D R I V E"
    lines.append(
        f"[m3]drawbox=x='(iw-150)/2':y='(ih/2-150)':w=150:h=2:"
        f"color=0x8CE8FF@0.75:t=fill:enable='between(t,0.3,1.82)',"
        f"drawtext=fontfile='{FONT_BOLD}':text='{title}':fontsize=76:"
        f"fontcolor=white@0.97:borderw=2:bordercolor=black@0.35:"
        f"shadowx=3:shadowy=3:shadowcolor=0x00AACC@0.65:"
        f"x='(w-text_w)/2':y='(h-text_h)/2-40':alpha='{main_alpha}':"
        f"enable='between(t,0.28,1.84)',"
        f"drawtext=fontfile='{FONT_MED}':text='A F T E R   D A R K   /   V O L .  0 1':"
        f"fontsize=26:fontcolor=white@0.96:borderw=1:bordercolor=black@0.55:"
        f"shadowx=2:shadowy=2:shadowcolor=black@0.75:"
        f"x='(w-text_w)/2':y='(h-text_h)/2+62':alpha='{sub_alpha}':"
        f"enable='between(t,0.53,1.84)'[m4]"
    )
    lines.append(
        f"[m4]drawtext=fontfile='{FONT_BOLD}':text='{title}':fontsize=64:"
        f"fontcolor=white@0.97:borderw=2:bordercolor=black@0.30:"
        f"shadowx=3:shadowy=3:shadowcolor=0x00AACC@0.6:"
        f"x='(w-text_w)/2':y='(h-text_h)/2-18':alpha='{end_alpha}':"
        f"enable='between(t,28.33,29.87)',"
        f"drawtext=fontfile='{FONT_MED}':text='3  V I D E O S   /   1  N I G H T   /   1 2 8  B P M':"
        f"fontsize=23:fontcolor=white@0.95:borderw=1:bordercolor=black@0.55:"
        f"shadowx=2:shadowy=2:shadowcolor=black@0.75:"
        f"x='(w-text_w)/2':y='(h-text_h)/2+70':alpha='{end_alpha}':"
        f"enable='between(t,28.33,29.87)',"
        f"format=yuv420p[vout]"
    )

    lines.append(f"[A:a]atrim=duration={DURATION},asetpts=PTS-STARTPTS,"
                 f"alimiter=limit=0.72:level=false:latency=true,"
                 f"afade=t=out:st=29.2:d=0.8,aresample=async=1:first_pts=0[aout]")
    return ";\n".join(lines)


def render_master(D, D_frames, TF):
    n = len(SHOT_DEFS)
    cmd = [FF, "-hide_banner", "-loglevel", "warning", "-stats"]
    for i in range(n):
        cmd += ["-i", os.path.join(SHOTS, f"shot_{i:02d}.mp4")]
    cmd += ["-i", os.path.join(WORK, "music.wav")]                       # A
    cmd += ["-loop", "1", "-framerate", str(FPS), "-t", str(DURATION),
            "-i", os.path.join(WORK, "leak_warm.png")]                   # W
    cmd += ["-loop", "1", "-framerate", str(FPS), "-t", str(DURATION),
            "-i", os.path.join(WORK, "leak_cool.png")]                    # C
    # give the extra inputs explicit labels by position:
    # 0..n-1 = shots, n = A(audio), n+1 = W(warm), n+2 = C(cool)
    fg = master_filter(D, D_frames, TF, n)
    # remap the generic labels to actual input indices
    fg = fg.replace("[A:a]", f"[{n}:a]").replace("[W:v]", f"[{n+1}:v]").replace("[C:v]", f"[{n+2}:v]")
    # avoid collision: shot labels s0.. are fine; 'vfinal' etc fine
    cmd += ["-filter_complex", fg, "-map", "[vout]", "-map", "[aout]",
            "-t", str(DURATION), "-r", str(FPS),
            "-c:v", "libx264", "-preset", "slow", "-crf", "21",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
            "-movflags", "+faststart",
            "-metadata", "title=NIGHT DRIVE — after dark",
            "-metadata", "comment=Original 128 BPM soundtrack, 3 source clips, beat-synced edit",
            "-y", OUT]
    print("rendering master ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-6000:])
        raise SystemExit("master render failed")
    print(r.stderr[-2000:])
    print("wrote", OUT, os.path.getsize(OUT), "bytes", flush=True)


def main():
    os.makedirs(WORK, exist_ok=True)
    make_leaks()
    D, D_frames, TF, T = durations()
    print(f"{len(SHOT_DEFS)} shots, {DURATION}s, {len(TRANSITIONS)} transitions, "
          f"sum D = {sum(D):.3f}s", flush=True)
    if not os.environ.get("MASTER_ONLY"):
        render_shots(D, D_frames, TF, T)
    render_master(D, D_frames, TF)


if __name__ == "__main__":
    main()
