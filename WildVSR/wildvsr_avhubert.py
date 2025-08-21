#!/usr/bin/env python3
import json, os, subprocess, cv2

ROOT = os.path.abspath("/home/rishabh/Desktop/Datasets/WildVSR")
VIDEO_DIR = os.path.join(ROOT, "videos")
OUT_DIR = os.path.abspath("/home/rishabh/Desktop/Datasets/WildVSR/wildvsr_manifest")
os.makedirs(OUT_DIR, exist_ok=True)

# make a tiny silent wav once (16kHz, 0.01s) if you have no audio
DUMMY_WAV = os.path.join(OUT_DIR, "silence.wav")
if not os.path.exists(DUMMY_WAV):
    import wave, struct
    with wave.open(DUMMY_WAV, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h"*160, *([0]*160)))  # 0.01s

def count_frames_ffprobe(path):
    try:
        cmd = [
            "ffprobe","-v","error","-select_streams","v:0",
            "-count_frames","-show_entries","stream=nb_read_frames,r_frame_rate",
            "-of","default=nokey=1:noprint_wrappers=1", path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip().splitlines()
        nb, fps = out[0], out[1]
        if nb == "N/A":
            raise RuntimeError("ffprobe nb_read_frames N/A")
        return int(nb)
    except Exception:
        # fallback: OpenCV
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

with open(os.path.join(ROOT, "labels.json")) as f:
    labels = json.load(f)

ids = sorted(labels.keys())
wrd_path = os.path.join(OUT_DIR, "test.wrd")
tsv_path = os.path.join(OUT_DIR, "test.tsv")

with open(wrd_path, "w", encoding="utf-8") as wrd, open(tsv_path, "w", encoding="utf-8") as tsv:
    tsv.write(f"{VIDEO_DIR}\n")  # header = root
    for uid in ids:
        filename = uid if uid.endswith(".mp4") else f"{uid}.mp4"
        vpath = os.path.join(VIDEO_DIR, filename)
        assert os.path.exists(vpath), vpath
        nframes_v = count_frames_ffprobe(vpath)
        wrd.write(labels[uid].strip() + "\n")
        # columns: id, abs_video, abs_audio(or dummy), nframes_video, nframes_audio
        tsv.write(f"{uid}\t{vpath}\t{DUMMY_WAV}\t{nframes_v}\t0\n")

print("Wrote:", tsv_path, wrd_path)
