import os
import torch
import torchaudio
import torchvision

def save_vid_aud_txt(
    dst_vid_filename,
    dst_aud_filename,
    dst_txt_filename,
    trim_vid_data,
    trim_aud_data,
    content,
    video_fps=25,
    audio_sample_rate=16000,
):
    # -- save video
    if dst_vid_filename is not None:
        save2vid(dst_vid_filename, trim_vid_data, video_fps)
    # -- save audio
    if dst_aud_filename is not None:
        save2aud(dst_aud_filename, trim_aud_data, audio_sample_rate)
    # -- save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    f = open(dst_txt_filename, "w")
    f.write(f"{content}")
    f.close()

def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)

def save2aud(filename, aud, sample_rate):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchaudio.save(filename, aud, sample_rate)