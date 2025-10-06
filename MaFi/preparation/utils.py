import os
import torch
import torchvision

def save_vid_txt(
    dst_vid_filename,
    dst_txt_filename,
    trim_vid_data,
    content,
    video_fps=25,
):
    """
    Save video and text files for MaFi dataset (no audio).
    
    Args:
        dst_vid_filename: Output video file path
        dst_txt_filename: Output text file path
        trim_vid_data: Video tensor data
        content: Text content (word)
        video_fps: Video frame rate (default: 25)
    """
    # Save video
    if dst_vid_filename is not None:
        save2vid(dst_vid_filename, trim_vid_data, video_fps)
    
    # Save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    with open(dst_txt_filename, "w") as f:
        f.write(f"{content}")

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
    """
    Save video, audio, and text files (for compatibility with other datasets).
    """
    import torchaudio
    
    # Save video
    if dst_vid_filename is not None:
        save2vid(dst_vid_filename, trim_vid_data, video_fps)
    
    # Save audio
    if dst_aud_filename is not None:
        save2aud(dst_aud_filename, trim_aud_data, audio_sample_rate)
    
    # Save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    with open(dst_txt_filename, "w") as f:
        f.write(f"{content}")

def save2vid(filename, vid, frames_per_second):
    """Save video tensor to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)

def save2aud(filename, aud, sample_rate):
    """Save audio tensor to file."""
    import torchaudio
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchaudio.save(filename, aud, sample_rate)
