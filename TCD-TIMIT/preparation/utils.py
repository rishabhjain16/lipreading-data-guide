import os

import torchaudio
import torchvision


def split_file(filename, max_frames=600, fps=25.0):

    lines = open(filename).read().splitlines()

    flag = 0
    stack = []
    res = []

    tmp = 0
    start_timestamp = 0.0

    threshold = max_frames / fps

    for line in lines:
        if "WORD START END ASDSCORE" in line:
            flag = 1
            continue
        if flag:
            word, start, end, score = line.split(" ")
            start, end, score = float(start), float(end), float(score)
            if end < tmp + threshold:
                stack.append(word)
                last_timestamp = end
            else:
                res.append(
                    [
                        " ".join(stack),
                        start_timestamp,
                        last_timestamp,
                        last_timestamp - start_timestamp,
                    ]
                )
                tmp = start
                start_timestamp = start
                stack = [word]
    if stack:
        res.append([" ".join(stack), start_timestamp, end, end - start_timestamp])
    return res


def save_vid_txt(
    dst_vid_filename, dst_txt_filename, trim_video_data, content, video_fps=25
):
    # -- save video
    save2vid(dst_vid_filename, trim_video_data, video_fps)
    # -- save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    f = open(dst_txt_filename, "w")
    f.write(f"{content}")
    f.close()


def save_vid_aud(
    dst_vid_filename,
    dst_aud_filename,
    trim_vid_data,
    trim_aud_data,
    video_fps=25,
    audio_sample_rate=16000,
):
    # -- save video
    save2vid(dst_vid_filename, trim_vid_data, video_fps)
    # -- save audio
    save2aud(dst_aud_filename, trim_aud_data, audio_sample_rate)


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
    
    # Handle both tensor and numpy array inputs
    if hasattr(vid, 'numpy'):
        vid_np = vid.numpy()
    else:
        vid_np = vid
    
    # Handle different video formats
    if len(vid_np.shape) == 3:  # (T, H, W) - grayscale
        # Convert to RGB by repeating the grayscale channel
        vid_np = vid_np[..., None].repeat(3, axis=-1)  # (T, H, W, 3)
    elif len(vid_np.shape) == 4:  # (T, H, W, C) - already has channels
        # Already in correct format
        pass
    
    # Ensure values are in [0, 255] range and uint8 type
    if vid_np.max() <= 1.0:
        vid_np = (vid_np * 255).astype('uint8')
    else:
        vid_np = vid_np.astype('uint8')
    
    torchvision.io.write_video(filename, vid_np, frames_per_second)


def save2aud(filename, aud, sample_rate):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchaudio.save(filename, aud, sample_rate)