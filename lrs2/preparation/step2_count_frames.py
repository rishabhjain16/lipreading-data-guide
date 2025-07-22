import cv2, math, os
from tqdm import tqdm
from scipy.io import wavfile

def count_frames(fids, base_dir):
    total_num_frames = []
    for fid in tqdm(fids):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        if not os.path.exists(wav_fn):
            print(f"Missing audio file: {wav_fn}")
            continue
        if not os.path.exists(video_fn):
            print(f"Missing video file: {video_fn}")
            continue
        num_frames_audio = len(wavfile.read(wav_fn)[1])
        cap = cv2.VideoCapture(video_fn)
        num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_num_frames.append([num_frames_audio, num_frames_video])
    return total_num_frames

def check(fids, base_dir):
    missing = []
    for fid in tqdm(fids):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        is_file = os.path.isfile(wav_fn) and os.path.isfile(video_fn)
        if not is_file:
            if not os.path.isfile(wav_fn):
                print(f"Missing audio file: {wav_fn}")
            if not os.path.isfile(video_fn):
                print(f"Missing video file: {video_fn}")
            missing.append(fid)
    return missing

def main():
    # Set the root directory for LRS2 dataset
    lrs2_root = '/home/rishabh/Desktop/Datasets/lrs2_rf/lrs2/lrs2_video_seg16s'
    
    # Read the file list
    file_list = os.path.join(lrs2_root, 'file.list')
    fids = [ln.strip() for ln in open(file_list).readlines()]
    print(f"{len(fids)} files")

    # Base directory where both audio and video files are located
    base_dir = lrs2_root

    # Debugging paths
    print(f"Base Directory: {base_dir}")

    # Check for missing files in the base directory
    missing_fids = check(fids, base_dir)

    if len(missing_fids) > 0:
        print(f"Some audio/video files do not exist, see {lrs2_root}/missing.list")
        with open(os.path.join(lrs2_root, 'missing.list'), 'w') as fo:
            fo.write('\n'.join(missing_fids) + '\n')
    else:
        num_frames = count_frames(fids, base_dir)
        audio_num_frames = [x[0] for x in num_frames]
        video_num_frames = [x[1] for x in num_frames]

        with open(os.path.join(lrs2_root, 'nframes.audio'), 'w') as fo:
            fo.write(''.join([f"{x}\n" for x in audio_num_frames]))
        with open(os.path.join(lrs2_root, 'nframes.video'), 'w') as fo:
            fo.write(''.join([f"{x}\n" for x in video_num_frames]))

if __name__ == "__main__":
    main()
