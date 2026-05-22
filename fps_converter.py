import os
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm

def convert_dataset(src_root, dst_root, target_fps=25):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    # 1. Collect all files to process
    all_files = list(src_root.rglob("*"))
    
    print(f">>> Found {len(all_files)} total items. Starting conversion to {target_fps} FPS...")
    
    for item in tqdm(all_files):
        # Create corresponding destination path
        rel_path = item.relative_to(src_root)
        dst_item = dst_root / rel_path
        
        # If it's a directory, just create it
        if item.is_dir():
            dst_item.mkdir(parents=True, exist_ok=True)
            continue
            
        # 2. Handle Video Files (.mp4)
        if item.suffix.lower() == ".mp4":
            dst_item.parent.mkdir(parents=True, exist_ok=True)
            
            # ffmpeg command:
            # -i: input
            # -r: force output frame rate to 25
            # -c:v libx264: encode with H.264
            # -crf 23: standard quality (lower is better)
            # -c:a copy: DO NOT touch the audio, just copy the stream to keep sync
            # -y: overwrite if exists
            cmd = [
                "ffmpeg", "-y", "-i", str(item),
                "-r", str(target_fps),
                "-c:v", "libx264",
                "-crf", "23",
                "-c:a", "copy",
                "-preset", "veryfast",
                str(dst_item)
            ]
            
            # Execute ffmpeg
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        # 3. Handle Audio and Text Files (Copy as-is)
        elif item.suffix.lower() in [".wav", ".txt", ".csv"]:
            dst_item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_item)

if __name__ == "__main__":
    # Update these paths to match your system
    SOURCE = "/data/ssd3/data_rishabh/candor_lips/candor_video_30"
    DESTINATION = "/data/ssd3/data_rishabh/candor_lips/candor_video/"

    convert_dataset(SOURCE, DESTINATION)
    print(f"\n>>> Done! 25 FPS dataset created at: {DESTINATION}")