import os
import math

# Test the file paths
vid_dir = "/home/rishabhjain/bandon/rishabh/downloading/voxceleb2/dev/mp4"
aud_dir = "/home/rishabhjain/bandon/rishabh/downloading/voxceleb2/aac"
label_dir = "/home/rishabhjain/Desktop/Experiments/lipreading-data-guide/Voxceleb2/preparation"
root_dir = "/media/rishabhjain/SSD/Data/VC2"
dataset = "vox2"
seg_duration = 24

# Load filenames like the original script
filenames = [
    os.path.join(vid_dir, _ + ".mp4")
    for _ in open(os.path.join(label_dir, "vox-en.id")).read().splitlines()
]

print(f"Total files to process: {len(filenames)}")
print(f"First 5 expected video files:")
for i in range(5):
    print(f"  {filenames[i]}")
    print(f"    Exists: {os.path.exists(filenames[i])}")

print(f"\nFirst 5 expected audio files:")
for i in range(5):
    aud_filename = filenames[i].replace(vid_dir, aud_dir)[:-4] + ".wav"
    print(f"  {aud_filename}")
    print(f"    Exists: {os.path.exists(aud_filename)}")

# Check output directory
dst_vid_dir = os.path.join(root_dir, dataset, f"{dataset}_video_seg{seg_duration}s")
print(f"\nOutput directory would be: {dst_vid_dir}")
print(f"Output directory exists: {os.path.exists(dst_vid_dir)}")

# Check if we can create the output directory
try:
    os.makedirs(dst_vid_dir, exist_ok=True)
    print(f"Successfully created output directory: {dst_vid_dir}")
except Exception as e:
    print(f"Error creating output directory: {e}")