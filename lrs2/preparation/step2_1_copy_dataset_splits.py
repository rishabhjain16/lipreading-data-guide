import os
import shutil

def copy_specific_txt_files(source_dir, target_dir):
    """
    Copy specific text files (test.txt, val.txt, train.txt, pretrain.txt) 
    from source_dir to target_dir.
    
    Args:
        source_dir (str): Source directory containing the text files
        target_dir (str): Target directory where files will be copied
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # List of specific files to copy
    files_to_copy = ["test.txt", "val.txt", "train.txt", "pretrain.txt"]
    
    # Copy each file if it exists
    for filename in files_to_copy:
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"Copied: {source_file} -> {target_file}")
        else:
            print(f"Warning: {source_file} does not exist, skipping.")

if __name__ == "__main__":
    # Source and target directories
    source = "/data/ssd2/data_rishabh/lrs2"
    target = "/data/ssd2/data_rishabh/lrs2/segmented/lrs2/lrs2_video_seg24s"
    
    # Copy the files
    copy_specific_txt_files(source, target)
    print("Text files copied successfully!")
