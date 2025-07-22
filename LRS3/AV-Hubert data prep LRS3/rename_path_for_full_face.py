import os
import glob


#Script tp rename the paths in .tsv files to point to full face videos instead of cropped ones.
def process_tsv_files(folder_path):
    """
    Process all .tsv files in the given folder, removing 'video/' from paths.
    """
    # Find all .tsv files in the folder
    tsv_files = glob.glob(os.path.join(folder_path, "*.tsv"))
    
    if not tsv_files:
        print(f"No .tsv files found in {folder_path}")
        return
    
    for tsv_file in tsv_files:
        print(f"Processing: {tsv_file}")
        
        # Read the entire file
        with open(tsv_file, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        modified_lines = []
        for line in lines:
            # Replace '/data/ssd2/data_rishabh/lrs3/video/' with '/data/ssd2/data_rishabh/lrs3/'
            modified_line = line.replace('/data/ssd2/data_rishabh/lrs3/video/', '/data/ssd2/data_rishabh/lrs3/')
            modified_lines.append(modified_line)
        
        # Write back to the file
        with open(tsv_file, 'w') as f:
            f.writelines(modified_lines)
        
        print(f"Updated: {tsv_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python fix_tsv_paths.py /path/to/tsv/folder")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    process_tsv_files(folder_path)
