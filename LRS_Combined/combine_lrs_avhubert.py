#!/usr/bin/env python3
import os
import argparse
import shutil

def merge_dictionaries(lrs2_dict_path, lrs3_dict_path, output_dict_path):
    """
    Merge two dictionary files while preserving the format and removing duplicates.
    """
    print("Merging dictionaries...")
    
    # Read dictionaries
    lrs2_dict = {}
    lrs3_dict = {}
    
    with open(lrs3_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                index_or_freq = parts[1]
                lrs3_dict[word] = index_or_freq
    
    with open(lrs2_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                index_or_freq = parts[1]
                lrs2_dict[word] = index_or_freq
    
    # Determine if dictionaries use indices or frequencies
    uses_indices = all(value.isdigit() for value in list(lrs3_dict.values()))
    
    # Merge dictionaries
    merged_dict = {**lrs2_dict, **lrs3_dict}  # LRS3 takes precedence for duplicates
    
    # If dictionaries use indices, reassign indices to be continuous
    if uses_indices:
        print("Dictionaries use indices, reassigning to be continuous...")
        merged_items = sorted(merged_dict.items(), key=lambda x: int(x[1]))
        merged_dict = {word: str(i) for i, (word, _) in enumerate(merged_items)}
    
    # Write merged dictionary
    with open(output_dict_path, 'w') as f:
        for word, value in sorted(merged_dict.items(), key=lambda x: int(x[1]) if uses_indices else x[0]):
            f.write(f"{word} {value}\n")
    
    print(f"Merged dictionary created with {len(merged_dict)} entries")
    return merged_dict

def combine_datasets(lrs2_path, lrs3_path, output_path):
    """
    Combine LRS2 and LRS3 datasets into a single dataset for training.
    Assumes both datasets already have .tsv, .wrd, and .cluster_counts files.
    """
    # Validate paths
    if not os.path.exists(lrs2_path):
        raise ValueError(f"Directory {lrs2_path} does not exist")
    if not os.path.exists(lrs3_path):
        raise ValueError(f"Directory {lrs3_path} does not exist")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Check for necessary files in both datasets
    required_files = [
        "train.tsv", "train.wrd", "train.cluster_counts",
        "test.tsv", "test.wrd", "test.cluster_counts",
        "valid.tsv", "valid.wrd", "valid.cluster_counts",
        "dict.wrd.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(lrs3_path, file)):
            raise FileNotFoundError(f"Required file {file} not found in {lrs3_path}")
        if not os.path.exists(os.path.join(lrs2_path, file)):
            raise FileNotFoundError(f"Required file {file} not found in {lrs2_path}")
    
    print("Combining datasets...")
    
    # Merge files for each split
    for split in ["train", "test", "valid"]:
        # Merge .tsv files - concatenate but keep only one header
        with open(os.path.join(output_path, f"{split}.tsv"), 'w') as outfile:
            # First line from LRS3 (the root directory)
            with open(os.path.join(lrs3_path, f"{split}.tsv"), 'r') as infile:
                outfile.write(infile.readline())  # Write the header line
                # Write the rest of LRS3 tsv
                outfile.write(infile.read())
            
            # Append LRS2 data (skipping the first line which is the root directory)
            with open(os.path.join(lrs2_path, f"{split}.tsv"), 'r') as infile:
                infile.readline()  # Skip the header line
                outfile.write(infile.read())
        
        # Merge .wrd files - simple concatenation
        with open(os.path.join(output_path, f"{split}.wrd"), 'w') as outfile:
            with open(os.path.join(lrs3_path, f"{split}.wrd"), 'r') as infile:
                outfile.write(infile.read())
            with open(os.path.join(lrs2_path, f"{split}.wrd"), 'r') as infile:
                outfile.write(infile.read())
        
        # Merge .cluster_counts files - simple concatenation
        with open(os.path.join(output_path, f"{split}.cluster_counts"), 'w') as outfile:
            with open(os.path.join(lrs3_path, f"{split}.cluster_counts"), 'r') as infile:
                outfile.write(infile.read())
            with open(os.path.join(lrs2_path, f"{split}.cluster_counts"), 'r') as infile:
                outfile.write(infile.read())
    
    # Merge dictionaries
    lrs2_dict_path = os.path.join(lrs2_path, "dict.wrd.txt")
    lrs3_dict_path = os.path.join(lrs3_path, "dict.wrd.txt")
    output_dict_path = os.path.join(output_path, "dict.wrd.txt")
    merged_dict = merge_dictionaries(lrs2_dict_path, lrs3_dict_path, output_dict_path)
    
    # Verify line counts match between corresponding files
    for split in ["train", "test", "valid"]:
        tsv_count = len(open(os.path.join(output_path, f"{split}.tsv")).readlines()) - 1  # Subtract header line
        wrd_count = len(open(os.path.join(output_path, f"{split}.wrd")).readlines())
        cluster_count = len(open(os.path.join(output_path, f"{split}.cluster_counts")).readlines())
        
        if not (tsv_count == wrd_count == cluster_count):
            print(f"Warning: Line count mismatch in {split} files:")
            print(f"  tsv: {tsv_count}, wrd: {wrd_count}, cluster_counts: {cluster_count}")
            print("  This may cause issues during training.")
        else:
            print(f"{split} set: {tsv_count} examples merged successfully")
    
    print(f"Combined dataset successfully created at {output_path}")
    print("Update your train.sh script to point to this new dataset:")
    print(f"DATA_PATH={output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine LRS2 and LRS3 datasets for training")
    parser.add_argument("--lrs2", required=True, help="Path to LRS2 preprocessed dataset with .tsv, .wrd, and .cluster_counts files")
    parser.add_argument("--lrs3", required=True, help="Path to LRS3 preprocessed dataset with .tsv, .wrd, and .cluster_counts files")
    parser.add_argument("--output", required=True, help="Path for the combined dataset")
    
    args = parser.parse_args()
    
    combine_datasets(args.lrs2, args.lrs3, args.output)

#Usage: python combine_datasets.py   --lrs2 /home/rishabh/Desktop/Datasets/lrs2_rf/lrs2/lrs2_video_seg16s/data_lrs2/   --lrs3 /home/rishabh/Desktop/Datasets/lrs3/433h_data   --output /home/rishabh/Desktop/Datasets/lrs_combined