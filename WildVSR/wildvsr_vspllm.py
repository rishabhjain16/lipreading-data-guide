import os
import sys
import json
import unicodedata
import re

# Function to normalize text
def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    return text

# Check if directory path is provided
if len(sys.argv) < 2:
    print("Usage: python test_prep.py /path/to/data_directory")
    sys.exit(1)

# Get the directory path from command line argument
data_dir = sys.argv[1]

# Define paths
videos_dir = os.path.join(data_dir, "videos")
labels_file = os.path.join(data_dir, "labels.json")
output_dir = os.path.join(data_dir, "test_data")

print(f"Working with data directory: {data_dir}")
print(f"Videos directory: {videos_dir}")
print(f"Labels file: {labels_file}")
print(f"Output directory: {output_dir}")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Get absolute path for videos directory
abs_videos_dir = os.path.abspath(videos_dir)
print(f"Absolute videos directory path: {abs_videos_dir}")

# Check if the required files/directories exist
if not os.path.exists(videos_dir):
    print(f"Error: Videos directory not found at {videos_dir}")
    sys.exit(1)
if not os.path.exists(labels_file):
    print(f"Error: Labels file not found at {labels_file}")
    sys.exit(1)

try:
    # Read the entire file content
    with open(labels_file, 'r') as f:
        content = f.read()
    
    # Clean up the content to make it valid JSON if needed
    content = content.strip()
    if not content.startswith('{'):
        content = '{' + content
    if not content.endswith('}'):
        content = content + '}'
    
    # Parse the JSON
    try:
        data = json.loads(content)
        print(f"Successfully loaded JSON with {len(data)} entries")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        
        # Try an alternative approach - parse line by line
        print("Attempting alternative parsing method...")
        data = {}
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                # Extract key and value using string operations
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('"').strip('"')
                    value = parts[1].strip().strip(',').strip('"').strip('"')
                    data[key] = value
        
        print(f"Alternative parsing found {len(data)} entries")
    
    # Create test.tsv
    output_tsv = os.path.join(output_dir, "test.tsv")
    with open(output_tsv, 'w') as f:
        # Write the root directory as the first line
        f.write(f"{abs_videos_dir}\n")
        
        for video_id, transcript in data.items():
            # Get the filename without extension to use as speaker ID
            speaker_id = os.path.splitext(video_id)[0]
            
            # Create full path for video
            video_path = os.path.join(abs_videos_dir, video_id)
            
            # Create audio path with .wav extension
            audio_path = os.path.join(abs_videos_dir, f"{speaker_id}.wav")
            
            # Format with video path, audio path, and 0, 0 as placeholders for frame counts
            line = f"{speaker_id}\t{video_path}\t{audio_path}\t0\t0\n"
            f.write(line)
    
    # Create test.wrd
    output_wrd = os.path.join(output_dir, "test.wrd")
    with open(output_wrd, 'w') as f:
        for transcript in data.values():
            normalized_transcript = normalize_text(transcript)
            f.write(f"{normalized_transcript}\n")
    
    print(f"Created {output_tsv} with {len(data)} entries")
    print(f"Created {output_wrd} with {len(data)} entries")
    
    # Print a sample of the output for verification
    print("\nSample of test.tsv content:")
    with open(output_tsv, 'r') as f:
        lines = f.readlines()
        for i in range(min(3, len(lines))):
            print(lines[i].strip())
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
