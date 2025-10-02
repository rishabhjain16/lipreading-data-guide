#!/usr/bin/env python3
"""
MuAViC Step 0: Download Raw Data

IMPORTANT: The official MuAViC get_data.py script downloads videos AND applies
their own face cropping. We want to use RetinaFace instead.

RECOMMENDED APPROACH:
1. Run the official script to download everything:
   cd muavic
   python get_data.py --root-path /path/to/muavic_data --src-lang ar

2. The raw downloaded videos will be in: muavic_data/mtedx/video/{lang}/
   These are the full-frame videos BEFORE their cropping.

3. Use step1_prepare_muavic_retinaface.py to apply RetinaFace to these raw videos.

This wrapper script just calls the official get_data.py for convenience.

Usage:
    python step0_download_muavic.py \
        --root-path /path/to/muavic_data \
        --src-lang ar

Requirements:
    pip install numpy==1.23.5 opencv-python==4.8.1.78 wget yt-dlp pandas tqdm ffmpeg-python
"""

import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Download MuAViC data using official script"
    )
    parser.add_argument(
        "--root-path",
        required=True,
        type=Path,
        help="Path where MuAViC dataset will be downloaded",
    )
    parser.add_argument(
        "--src-lang",
        required=True,
        choices=["ar", "de", "el", "en", "es", "fr", "it", "pt", "ru"],
        help="Source language code",
    )
    
    args = parser.parse_args()
    
    # Find the official get_data.py script
    script_dir = Path(__file__).parent.parent
    muavic_script = script_dir / "muavic" / "get_data.py"
    
    if not muavic_script.exists():
        print(f"❌ Error: Official MuAViC script not found at: {muavic_script}")
        print("\nMake sure you have cloned the official MuAViC repository:")
        print("  git clone https://github.com/facebookresearch/muavic.git")
        sys.exit(1)
    
    print("="*60)
    print(f"Downloading MuAViC-{args.src_lang} data...")
    print("="*60)
    print(f"\nUsing official script: {muavic_script}")
    print(f"Output directory: {args.root_path}")
    print()
    
    # Run the official script
    cmd = [
        sys.executable,
        str(muavic_script),
        "--root-path", str(args.root_path),
        "--src-lang", args.src_lang
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*60)
        print(f"✅ Download completed!")
        print("="*60)
        print(f"\nNext step: Use step1_prepare_muavic_retinaface.py to apply RetinaFace preprocessing")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Download failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
