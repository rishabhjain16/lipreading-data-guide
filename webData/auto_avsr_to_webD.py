#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import csv
import re
import shutil
import tempfile
import hashlib
import uuid
import ffmpeg
import cv2
import tqdm
import webdataset as wds

class CsvDatasetConverter:
    def __init__(self, video_root, text_root, csv_file, output_dir, samples_per_shard, dataset_name):
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.samples_per_shard = samples_per_shard
        self.dataset_name = dataset_name

        if not self.video_root.is_dir():
            raise FileNotFoundError(f"Video root {video_root} does not exist.")
        if not self.text_root.is_dir():
            raise FileNotFoundError(f"Text root {text_root} does not exist.")
        if not self.csv_file.is_file():
            raise FileNotFoundError(f"CSV file {csv_file} does not exist.")

        parts = self.csv_file.stem.split('_')
        self.split = parts[1] if len(parts) > 1 else 'custom'

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_pattern = str(self.output_dir / f"{self.dataset_name}_{self.split}-%06d.tar")

    def parse_csv(self):
        samples = []
        with self.csv_file.open('r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                _, video_path_str, *_ = row
                video_rel_path = Path(*Path(video_path_str).parts[1:])
                video_path = self.video_root / video_rel_path
                text_path = self.text_root / video_rel_path.parent / (video_rel_path.stem + '.txt')
                wav_path = video_path.with_suffix('.wav')

                # Don't skip - raise error if missing
                assert video_path.is_file(), f"Video not found: {video_path}"
                assert text_path.is_file(), f"Text not found: {text_path}"

                sample_id = video_rel_path.with_suffix('').as_posix()

                samples.append({
                    'video_path': video_path,
                    'text_path': text_path,
                    'wav_path': wav_path if wav_path.is_file() else None,
                    'sample_id': sample_id,
                })
        print(f"Loaded {len(samples)} samples from {self.csv_file.name}")
        return samples

    def read_text(self, path):
        text = path.read_text(encoding='utf-8').strip()
        text = re.sub(r"[^a-zA-Z' ]", "", text)
        return text.upper()

    def get_video_metadata(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Cannot open video: {video_path}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = frames / fps if fps > 0 else 0
        return {"fps": fps, "num_frames": frames, "duration": duration, "width": width, "height": height}

    def generate_temp_path(self, video_path, wav_path, temp_dir):
        key_str = str(video_path) + str(wav_path)
        hashed = hashlib.md5(key_str.encode()).hexdigest()
        uid = uuid.uuid4().hex[:6]
        return temp_dir / f"{hashed}_{uid}.mp4"

    def merge_audio_video(self, video_path, wav_path, out_path):
        v = ffmpeg.input(str(video_path))
        a = ffmpeg.input(str(wav_path))
        ffmpeg.output(v.video, a.audio, str(out_path), vcodec='copy', acodec='aac', strict='experimental', loglevel='panic').overwrite_output().run()

    def convert(self):
        samples = self.parse_csv()
        assert len(samples) > 0, "No samples found"

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"merge_{self.split}_"))
        print(f"Temporary dir: {tmp_dir}")

        written_count = 0

        with wds.ShardWriter(self.output_pattern, maxcount=self.samples_per_shard) as sink:
            for idx, sample in enumerate(tqdm.tqdm(samples, desc="Converting")):
                # No try-except - let it crash if there's a problem
                
                if sample['wav_path'] and sample['wav_path'].is_file():
                    merged_path = self.generate_temp_path(sample['video_path'], sample['wav_path'], tmp_dir)
                    if not merged_path.is_file():
                        self.merge_audio_video(sample['video_path'], sample['wav_path'], merged_path)
                    video_file = merged_path
                else:
                    video_file = sample['video_path']

                video_bytes = video_file.read_bytes()
                transcript = self.read_text(sample['text_path'])
                meta = self.get_video_metadata(video_file)
                length_str = str(meta['num_frames'])
                meta['sample_id'] = sample['sample_id']
                meta['length'] = length_str

                item = {
                    '__key__': f"{idx:08d}",
                    'video': video_bytes,
                    'label': transcript.encode('utf-8'),
                    'length': length_str.encode('utf-8'),
                    'sample_id': sample['sample_id'].encode('utf-8'),
                    'json': json.dumps(meta).encode('utf-8'),
                }

                sink.write(item)
                written_count += 1
                
                # Print every 1000 samples
                if (idx + 1) % 1000 == 0:
                    print(f"\nWritten {written_count} samples so far...")

        shutil.rmtree(tmp_dir)
        
        print(f"\nTotal written: {written_count}")
        print(f"Expected: {len(samples)}")
        
        if written_count != len(samples):
            print(f"ERROR: Missing {len(samples) - written_count} samples!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--text_root', required=True)
    parser.add_argument('--csv_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--samples_per_shard', type=int, default=500)

    args = parser.parse_args()

    converter = CsvDatasetConverter(
        video_root=Path(args.video_root),
        text_root=Path(args.text_root),
        csv_file=Path(args.csv_file),
        output_dir=Path(args.output_dir),
        samples_per_shard=args.samples_per_shard,
        dataset_name=args.dataset_name,
    )
    converter.convert()
