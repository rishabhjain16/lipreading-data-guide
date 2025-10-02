# MuAViC Preprocessing: Two Approaches

## Overview

We provide two preprocessing scripts for MuAViC dataset with RetinaFace:

1. **step1_prepare_muavic_retinaface.py** - Standard approach (uses pre-made segments)
2. **step1_prepare_muavic_retinaface_smart.py** - Smart approach (VTT-based with face verification)

## Approach 1: Standard (step1_prepare_muavic_retinaface.py)

### How it works:
- Uses the pre-made `segments` file from mTEDx dataset
- Segments are based on audio/subtitle timing
- Processes all segments regardless of face presence
- Fast and straightforward

### Pros:
- ✅ Follows official mTEDx segmentation
- ✅ Faster processing (no face verification step)
- ✅ Consistent with original MuAViC dataset structure
- ✅ Good for datasets where speaker is always on-camera

### Cons:
- ❌ May include segments where speaker is off-camera
- ❌ May include segments with no visible face
- ❌ May include segments with multiple speakers
- ❌ No quality filtering based on face detection

### Usage:
```bash
python step1_prepare_muavic_retinaface.py \
    --data-dir /path/to/data \
    --root-dir /path/to/output \
    --language ar \
    --split test \
    --crop-type lips
```

### Output structure:
```
muavic_video/
  ar/
    test/
      VIDEO_ID/
        VIDEO_ID_0000.mp4  # Uses original segment IDs
        VIDEO_ID_0001.mp4
```

---

## Approach 2: Smart (step1_prepare_muavic_retinaface_smart.py)

### How it works:
- Reads VTT subtitle files directly for transcript timing
- Merges short segments into longer, more natural clips
- **Samples frames to check face presence** before full processing
- Only processes segments with consistent face detection (default: 70% of frames)
- Filters out segments where speaker is off-camera

### Pros:
- ✅ Higher quality dataset (only segments with visible faces)
- ✅ Filters out off-camera speech
- ✅ Filters out segments with no face
- ✅ More natural segment boundaries (merged from VTT)
- ✅ Includes face_ratio metric in CSV for quality analysis
- ✅ Configurable face detection threshold

### Cons:
- ❌ Slower (face verification adds processing time)
- ❌ May skip valid segments if face detection fails
- ❌ Different segmentation than original mTEDx
- ❌ Smaller dataset (filtered)

### Usage:
```bash
python step1_prepare_muavic_retinaface_smart.py \
    --data-dir /path/to/data \
    --root-dir /path/to/output \
    --language ar \
    --split test \
    --crop-type lips \
    --min-face-ratio 0.7 \
    --min-duration 1.0 \
    --max-duration 15.0
```

### Parameters:
- `--min-face-ratio`: Minimum ratio of frames with detected faces (0.0-1.0, default: 0.7)
  - 0.7 = at least 70% of frames must have a detectable face
  - Higher = stricter filtering, higher quality
  - Lower = more permissive, larger dataset
- `--min-duration`: Minimum segment duration in seconds (default: 1.0)
- `--max-duration`: Maximum segment duration in seconds (default: 15.0)

### Output structure:
```
muavic_video_smart/
  ar/
    test/
      VIDEO_ID/
        VIDEO_ID_0000.mp4  # New segment IDs based on VTT
        VIDEO_ID_0001.mp4
```

### CSV includes face_ratio:
```csv
language,split,seg_id,video_id,video_path,transcript,word_count,start_sec,end_sec,face_ratio,detector,crop_type,resolution
ar,test,U7yWsUX5TLY_0000,U7yWsUX5TLY,muavic_video_smart/ar/test/...,صباح الخير,2,2.89,6.66,0.85,retinaface,lips,96x96
```

---

## Which one should you use?

### Use **Standard** if:
- You want to match the original MuAViC dataset structure
- You need faster processing
- Your videos have speakers consistently on-camera
- You want maximum dataset size

### Use **Smart** if:
- You want higher quality segments (only with visible faces)
- You're okay with a smaller but cleaner dataset
- Your videos have speakers going off-camera
- You want to filter out low-quality segments
- You need face presence metrics for analysis

---

## Example Comparison

### Standard approach:
```
Total segments: 1066
Processed: 950
Skipped: 116 (processing errors)
```

### Smart approach:
```
Total VTT segments: 1200
After merging: 800
Processed: 650
Skipped (no face): 100
Skipped (other): 50

Result: 650 high-quality segments with verified face presence
```

---

## Performance Tips

### For Standard:
- Use `--groups` and `--job-index` for parallel processing
- Process multiple splits simultaneously

### For Smart:
- Adjust `--min-face-ratio` based on your quality requirements
  - 0.5 = permissive (50% frames with face)
  - 0.7 = balanced (70% frames with face) **recommended**
  - 0.9 = strict (90% frames with face)
- Use `--min-duration` to filter very short clips
- Use `--max-duration` to avoid very long segments

---

## Data Quality Analysis

After processing with Smart approach, you can analyze face detection quality:

```python
import pandas as pd

df = pd.read_csv('labels/muavic_ar_test_smart.csv')

# Distribution of face ratios
print(df['face_ratio'].describe())

# Filter by quality
high_quality = df[df['face_ratio'] >= 0.8]
print(f"High quality segments (>80% face): {len(high_quality)}")

# Check segments with lower face ratios
low_quality = df[df['face_ratio'] < 0.7]
print(f"Lower quality segments: {len(low_quality)}")
```

---

## Recommendation

**Start with Smart approach** for most use cases:
- Better data quality
- Face presence verification
- More robust for real-world videos
- Includes quality metrics

**Fall back to Standard** if:
- You need exact mTEDx compatibility
- Processing time is critical
- Dataset size is more important than quality
