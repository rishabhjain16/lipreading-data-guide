# Candor Preprocessing Workflow

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW CANDOR DATA                          │
│  • Candor_test/ (videos)                                        │
│  • candor_speechmatics/ (transcripts)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: Video Processing                      │
│  python preparation/step1_prepare_candor.py                     │
│                                                                  │
│  • Face detection (RetinaFace)                                  │
│  • Phrase segmentation (2-5s)                                   │
│  • Quality filtering                                            │
│  • Audio extraction (16kHz)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSED DATA                              │
│  candor_output/                                                 │
│  ├── candor_video/                                              │
│  │   └── {session_id}/                                          │
│  │       ├── {session_id}_spk0_0000.mp4                        │
│  │       ├── {session_id}_spk0_0000.wav                        │
│  │       └── ...                                                │
│  ├── candor_text/                                               │
│  └── labels/candor.csv                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  FIRST TIME?    │
                    │  (Full dataset) │
                    └─────────────────┘
                         │         │
                    YES  │         │  NO
                         │         │
                         ▼         │
        ┌────────────────────────┐│
        │ STEP 1.5: Create Splits││
        │ (ONE TIME ONLY!)       ││
        │                        ││
        │ create_official_splits ││
        │ --seed 42              ││
        └────────────────────────┘│
                         │         │
                         ▼         │
        ┌────────────────────────┐│
        │   splits/              ││
        │   ├── candor-train.id  ││
        │   ├── candor-valid.id  ││
        │   └── candor-test.id   ││
        └────────────────────────┘│
                         │         │
                         └────┬────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: Generate Training Manifests                 │
│  python preparation/step2_generate_file_lists.py                │
│  --use-official-splits                                          │
│                                                                  │
│  • Load official splits                                         │
│  • Count frames                                                 │
│  • Create TSV manifests                                         │
│  • Create word files                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING-READY DATA                           │
│  candor_output/metadata/                                        │
│  ├── train.tsv, train.wrd                                       │
│  ├── valid.tsv, valid.wrd                                       │
│  └── test.tsv, test.wrd                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  TRAIN MODEL!   │
                    └─────────────────┘
```

## Two Scenarios

### Scenario A: You're Processing Full Dataset (First Time)

```bash
# 1. Process videos
python preparation/step1_prepare_candor.py \
    --candor-path /path/to/Candor \
    --speechmatics-path /path/to/candor_speechmatics \
    --output-path ./candor_output \
    --crop-type lips

# 2. Create official splits (ONE TIME!)
python preparation/create_official_splits.py \
    --candor-data-dir ./candor_output/candor_video \
    --output-dir ./splits \
    --seed 42

# 3. Generate manifests using your new splits
python preparation/step2_generate_file_lists.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --use-official-splits \
    --splits-dir ./splits

# 4. Share splits/ folder with others!
```

### Scenario B: You're Using Existing Official Splits

```bash
# 0. Get official splits (already in splits/ folder)

# 1. Process videos
python preparation/step1_prepare_candor.py \
    --candor-path /path/to/Candor \
    --speechmatics-path /path/to/candor_speechmatics \
    --output-path ./candor_output \
    --crop-type lips

# 2. Generate manifests using existing splits
python preparation/step2_generate_file_lists.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --use-official-splits \
    --splits-dir ./splits

# Done!
```

## Key Points

1. **Step 1.5 is optional** - Only run if you're creating official splits
2. **Run Step 1.5 AFTER Step 1** - Need processed data to split
3. **Run Step 1.5 BEFORE Step 2** - Need splits to create manifests
4. **Run Step 1.5 ONCE** - Splits should be fixed and shared
5. **Most users skip Step 1.5** - Just use existing splits

## Timeline

**Your Current Situation** (10-session test subset):
- ✅ Step 1: Done (processed videos)
- ✅ Step 1.5: Done (created test splits)
- ✅ Step 2: Can run now with `--use-official-splits`

**When You Get Full Dataset**:
- ✅ Step 1: Process all sessions
- ✅ Step 1.5: Re-run to create full splits (replaces test splits)
- ✅ Step 2: Generate manifests with full splits
- ✅ Share: Distribute splits/ folder with preprocessed data

## Summary

Think of it as:
- **Step 1**: Process raw data → processed data
- **Step 1.5**: Decide train/val/test splits (once)
- **Step 2**: Create training manifests using splits
