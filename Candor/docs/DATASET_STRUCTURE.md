# Candor Dataset Structure Analysis

## Overview
Candor is a **conversational dyadic dataset** featuring natural conversations between two speakers. Each conversation session contains synchronized video, audio, and word-level transcriptions with precise timestamps.

## Dataset Statistics
- **Sample Size**: 10 conversation sessions (in test sample)
- **Duration**: ~26 minutes per session (average)
- **Format**: Dyadic conversations (2 speakers per session)
- **Video Quality**: 320x240 per speaker, 640x240 combined, 30 fps
- **Audio**: Stereo (L/R channels for each speaker)

---

## Directory Structure

### Main Dataset Structure
```
Candor_test/
├── {session_id}/                                    # UUID format (e.g., 23d4ec0e-9357-400e-ba97-386dcd264a9d)
│   ├── processed/                                   # ✅ PRIMARY FOLDER FOR AVSR
│   │   ├── {session_id}.mp4                        # Combined video (640x240, side-by-side)
│   │   ├── {session_id}.mp3                        # Combined audio (stereo)
│   │   ├── {user_id_1}.mp4                         # Speaker 1 individual video (320x240) ✅
│   │   ├── {user_id_2}.mp4                         # Speaker 2 individual video (320x240) ✅
│   │   ├── channel_map.json                        # Maps L/R channels to user IDs
│   │   └── thumbnail.png                           # Session thumbnail
│   ├── transcription/                               # ✅ TRANSCRIPTS
│   │   ├── transcribe_output.json                  # Raw ASR output (detailed)
│   │   ├── transcript_audiophile.csv               # Turn-level transcripts (clean)
│   │   ├── transcript_backbiter.csv                # Alternative transcript version
│   │   └── transcript_cliffhanger.csv              # Alternative transcript version
│   ├── raw/                                         # Original raw recordings
│   ├── metadata.json                                # Session metadata (speakers, timing)
│   ├── audio_video_features.csv                    # Extracted features (MFCC, facial, etc.)
│   └── survey.csv                                   # Post-conversation survey data
```

### Speechmatics Folder (Word-Level Annotations)
```
candor_speechmatics/
├── {session_id}_0.json                              # Speaker 0 word-level JSON ✅
├── {session_id}_1.json                              # Speaker 1 word-level JSON ✅
└── {session_id}.TextGrid                            # Combined TextGrid (both speakers) ✅
```

---

## Key Files for AVSR Pipeline

### 1. Video Files (Individual Speakers)
**Location**: `processed/{user_id}.mp4`

**Properties**:
- Resolution: 320x240 pixels
- Frame rate: 30 fps
- Duration: ~1616 seconds (~26 minutes)
- Format: MP4 with H.264 video + AAC audio
- Contains: Individual speaker's video with their audio track

**Example**:
```
5e531b2205acdb33c0f5f24c.mp4  (Speaker 1)
5ee91d6e634e8e1290804250.mp4  (Speaker 2)
```

### 2. Transcripts (Turn-Level)
**Location**: `transcription/transcript_audiophile.csv`

**Format**:
```csv
turn_id,speaker,start,stop,utterance,interval,delta,questions,end_question,overlap,n_words
0,5e531b2205acdb33c0f5f24c,11.34,95.56,"shit. Go go go...",84.22,0,False,False,21
1,5ee91d6e634e8e1290804250,95.84,96.25,"Yeah,",0.28,0.41,0,False,False,1
```

**Columns**:
- `turn_id`: Sequential turn number
- `speaker`: User ID (matches video filename)
- `start`: Start time in seconds
- `stop`: End time in seconds
- `utterance`: Transcript text
- `interval`: Duration of utterance
- `delta`: Time since previous turn
- `overlap`: Boolean indicating speaker overlap
- `n_words`: Word count

**Key Features**:
- ✅ Turn-level segmentation (natural conversation chunks)
- ✅ Precise timestamps (start/stop)
- ✅ Speaker identification
- ✅ Overlap detection (important for conversational AVSR)

### 3. Word-Level Transcripts (Speechmatics)
**Location**: `candor_speechmatics/{session_id}_0.json` and `{session_id}_1.json`

**Format** (JSON):
```json
{
  "results": [
    {
      "alternatives": [{"content": "He", "confidence": 0.49, "speaker": "UU"}],
      "start_time": 11.53,
      "end_time": 11.77,
      "type": "word"
    },
    {
      "alternatives": [{"content": "said", "confidence": 0.72, "speaker": "UU"}],
      "start_time": 11.77,
      "end_time": 12.28,
      "type": "word"
    }
  ]
}
```

**Key Features**:
- ✅ Word-level timestamps (precise alignment)
- ✅ Confidence scores per word
- ✅ Punctuation markers (separate entries)
- ✅ Speaker diarization

**Location**: `candor_speechmatics/{session_id}.TextGrid`

**Format** (Praat TextGrid):
```
File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0.0
xmax = 1614.45
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "23d4ec0e-9357-400e-ba97-386dcd264a9d_0"  # Speaker 0
        intervals: size = 4582
            intervals [2]:
                xmin = 11.53
                xmax = 11.77
                text = "He"
    item [2]:
        class = "IntervalTier"
        name = "23d4ec0e-9357-400e-ba97-386dcd264a9d_1"  # Speaker 1
        intervals: size = 3891
```

**Key Features**:
- ✅ Two tiers (one per speaker)
- ✅ Word-level intervals with timestamps
- ✅ Empty intervals for silence/pauses
- ✅ Compatible with Praat for visualization

### 4. Channel Mapping
**Location**: `processed/channel_map.json`

**Format**:
```json
{
  "L": "5e531b2205acdb33c0f5f24c",
  "R": "5ee91d6e634e8e1290804250"
}
```

Maps audio channels (L/R) to user IDs (video filenames).

**CRITICAL MAPPING RULE**:
```
Speechmatics _0 → Channel L (Left)  → First speaker in metadata.json
Speechmatics _1 → Channel R (Right) → Second speaker in metadata.json
```

**Example**:
```
Session: 23d4ec0e-9357-400e-ba97-386dcd264a9d

23d4ec0e-9357-400e-ba97-386dcd264a9d_0.json (Speechmatics)
  ↓
channel_0.wav (Left audio channel)
  ↓
channel_map.json: "L" = "5e531b2205acdb33c0f5f24c"
  ↓
Video file: 5e531b2205acdb33c0f5f24c.mp4

23d4ec0e-9357-400e-ba97-386dcd264a9d_1.json (Speechmatics)
  ↓
channel_1.wav (Right audio channel)
  ↓
channel_map.json: "R" = "5ee91d6e634e8e1290804250"
  ↓
Video file: 5ee91d6e634e8e1290804250.mp4
```

**Verification Code**:
```python
import json

session_id = "23d4ec0e-9357-400e-ba97-386dcd264a9d"

# Load channel map
with open(f"processed/channel_map.json") as f:
    channel_map = json.load(f)

# Mapping rule
video_file_0 = f"{channel_map['L']}.mp4"  # For _0.json
video_file_1 = f"{channel_map['R']}.mp4"  # For _1.json

print(f"{session_id}_0.json → {video_file_0}")
print(f"{session_id}_1.json → {video_file_1}")
```

### 5. Session Metadata
**Location**: `metadata.json`

**Key Information**:
- Session ID and creation timestamp
- Speaker user IDs
- Original video filenames and durations
- Audio/video synchronization offsets
- File sizes and technical details

---

## Preprocessing Strategy for AVSR

### Recommended Approach: Word-Level Segmentation

**Why Word-Level?**
1. ✅ Precise timestamps for each word (from Speechmatics JSON/TextGrid)
2. ✅ Better alignment with lip movements
3. ✅ Consistent with LRS2/LRS3 preprocessing (utterance-level)
4. ✅ Avoids long segments with silence/pauses

**Alternative: Turn-Level Segmentation**
- Pros: Natural conversation chunks, easier to process
- Cons: Long segments (up to 84 seconds), includes pauses/overlaps
- Use case: Conversational AVSR with context

### Segmentation Options

#### Option 1: Word-Level Segments (Recommended)
**Input**: `candor_speechmatics/{session_id}_0.json`
**Output**: Individual word clips (e.g., `{session_id}_spk0_0001.mp4`)

**Advantages**:
- Fine-grained alignment
- Consistent with LRS2/LRS3 word-level training
- Better for pure lip reading models

**Challenges**:
- Very short segments (0.2-1.0 seconds per word)
- May need to group into phrases (2-5 words)

#### Option 2: Phrase-Level Segments (Balanced)
**Input**: Group consecutive words from Speechmatics JSON
**Output**: Phrase clips (e.g., `{session_id}_spk0_0001.mp4`)

**Strategy**:
- Group words into 2-5 second phrases
- Break at punctuation (periods, commas)
- Respect silence gaps (>0.5s)

**Advantages**:
- ✅ Better segment length for training (2-5 seconds)
- ✅ Natural phrase boundaries
- ✅ Still precise alignment

#### Option 3: Turn-Level Segments (Conversational)
**Input**: `transcription/transcript_audiophile.csv`
**Output**: Turn clips (e.g., `{session_id}_spk0_turn001.mp4`)

**Advantages**:
- Natural conversation flow
- Includes prosody and context
- Good for conversational AVSR

**Challenges**:
- Variable length (0.5-84 seconds)
- May include overlaps
- Long silences within turns

---

## Recommended Pipeline (3-Step)

### Step 1: Video Preprocessing with Face Detection
**Input**: 
- Videos: `processed/{user_id}.mp4`
- Transcripts: `candor_speechmatics/{session_id}_0.json` and `_1.json`
- Channel map: `processed/channel_map.json`

**Process**:
1. Load individual speaker videos
2. Parse Speechmatics JSON for word-level timestamps
3. Group words into phrases (2-5 seconds, break at punctuation)
4. For each phrase:
   - Extract video segment using timestamps
   - Apply RetinaFace face detection
   - Crop lips (96x96) or face (224x224)
   - Extract audio segment (16kHz mono WAV)
   - Save text transcript

**Output Structure**:
```
candor_video/
├── {session_id}/
│   ├── spk0/
│   │   ├── {session_id}_spk0_0001.mp4
│   │   ├── {session_id}_spk0_0001.wav
│   │   ├── {session_id}_spk0_0001.txt
│   │   └── ...
│   └── spk1/
│       ├── {session_id}_spk1_0001.mp4
│       └── ...
```

### Step 2: Generate File Lists
**Input**: Processed videos from Step 1
**Output**: 
- `file.list`: List of video paths
- `label.list`: List of transcripts
- Train/val/test splits (speaker-based or session-based)

### Step 3: Create Training Metadata
**Input**: File lists from Step 2
**Output**:
- `train.tsv`, `valid.tsv`, `test.tsv`: Training manifests
- `train.wrd`, `valid.wrd`, `test.wrd`: Word transcripts
- `nframes.audio`, `nframes.video`: Frame counts
- `dict.wrd.txt`: Vocabulary dictionary

---

## Key Considerations

### 1. Speaker Overlap Handling
**Challenge**: Conversational dataset has overlapping speech
**Solutions**:
- Use `overlap` column in CSV to filter/flag overlapping segments
- Option A: Skip overlapping segments (clean training data)
- Option B: Keep overlaps (realistic conversational AVSR)

### 2. Silence/Pause Handling
**Challenge**: Long pauses within turns
**Solutions**:
- Use word-level timestamps to detect silence gaps
- Break segments at gaps >0.5 seconds
- Filter out segments with <50% speech activity

### 3. Segmentation Granularity
**Recommendation**: Phrase-level (2-5 seconds)
**Rationale**:
- Too short (word-level): 0.2-1.0s, insufficient context
- Too long (turn-level): 0.5-84s, too variable
- Just right (phrase-level): 2-5s, natural boundaries

### 4. Speaker Identification
**Mapping**:
- User IDs (UUIDs) → Anonymous speaker IDs (spk0, spk1)
- Maintain mapping in CSV metadata
- Use channel_map.json to link audio channels

### 5. Data Splits
**Options**:
- **Session-based**: Split by conversation sessions (no session leakage)
- **Speaker-based**: Split by speakers (no speaker leakage)
- **Recommended**: Session-based (70/15/15 train/val/test)

---

## Comparison with Other Datasets

| Feature | Candor | LRS2/LRS3 | RoomReader | GRID |
|---------|--------|-----------|------------|------|
| **Type** | Dyadic conversation | Monologue (TV) | Multiparty tutorial | Controlled sentences |
| **Speakers** | 2 per session | 1 per video | 3-5 per session | 1 per video |
| **Duration** | ~26 min/session | 3-10 sec/clip | 10-15 min/session | ~3 sec/clip |
| **Overlap** | Yes (natural) | No | Yes (frequent) | No |
| **Vocabulary** | Unrestricted | Unrestricted | Unrestricted | 51 words (fixed) |
| **Segmentation** | Word/turn-level | Sentence-level | Utterance-level | Sentence-level |
| **Video Quality** | 320x240, 30fps | 224x224, 25fps | Variable (online) | 360x288, 25fps |
| **Transcripts** | Word + turn-level | Sentence-level | Word-level | Sentence-level |

**Candor's Unique Challenges**:
- ✅ Conversational dynamics (overlaps, turn-taking)
- ✅ Natural speech (disfluencies, interruptions)
- ✅ Longer segments requiring smart segmentation
- ✅ Two speakers per video (need individual processing)

---

## Next Steps

1. **Decide on segmentation strategy**: Word-level, phrase-level, or turn-level
2. **Create preprocessing script**: Adapt RoomReader/LRS3 pipeline for Candor
3. **Handle overlaps**: Filter or flag overlapping segments
4. **Test on sample**: Process 1-2 sessions to validate pipeline
5. **Scale to full dataset**: Process all sessions with parallel processing

---

## Files Summary

### Essential for AVSR:
- ✅ `processed/{user_id}.mp4` - Individual speaker videos
- ✅ `candor_speechmatics/{session_id}_0.json` - Word-level timestamps (Speaker 0)
- ✅ `candor_speechmatics/{session_id}_1.json` - Word-level timestamps (Speaker 1)
- ✅ `processed/channel_map.json` - Speaker identification

### Optional/Supplementary:
- `transcription/transcript_audiophile.csv` - Turn-level transcripts (alternative)
- `candor_speechmatics/{session_id}.TextGrid` - Praat format (visualization)
- `metadata.json` - Session metadata
- `audio_video_features.csv` - Pre-extracted features (not needed for raw AVSR)

---

## Questions to Resolve

1. **Segmentation granularity**: Word-level, phrase-level (2-5s), or turn-level?
2. **Overlap handling**: Skip overlaps or keep for conversational AVSR?
3. **Minimum segment length**: Filter segments <1 second?
4. **Data split strategy**: Session-based or speaker-based?
5. **Transcript source**: Use Speechmatics JSON (word-level) or CSV (turn-level)?

**Recommendation**: Start with **phrase-level segmentation** (2-5 seconds) using **Speechmatics JSON**, **skip overlaps** for clean training data, and use **session-based splits**.
