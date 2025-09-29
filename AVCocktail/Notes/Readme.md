
# Understanding AVSR Chunking & WER in **AVCocktail**

This note explains **why chunking exists**, how the **AVCocktail** dataset structures audio–video–text, how each **chunking strategy** works (`fixed_chunk`, `asd_chunk`, `gold_chunk`), and **how WER is computed** per split. It also shows **why the same session yields different scores** under different chunking schemes and how **overlapping transcript cues** are handled.

> **TL;DR**  
> - AVSR models work on **short segments**, so long sessions are **chunked**.  
> - **Gold** ≈ sentence-aligned → *lowest WER*. **ASD** ≈ speech-activity-aligned → *middle*. **Fixed** ≈ time-only → *highest WER*.  
> - WER is computed **per chunk** against text pulled from the session-level `.label` (WEBVTT) by **time overlap**; then aggregated over the split.

---

## Table of Contents
- [1. Why chunking?](#1-why-chunking)
- [2. What’s in the dataset on disk?](#2-whats-in-the-dataset-on-disk)
- [3. How does the loader build audio / video / text?](#3-how-does-the-loader-build-audio--video--text)
- [4. Chunking strategies](#4-chunking-strategies)
- [5. How WER is computed per split](#5-how-wer-is-computed-per-split)
- [6. Handling overlapping cues](#6-handling-overlapping-cues)
- [7. Why outputs differ across chunking](#7-why-outputs-differ-across-chunking)
- [8. Worked example](#8-worked-example)
- [9. Typical WER differences (from your run)](#9-typical-wer-differences-from-your-run)
- [10. FAQ](#10-faq)
- [References](#references)

---

## 1. Why chunking?
Long sessions (minutes+) won’t fit the usual AVSR model context window or GPU memory. Datasets therefore ship **shorter segments** (chunks) so that:

- Models receive **manageable inputs** (e.g., 5–10 s).  
- Evaluation can report **segment-level WER** and then aggregate.

> AVSR pipelines commonly decode lip frames & audio per segment and fuse them in the encoder (e.g., AV‑HuBERT), then decode with CTC/attention.  
> See HF Datasets’ discussion of Arrow-backed random access & caching, and video/audio features. [^hf-arrow] [^hf-video] [^hf-audio]

---

## 2. What’s in the dataset on disk?
A typical session (e.g., `video_32`) contains two top folders:

```
AVCocktail/
├─ videos/
│  └─ video_32/
│     ├─ asd-chunk-eval-0.tar
│     ├─ fixed-chunk-eval-0.tar
│     └─ gold-chunk-eval-0.tar
└─ labels/
   └─ video_32/
      └─ asd-chunk-eval-0.label   # WEBVTT transcript for the session timeline
```

- **videos/** shards are **WebDataset** tars; each tar holds many samples. [^wds]
- **labels/** has a **single `.label`** (WEBVTT) per session with *all* cues over time.
- When you `load_dataset(...)`, HF Datasets resolves shards, converts them to **Arrow** for fast memory-mapped access, and writes a `dataset_info.json` describing **features, splits, checksums, sizes**. [^hf-cache]

> In the cached Arrow, you’ll see columns like `video` (MP4 bytes), `start_time`, `end_time`, `sample_id`, plus `__key__`/`__url__` from WebDataset.

---

## 3. How does the loader build audio / video / text?
1) **Video**  
Cast the `video` column to a `Video()` feature; frames are decoded lazily on access. [^hf-video]

2) **Audio**  
The audio track is **demuxed from the same MP4** using the decoder backend (ffmpeg/pyav/torchcodec) when preparing model inputs. (Many loaders, e.g., PyTorchVideo, expose `decode_audio=True`.)

3) **Text**  
The evaluation code reads the session’s `.label` (WEBVTT) and, for each chunk, **collects all cues whose time range intersects the chunk’s `[start_time, end_time]`**, concatenating them to form the **reference transcript** for that chunk.

---

## 4. Chunking strategies

### `fixed_chunk`
- Equal-length windows (e.g., every 10 s), **ignoring sentence boundaries**.
- Pros: simple, uniform sizes.  
- Cons: often splits words/phrases → **poor context**.

### `asd_chunk`
- Chunks follow **Automatic Speech Detection** boundaries (speech-active regions).  
- Pros: closer to natural utterances.  
- Cons: boundary errors if ASD is imperfect.

### `gold_chunk`
- **Ground-truth** segmentation (utterance-aligned).  
- Pros: best alignment to text → **easiest for the model**.  
- Cons: variable lengths; relies on perfect labels.

> The AVSRCocktail model card & README enumerate these sets and how evaluation reports WER by set. [^avsrc-model] [^avsrc-github]

---

## 5. How WER is computed per split
For each chunk in a split (e.g., `fixed_chunk`):

1. **Build reference text** from `.label` by time overlap with `[start_time, end_time]`.
2. **Run the model** on the chunk’s audio+video to get a hypothesis string.
3. **Normalize** (lowercase, strip punctuation per script’s rules).  
4. **Compute WER per chunk** using Levenshtein on word tokens:

```text
WER = (Substitutions + Deletions + Insertions) / (Number of reference words)
```

5. **Aggregate across the split**:
   - Most pipelines micro-average (sum errors & words over all chunks, then divide). Some print a simple mean of per-chunk WERs. Check the script’s final reducer if you need exact aggregation semantics.

---

## 6. Handling overlapping cues
WEBVTT cues (and chunk windows) can overlap. A robust rule is **interval intersection**:

```python
# Pseudocode
for chunk in chunks:
    ref_cues = [c for c in vtt_cues if c.end > chunk.start and c.start < chunk.end]
    ref_text = " ".join(normalize(c.text) for c in ref_cues)
    hyp_text = decode_model(chunk.audio, chunk.video)
    wer_chunk = WER(ref_text, hyp_text)
```

- This ensures **no words are dropped** at boundaries.  
- Some duplication across adjacent chunks is expected; it doesn’t bias WER because evaluation is **per chunk**.

---

## 7. Why outputs differ across chunking
Even for the **same session**, chunking changes what the model sees:

- **Boundary cuts** (fixed) remove preceding/following context → more **insertions/deletions**.  
- **Utterance-aligned** (gold) preserves linguistic context → few edits.  
- **ASD** sits between both extremes.

---

## 8. Worked example
Session text:

```text
00:00:08.000 → 00:00:15.000  I work in a law firm and handle accounting.
```

**Gold chunk** (8–15 s):  
- Ref: *I work in a law firm and handle accounting*  
- Hyp: (correct) → WER = 0/9 = **0.00**

**Fixed chunk A** (0–10 s):  
- Ref (overlap): *I work in a law* (4–10 s)  
- Hyp: *I work in a law firm*  
- Errors: 1 insertion (*firm*) → WER = 1/5 = **0.20**

**Fixed chunk B** (10–20 s):  
- Ref (overlap): *firm and handle accounting* (10–15 s)  
- Hyp: *and handle accounting*  
- Errors: 1 deletion (*firm*) → WER = 1/4 = **0.25**

Aggregating over many chunks, **fixed** tends to inflate WER, **gold** to reduce it, **asd** in-between.

---

## 9. Typical WER differences (from your run)

| Split         | WER     |
|---------------|---------|
| `gold_chunk`  | 0.0769  |
| `asd_chunk`   | 0.1818  |
| `fixed_chunk` | 0.6084  |

These magnitudes are expected given the boundary effects explained above.

---

