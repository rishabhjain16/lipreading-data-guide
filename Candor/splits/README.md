# Official Candor Dataset Splits

This directory contains the official train/validation/test splits for the Candor dataset.

## Purpose

These split files ensure that everyone using the Candor dataset for AVSR research uses the **exact same train/val/test splits**, enabling fair comparison of results across different papers and implementations.

## Files

- `candor-train.id` - Training session IDs
- `candor-valid.id` - Validation session IDs  
- `candor-test.id` - Test session IDs

## Format

Each file contains one session ID per line:
```
23d4ec0e-9357-400e-ba97-386dcd264a9d
2c9f9799-b379-4c6e-91dd-38a1897d4ff6
...
```

## Creating Official Splits

If you're the first to process the full Candor dataset, create the official splits:

```bash
# After running step1 preprocessing
python preparation/create_official_splits.py \
    --candor-data-dir ./candor_output/candor_video \
    --output-dir ./splits \
    --split-ratios 0.7,0.15,0.15 \
    --seed 42
```

**Important**: Use seed=42 for consistency!

## Using Official Splits

When preprocessing, use the `--use-official-splits` flag:

```bash
python preparation/step2_generate_file_lists.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --use-official-splits \
    --splits-dir ./splits
```

This ensures your train/val/test splits match the official ones.

## Split Statistics

**Total Sessions**: [To be filled after full dataset processing]

| Split | Sessions | Percentage |
|-------|----------|------------|
| Train | TBD | 70% |
| Valid | TBD | 15% |
| Test | TBD | 15% |

## Reproducibility

**Random Seed**: 42  
**Split Method**: Session-based (no session overlap between splits)  
**Split Ratios**: 70% train, 15% validation, 15% test

## Distribution

When sharing your preprocessed Candor dataset or publishing results:

1. Include these split files in your distribution
2. Cite the Candor paper
3. Mention that you used the official splits
4. Report results on the official test set

## Verification

To verify you're using the correct splits:

```bash
# Check number of sessions in each split
wc -l splits/candor-*.id

# Check for overlaps (should return nothing)
cat splits/candor-*.id | sort | uniq -d
```

## Citation

If you use these splits, please cite the Candor dataset:

```bibtex
@article{reece2023candor,
  title={The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation},
  author={Reece, Andrew and Cooney, Gus and Bull, Peter and Chung, Christine},
  journal={Science Advances},
  volume={9},
  pages={eadf3197},
  year={2023}
}
```

## Notes

- These splits are **session-based**, meaning entire conversations are in one split
- This prevents data leakage (same conversation in train and test)
- Speakers may appear in multiple splits (different conversations)
- For speaker-independent evaluation, use speaker-based splits instead
