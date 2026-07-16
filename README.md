# Lipreading Data Guide

A comprehensive toolkit for preparing popular lipreading and audio-visual speech recognition datasets into a unified, standardized format. Automates preprocessing steps including face tracking, mouth ROI extraction, and transcript alignment to simplify model training workflows. Works well with AV-HuBERT and Auto-AVSR.

## Supported Datasets

| Dataset | Status | Description | Size | Key Features |
|---------|--------|-------------|------|--------------|
| **LRS2** | ✅ Ready | Lip Reading Sentences 2 | ~140k utterances | BBC broadcasts, in-the-wild |
| **LRS3** | ✅ Ready | Lip Reading Sentences 3 | ~150k utterances | TED talks, high quality |
| **LRS_Combined** | ✅ Ready | Merged LRS2 + LRS3 | ~290k utterances | Unified corpus |
| **TCD-TIMIT** | ✅ Ready | TCD-TIMIT audiovisual corpus | ~27k utterances | HD video, controlled, 2 camera angles |
| **GRID** | ✅ Ready | Audio-visual speech corpus | ~34k utterances | Controlled vocabulary, 34 speakers |
| **LombardGrid** | ✅ Ready | Lombard effect speech | ~5.4k utterances | Noisy conditions, 54 speakers |
| **RoomReader** | ✅ Ready | Multiparty conversations | ~322 videos | Online meetings, 118 participants |
| **Candor** | ✅ Ready | Naturalistic conversations | ~1,656 conversations | Unscripted dyadic conversations |
| **VoxCeleb2** | 🔄 Planned | Speaker recognition dataset | ~1M utterances | Whisper transcription pipeline planned |
| **WildVSR** | ✅ Ready | Wild VSR test set | Test set | Generalization benchmark |
| **AVCocktail** | ✅ Ready | Cocktail party speech | Challenge dataset | Multi-speaker, noisy |
| **Muavic** | ⚠️ Experimental | Multilingual audio-visual | 9 languages | Speech recognition + translation |
| **MultiVSR** | 🔄 Planned | Large-scale multilingual VSR | ~1,400 hours | 20+ languages |

**Legend:**
- ✅ Ready: Fully tested and production-ready
- ⚠️ Experimental: Available but not fully tested
- 🔄 Planned: Future support planned (download instructions may be available)

## Utilities

- **Phones**: Phoneme conversion and mapping utilities for phonetic-level model training
- **webData**: Tools to convert Auto-AVSR prepared data into WebDataset and Hugging Face compatible formats

## Getting Started

Each dataset folder includes comprehensive documentation with step-by-step instructions for data acquisition, preprocessing, and preparation. Navigate to the specific dataset directory for detailed setup guides and processing workflows.

## Roadmap

We plan to expand the toolkit beyond English-centric benchmarks by adding support for multilingual and multi-party conversational datasets, including MultiVSR, MARC, MISP, MLD-VC, CI-AVSR, RUSAVIC, KMSAV, VISPER, Friends-MMC, AVSD, HAVRUS, ViCocktail, OLKAVS, F2F-JF, and Seamless Interaction. We believe these datasets can be effectively converted into a unified format, making them readily usable for AVSR and VSR research while enabling more comprehensive and standardized benchmarking across diverse languages and conversational settings.

## Dataset References

1. **LRS2**: Afouras, T., Chung, J. S., Senior, A., Vinyals, O., & Zisserman, A. (2018). Deep Audio-Visual Speech Recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)

2. **LRS3**: Afouras, T., Chung, J. S., & Zisserman, A. (2018). LRS3-TED: a large-scale dataset for visual speech recognition. *arXiv preprint arXiv:1809.00496*. [https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)

3. **TCD-TIMIT**: Harte, N., & Gillen, E. (2015). TCD-TIMIT: An audio-visual corpus of continuous speech. *IEEE Transactions on Multimedia*, 17(5), 603-615. [https://sigmedia.tcd.ie/](https://sigmedia.tcd.ie/TCDTIMIT/)

4. **WildVSR**: Djilali, Y. A. D., Narayan, S., LeBihan, E., Boussaid, H., Almazrouei, E., & Debbah, M. (2024). Do VSR Models Generalize Beyond LRS3? *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 6635-6644. [https://github.com/YasserdahouML/VSR_test_sethttps://github.com/YasserdahouML/VSR_test_set](https://github.com/YasserdahouML/VSR_test_set)

5. **VoxCeleb2**: Chung, J. S., Nagrani, A., & Zisserman, A. (2018). VoxCeleb2: Deep Speaker Recognition. *Interspeech 2018*. [https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)

6. **AVCocktail & AVYT**: Nguyen, T.-B., Pham, N.-Q., Waibel, A. (2025). Cocktail-Party Audio-Visual Speech Recognition. *Proc. Interspeech 2025*, 1828-1832. [https://arxiv.org/abs/2506.02178](https://arxiv.org/abs/2506.02178)

7. **MuAViC**: Anwar, A., Shi, B., Goswami, V., Hsu, W. N., Pino, J., & Wang, C. (2023). MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation. *arXiv preprint arXiv:2303.00628*. [https://github.com/facebookresearch/muavic](https://github.com/facebookresearch/muavic)

8. **GRID**: Cooke, M., Barker, J., Cunningham, S., & Shao, X. (2006). An audio-visual corpus for speech perception and automatic speech recognition. *The Journal of the Acoustical Society of America*, 120(5), 2421-2424. [https://zenodo.org/records/3625687](https://zenodo.org/records/3625687)

9. **Lombard GRID**: Alghamdi, N., Maddock, S., Marxer, R., Barker, J., & Brown, G. J. (2018). A corpus of audio-visual Lombard speech with frontal and profile views. *The Journal of the Acoustical Society of America*, 143(6), EL523-EL529. [https://zenodo.org/records/3228148](https://zenodo.org/records/3228148)

10. **RoomReader**: Reverdy, J., O'Connor Russell, S., Duquenne, L., Garaialde, D., Cowan, B. R., & Harte, N. (2022). RoomReader: A Multimodal Corpus of Online Multiparty Conversational Interactions. *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, 2517-2527. [https://aclanthology.org/2022.lrec-1.268/](https://aclanthology.org/2022.lrec-1.268/)

11. **MultiVSR**: Prajwal, K. R., Hegde, S., & Zisserman, A. (2025). Scaling Multilingual Visual Speech Recognition. *ICASSP 2025 - IEEE International Conference on Acoustics, Speech and Signal Processing*, 1-5. [https://github.com/Sindhu-Hegde/multivsr](https://github.com/Sindhu-Hegde/multivsr)

12. **Candor**: Reece, A., Cooney, G., Bull, P., & Chung, C. (2023). The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation. *Science Advances*, 9, eadf3197. [https://www.science.org/doi/10.1126/sciadv.adf3197](https://www.science.org/doi/10.1126/sciadv.adf3197) | [https://candor.usc.edu/](https://candor.usc.edu/)



## Codebase References

#### 1. AV-HuBERT
Shi, B., Hsu, W.-N., Lakhotia, K., & Mohamed, A. (2022).  
**Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction (AV-HuBERT)**.  
Paper: https://arxiv.org/abs/2201.02184  
GitHub: https://github.com/facebookresearch/av_hubert  

#### 2. Auto-AVSR
Ma, P., Haliassos, A., Fernandez-Lopez, A., Chen, H., Petridis, S., & Pantic, M. (2023).  
**Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels**.  
Paper: https://arxiv.org/abs/2303.14307  
GitHub: https://github.com/mpc001/auto_avsr
