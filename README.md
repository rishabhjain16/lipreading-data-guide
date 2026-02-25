# Lipreading Data Guide

A comprehensive toolkit for preparing popular lip reading and visual speech recognition datasets into a unified, standardized format that simplifies model training workflows.

## Supported Datasets

- **LRS2**: Preparation and processing scripts for the Lip Reading Sentences 2 dataset
- **LRS3**: Preparation and processing scripts for the Lip Reading Sentences 3 dataset
- **LRS_Combined**: Utilities to merge prepared LRS2 and LRS3 datasets into a single corpus
- **TCD-TIMIT**: Processing pipeline for the TCD-TIMIT audiovisual speech corpus
- **WildVSR**: Preparation scripts for the Wild Visual Speech Recognition dataset
- **Voxceleb2**: Processing tools for the VoxCeleb2 speaker recognition dataset
- **AVCocktail and AVYT**: WebDataset format preparation for AVCocktail and MCOREC challenge datasets
- **Muavic**: Modified preprocessing compatible with RetinaFace and AutoAVSR pipelines (see Muavic folder for details)
- **Grid**: Processing with adjusted RetinaFace thresholds for better face region and landmark coverage (see folder for details)
- **Lombard Grid**: Specialized preprocessing with optimized RetinaFace parameters for Lombard effect speech (see folder for details)
- **RoomReader**: Work in progress - segment concatenation improvements needed to handle fragmentation
- **MultiVSR**: Large-scale multilingual VSR dataset - planned for future support due to dataset size
- **Candor**: Under consideration for future integration

## Utilities

- **Phones**: Phoneme conversion and mapping utilities for phonetic-level model training
- **webData**: Tools to convert Auto-AVSR prepared data into WebDataset and Hugging Face compatible formats

## Getting Started

Each dataset folder includes comprehensive documentation with step-by-step instructions for data acquisition, preprocessing, and preparation. Navigate to the specific dataset directory for detailed setup guides and processing workflows.

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

12. **Candor**: Reece, A., Cooney, G., Bull, P., & Chung, C. (2023). The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation. *Science Advances*, 9, eadf3197. [https://www.science.org/doi/10.1126/sciadv.adf3197](https://www.science.org/doi/10.1126/sciadv.adf3197)



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
