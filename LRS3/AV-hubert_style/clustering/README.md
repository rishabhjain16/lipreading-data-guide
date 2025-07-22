# AV-HuBERT Label Preparation

This folder contains scripts for preparing AV-HUBERT labels from tsv files, the
steps are:
1. feature extraction
2. k-means clustering
3. k-means application

## Installation
To prepare labels, you need some additional packages:
```
pip install -r requirements.txt
```

## Data preparation

`*.tsv` files contains a list of audio, where each line is the root, and
following lines are the subpath and number of frames of each video and audio separated by `tab`:
```
<root-dir>
<id-1> <video-path-1> <audio-path-1> <video-number-frames-1> <audio-number-frames-1>
<id-2> <video-path-2> <audio-path-2> <video-number-frames-2> <audio-number-frames-2>
...
```
See [here](https://github.com/Sally-SH/VSP-LLM/blob/main/README.md#data-preprocessing) for data preparation for LRS3. 

## AV-HuBERT feature extraction
To extract features from the 12-th transformer layer of a trained
AV-HuBERT model saved at `${ckpt_path}`, run:
```sh
python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} 12 ${nshard} ${rank} ${feat_dir} --user_dir `pwd`/../
Example: python dump_hubert_feature.py /home/rishabh/Desktop/Dataset/lrs3/433h_data train /home/rishabh/Desktop/Experiments/VSP-LLM/checkpoints/large_vox_iter5.pt 12 1 0 /home/rishabh/Desktop/Experiments/VSP-LLM/features/433h/ --user_dir `pwd`/../

```
Features would also be saved at `${feat_dir}/${split}_${rank}_${nshard}.{npy,len}`.

- if out-of-memory, decrease the chunk size with `--max_chunk`


## K-means clustering
To fit a k-means model with 200 clusters on 10% of the `${split}` data, run
```sh
python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} 200 --percent 0.1

Example: python learn_kmeans.py /data/ssd4/data_rishabh/avhubert_prep/dump_feat/ train 1 /data/ssd4/data_rishabh/avhubert_prep/kmeans/kmean_500_random_n_20.km 500 --percent 0.2 --init "random" --batch_size 20000 --max_no_improvement 5000 --n_init=20 

```
This saves the k-means model to `${km_path}`.

- set `--precent -1` to use all data
- more kmeans options can be found with `-h` flag


## K-means application
To apply a trained k-means model `${km_path}` to obtain labels for `${split}`, run
```sh
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}

Example: python dump_km_label.py /data/ssd2/data_rishabh/avhubert_prep/dump_feat/ train  /data/ssd2/data_rishabh/avhubert_prep/kmeans_lrs2/kmean_500_plus_n_20.km  1 0 /data/ssd2/data_rishabh/avhubert_prep/km_labels_lrs2/
```
This would extract labels for the `${rank}`-th shard out of `${nshard}` shards
and dump them to `${lab_dir}/${split}_${rank}_${shard}.km`


Finally, merge shards for `${split}` by running
```sh
for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km
```

## Count clusters
Open cluster_counts.py and replace unit_pth to `$lab_dir/${split}.km`.\
Then you can get `$lab_dir/${split}.cluster_counts`.
