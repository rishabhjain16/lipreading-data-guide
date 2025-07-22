import os
import glob
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path
from gen_subword import gen_vocab
from tempfile import NamedTemporaryFile

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LRS2 tsv preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs2', type=str, help='lrs2 root dir')
    parser.add_argument('--vocab-size', type=int, default=1000, help='vocabulary size')
    args = parser.parse_args()

    file_list, label_list = f"{args.lrs2}/file.list", f"{args.lrs2}/label.list"
    assert os.path.isfile(file_list), f"{file_list} not exist"
    assert os.path.isfile(label_list), f"{label_list} not exist"
    nframes_audio_file, nframes_video_file = f"{args.lrs2}/nframes.audio", f"{args.lrs2}/nframes.video"
    assert os.path.isfile(nframes_audio_file), f"{nframes_audio_file} not exist"
    assert os.path.isfile(nframes_video_file), f"{nframes_video_file} not exist"

    print(f"Generating sentencepiece units")
    vocab_size = args.vocab_size
    vocab_dir = (Path(f"{args.lrs2}")/f"spm{vocab_size}").absolute()
    vocab_dir.mkdir(exist_ok=True)
    spm_filename_prefix = f"spm_unigram{vocab_size}"
    
    with NamedTemporaryFile(mode="w") as f:
        label_text = [ln.strip() for ln in open(label_list).readlines()]
        for t in label_text:
            f.write(t.lower() + "\n")
        gen_vocab(Path(f.name), vocab_dir/spm_filename_prefix, 'unigram', args.vocab_size)
    vocab_path = (vocab_dir/spm_filename_prefix).as_posix()+'.txt'

    def setup_target(target_dir, train, valid, test):
        os.makedirs(target_dir, exist_ok=True)
        for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, _, nf_audio, nf_video in data:
                    # Determine if the file is in 'main' or 'pretrain'
                    if 'pretrain' in fid:
                        prefix = 'pretrain'
                    else:
                        prefix = 'main'
                    clean_fid = fid.replace('main/', '').replace('pretrain/', '')
                    fo.write('\t'.join([
                        clean_fid,
                        os.path.abspath(f"{args.lrs2}/{prefix}/{clean_fid}.mp4"),
                        os.path.abspath(f"{args.lrs2}/{prefix}/{clean_fid}.wav"),
                        str(nf_video),
                        str(nf_audio)
                    ])+'\n')
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")

    # Read all data
    fids = [x.strip() for x in open(file_list).readlines()]
    labels = [x.strip().lower() for x in open(label_list).readlines()]
    nfs_audio = [x.strip() for x in open(nframes_audio_file).readlines()]
    nfs_video = [x.strip() for x in open(nframes_video_file).readlines()]

    # Read only test and val splits
    valid_ids = set(line.strip().split()[0] for line in open(f"{args.lrs2}/val.txt"))
    test_ids = set(line.strip().split()[0] for line in open(f"{args.lrs2}/test.txt"))

    train, valid, test = [], [], []
    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        clean_fid = fid.replace('main/', '').replace('pretrain/', '')
        data_item = [fid, label, nf_audio, nf_video]
        
        if clean_fid in test_ids:
            test.append(data_item)
        elif clean_fid in valid_ids:
            valid.append(data_item)
        else:
            train.append(data_item)

    output_dir = f"{args.lrs2}/data_lrs2"
    print("Setting up processed data directory")
    setup_target(output_dir, train, valid, test)

if __name__ == '__main__':
    main()