import sys
import os

s2s_tsv = sys.argv[1]
src_audio_dir = sys.argv[2]
tgt_audio_dir = sys.argv[3]
out_dir = sys.argv[4]

os.makedirs(out_dir, exist_ok=True)

src_file = open(f"{out_dir}/src_wavs", "w")
tgt_file = open(f"{out_dir}/tgt_wavs", "w")

itr = 0
with open(s2s_tsv) as f:
    next(f)
    for line in f:
        itr += 1
        # if itr % 10000 == 0:
        #     break
        splits = line.rstrip().split("\t")
        src_id = splits[0]
        tgt_id = splits[-1]
        uttid = f"{src_id}__{tgt_id}"
        src_audio = f"{src_audio_dir}/{src_id[:4]}/{src_id}.ogg"
        tgt_audio = f"{tgt_audio_dir}/{tgt_id[:4]}/{tgt_id}.ogg"
        print(f"{uttid} {src_audio}", file=src_file)
        print(f"{uttid} {tgt_audio}", file=tgt_file)
