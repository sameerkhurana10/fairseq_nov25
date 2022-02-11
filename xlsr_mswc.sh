#!/bin/bash

module load libsndfile

wav_scp=$1
wrd2utt=$2
ckpt=$3
nshard=$4
rank=$5
root=$6
feats=$7

python examples/wav2vec/scripts/extract_xlsr/extract_xlsr_mswc.py ${wav_scp} ${wrd2utt} ${ckpt} ${nshard} ${rank} ${root} ${feats}
