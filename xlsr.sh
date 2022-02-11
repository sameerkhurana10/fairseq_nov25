#!/bin/bash

tsv=$1
split=$2
nj=$3
JOB=$4
feats=$5
ckpt=$6

module load libsndfile

/gpfswork/rech/iqh/upp27cx/conda-envs/fair_nov25/bin/python examples/wav2vec/scripts/extract_xlsr/extract_xlsr_mustc.py ${tsv} ${split} $nj $JOB ${feats} ${ckpt}
