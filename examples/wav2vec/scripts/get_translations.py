import sys
from jiwer import wer

tgt_tsv = sys.argv[1]
src_tgt_map = sys.argv[2]

uttid_trans = {}
with open(tgt_tsv) as f:
    for line in f:
        splits = line.rstrip().split("\t")
        # remove extension
        audio = splits[0].split(".")[0]
        id = splits[-1]
        uttid = f"{id}-{audio}"
        src = splits[1]
        tgt = splits[2]
        uttid_trans[uttid] = tgt

true = []
pred = []
with open(src_tgt_map) as f:
    for line in f:
        splits = line.rstrip().split()
        src_id = splits[0]
        tgt_id = splits[1]
        true_trans = uttid_trans[src_id]
        retrieved_trans = uttid_trans[tgt_id]
        true.append(true_trans)
        pred.append(retrieved_trans)

print("WER:", 100*wer(true, pred))