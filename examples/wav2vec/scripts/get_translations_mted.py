import sys
from jiwer import wer

tgt_tsv = sys.argv[1]
src_tgt_map = sys.argv[2]

tgt_tsvs = tgt_tsv.split(',')

uttid_trans = {}
tgt_tot = 0
for tgt_tsv in tgt_tsvs:
    with open(tgt_tsv) as f:
        next(f)
        for line in f:
            tgt_tot+=1
            splits = line.rstrip().split("\t")
            # remove extension
            uttid = splits[0]
            src = splits[1]
            tgt = splits[2]
            uttid_trans[uttid] = tgt

true = []
pred = []
src_tot = 0
with open(src_tgt_map) as f:
    for line in f:
        src_tot+=1
        splits = line.rstrip().split()
        src_id = splits[0]
        tgt_id = splits[1]
        true_trans = uttid_trans[src_id]
        retrieved_trans = uttid_trans[tgt_id]
        if true_trans.strip() and retrieved_trans.strip():
            true.append(true_trans)
            pred.append(retrieved_trans)

print(len(true))
print(len(pred))
print(f"{src_tot}/{tgt_tot}")

print("WER:", 100*wer(true, pred))