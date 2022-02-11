import sys

tsv = sys.argv[1]
trans_map_file = sys.argv[2]
out_results_file = sys.argv[3]

utt_to_tgt_map = {}
utt_to_src_map = {}
with open(tsv) as f:
    next(f)
    for line in f:
        audio, src, tgt, id = line.rstrip().split("\t")
        audio_name = audio.split('.')[0]
        uttid = f"{id}-{audio_name}"
        utt_to_tgt_map[uttid] = tgt
        utt_to_src_map[uttid] = src

# print(utt_to_tgt_map)
hyps = []
refs = []
out_results_file = open(out_results_file, "w")
print("uttid\tsrc_txt\tgt_trans\tretrieved_trans", file=out_results_file)
with open(trans_map_file) as f:
    for line in f:
        splits = line.rstrip().split()
        retrieved_trans_txt = utt_to_tgt_map[splits[1]]
        gt_trans_txt = utt_to_tgt_map[splits[0]]
        gt_src_txt = utt_to_src_map[splits[0]]
        hyps.append(retrieved_trans_txt)
        refs.append(gt_trans_txt)
        print(f"{splits[0]}\t{gt_src_txt}\t{gt_trans_txt}\t{retrieved_trans_txt}", file=out_results_file)

from sacrebleu.metrics import BLEU, CHRF, TER
bleu = BLEU()
chrf = CHRF()
ter = TER()
bleu = bleu.corpus_score(hyps, [refs])
chrf = chrf.corpus_score(hyps, [refs])
ter = ter.corpus_score(hyps, [refs])
print(bleu)
print(chrf)
print(ter)