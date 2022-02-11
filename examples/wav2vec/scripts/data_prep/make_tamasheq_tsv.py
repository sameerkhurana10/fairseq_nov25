import sys

wav_scp = sys.argv[1]
utt2num_samples = sys.argv[2]
labse_scp = sys.argv[3]

utt2wav = {}
with open(wav_scp) as f:
    for line in f:
        splits = line.rstrip().split()
        utt2wav[splits[0]] = splits[1]

utt2samp = {}
with open(utt2num_samples) as f:
    for line in f:
        splits = line.rstrip().split()
        utt2samp[splits[0]] = splits[1]

utt2labse = {}
with open(labse_scp) as f:
    for line in f:
        splits = line.rstrip().split()
        utt2labse[splits[0]] = splits[1]

for k, v in utt2labse.items():
    labse = v
    wav = utt2wav[k]
    sampl = utt2samp[k]
    print(f"{wav}\t{sampl}\t{labse}")