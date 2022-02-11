import sys
import soundfile as sf

wav_scp = sys.argv[1]

with open(wav_scp) as f:
    for line in f:
        splits = line.rstrip().split()
        uttid = splits[0]
        wav = splits[1]
        y, _ = sf.read(wav)
        print(uttid, y.shape[0])
