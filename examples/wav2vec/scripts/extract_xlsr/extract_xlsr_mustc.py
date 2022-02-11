import os
import sys
import tqdm
import torch
import torch.nn.functional as F

import fairseq
import soundfile as sf
import kaldiio
from kaldiio import WriteHelper
import numpy as np
import os.path as osp

tsv = sys.argv[1]
split = sys.argv[2]
nshard = int(sys.argv[3])
rank = int(sys.argv[4])
feat_dir = sys.argv[5]
checkpoint = sys.argv[6]

class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer=None):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model = model.w2v_encoder
        model.eval()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            if len(x.shape) > 1:
                # take audio from one channel
                x = np.mean(x, axis=-1)
            source = torch.from_numpy(x).float()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source)
            audio_embed = m_res["encoder_out"].squeeze(1).t()
            audio_embed = F.adaptive_avg_pool1d(audio_embed, 1).t()
            audio_embed = F.normalize(audio_embed, p=2).squeeze()
            return audio_embed

def get_shard_range(tot, nshard, rank):
    print(rank, nshard)
    assert rank < nshard and rank >= 0, f"invalid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    print(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end

def get_iterator(tsv, nshard, rank):
    with open(tsv, "r") as fp:
        lines = fp.read().split("\n")
        wavs = [line.split('\t')[0] for line in lines if len(line) > 0]
        start, end = get_shard_range(len(lines), nshard, rank)
        wavs = wavs[start:end]
        num = len(wavs)
        reader = Wav2VecFeatureReader(checkpoint)

        def iterate():
            for fname in wavs:
                uttid = fname.split('/')[-1].replace('.flac', '')
                w2v_feats = reader.get_feats(fname)
                yield uttid, w2v_feats
    return iterate, num

generator, num = get_iterator(tsv, nshard, rank)
iterator = generator()

os.makedirs(feat_dir, exist_ok=True)

feat_path = f"{feat_dir}/xlsr_{split}_{rank}_{nshard}.scp"
ark_path = f"{feat_dir}/xlsr_{split}_{rank}_{nshard}.ark"

writer = WriteHelper('ark,scp:{},{}'.format(ark_path, feat_path))

for uttid, w2v_feats in tqdm.tqdm(iterator, total=num):
    writer(uttid, w2v_feats.numpy())




