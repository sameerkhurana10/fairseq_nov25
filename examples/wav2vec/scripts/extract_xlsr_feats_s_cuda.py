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

feat_dir = sys.argv[1]
checkpoint = sys.argv[2]

class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer=None):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model = model.w2v_encoder
        model.eval()
        model.cuda()
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
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source)
            audio_embed = m_res["encoder_out"].squeeze(1).t()
            audio_embed = F.adaptive_avg_pool1d(audio_embed, 1)
            audio_embed = F.normalize(audio_embed, p=2).squeeze()
            return audio_embed.cpu().numpy()

os.makedirs(feat_dir, exist_ok=True)

feat_path = f"{feat_dir}/xlsr.scp"
ark_path = f"{feat_dir}/xlsr.ark"

writer = WriteHelper('ark,scp:{},{}'.format(ark_path, feat_path))
w2v_reader = Wav2VecFeatureReader(checkpoint)
line_indx = 0
for wav_file in sys.stdin:
    wav_file = wav_file.rstrip()
    line_indx += 1
    if line_indx % 10 == 0:
        print("Done %d" % line_indx)
    if line_indx == 10000:
        break
    w2v_feats = w2v_reader.get_feats(wav_file)
    writer(str(line_indx), w2v_feats)