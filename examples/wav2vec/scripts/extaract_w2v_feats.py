#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F

import fairseq
import soundfile as sf
import kaldiio


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec ctc model', required=True)
    parser.add_argument('--layer', type=int, default=None, help='which layer to use')
    # fmt: on

    return parser


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

    def get_feats(self, loc, labse_loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            text_embed = kaldiio.load_mat(labse_loc).copy()
            text_embed = torch.from_numpy(text_embed).float()[1]
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source)
            audio_embed = m_res["encoder_out"].squeeze(1).t()
            audio_embed = F.adaptive_avg_pool1d(audio_embed, 1)
            audio_embed = F.normalize(audio_embed, p=2).squeeze()
            return audio_embed.cpu(), text_embed


def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]
        files_labse = [line.split("\t")[-1] for line in lines if len(line) > 0]

        num = len(files)
        reader = Wav2VecFeatureReader(args.checkpoint)

        def iterate():
            for i, fname in enumerate(files):
                w2v_feats, labse_feats = reader.get_feats(fname, files_labse[i])
                yield w2v_feats, labse_feats

    return iterate, num

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_recalls(S):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images and columns are captions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def main():
    parser = get_parser()
    args = parser.parse_args()

    generator, num = get_iterator(args)
    iterator = generator()

    audio_embeds = []
    text_embeds = []
    for audio_embed, text_embed in tqdm.tqdm(iterator, total=num):
        audio_embeds.append(audio_embed)
        text_embeds.append(text_embed)
        if len(audio_embeds) == 1000:
            break

    audio_embeds = torch.stack(audio_embeds, dim=0)
    text_embeds = torch.stack(text_embeds, dim=0)
    S = torch.mm(text_embeds, audio_embeds.t())
    recalls = calc_recalls(S)
    print(recalls["A_r10"], recalls["I_r10"])
    print(recalls["A_r5"], recalls["I_r5"])
    print(recalls["A_r1"], recalls["I_r1"])


if __name__ == "__main__":
    main()
