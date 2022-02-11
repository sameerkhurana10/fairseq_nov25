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
import kaldiio
import numpy as np
import sys

labse_scp = sys.argv[1]
out_dir = sys.argv[2]

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
    os.makedirs(out_dir, exist_ok=True)
    out_r10 = open(f"{out_dir}/{labse_scp.split('/')[-1]}.r10", "w")
    out_r5 = open(f"{out_dir}/{labse_scp.split('/')[-1]}.r5", "w")
    out_r1 = open(f"{out_dir}/{labse_scp.split('/')[-1]}.r1", "w")
    audio_embeds = []
    text_embeds = []
    num = 0
    with open(labse_scp) as f:
        for i, line in enumerate(f):
            num += 1
            if (i+1) % 100 == 0:
                print(f"Done {i}")
            splts = line.rstrip().split()
            embeddings = torch.tensor(kaldiio.load_mat(splts[1]))
            audio_embeds.append(embeddings[0])
            text_embeds.append(embeddings[1])
            # if len(audio_embeds) == 3000:
            #     break

    audio_embeds = torch.stack(audio_embeds, dim=0)
    text_embeds = torch.stack(text_embeds, dim=0)
    S = torch.mm(text_embeds, audio_embeds.t())
    recalls = calc_recalls(S)
    print(recalls["A_r10"], recalls["I_r10"], num, file=out_r10)
    print(recalls["A_r5"], recalls["I_r5"], num, file=out_r5)
    print(recalls["A_r1"], recalls["I_r1"], num, file=out_r1)


if __name__ == "__main__":
    main()
