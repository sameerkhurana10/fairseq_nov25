# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


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
    A2I_scores, A2I_ind = S.topk(1, 0)
    I2A_scores, I2A_ind = S.topk(1, 1)
    A_r1 = AverageMeter()
    # A_r5 = AverageMeter()
    # A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    # I_r5 = AverageMeter()
    # I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(1):
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
        # if A_foundind >= 0 and A_foundind < 5:
        #     A_r5.update(1)
        # else:
        #     A_r5.update(0)
        # if I_foundind >= 0 and I_foundind < 5:
        #     I_r5.update(1)
        # else:
        #     I_r5.update(0)
        # # do r10s
        # if A_foundind >= 0 and A_foundind < 10:
        #     A_r10.update(1)
        # else:
        #     A_r10.update(0)
        # if I_foundind >= 0 and I_foundind < 10:
        #     I_r10.update(1)
        # else:
        #     I_r10.update(0)

    # recalls = {'A_r1':A_r1, 'A_r5':A_r5, 'A_r10':A_r10,
    #             'I_r1':I_r1, 'I_r5':I_r5, 'I_r10':I_r10}
    recalls = {'A_r1': A_r1, 'I_r1':I_r1}
    return recalls


@dataclass
class STCriterionConfig(FairseqDataclass):
    loss_type: str = field(
        default="l1_loss",
    )


@register_criterion("st_crit_tanh", dataclass=STCriterionConfig)
class STCriterion(FairseqCriterion):
    def __init__(self, cfg: STCriterionConfig, task: FairseqTask):
        super().__init__(task)

    def avg_pool(self, audio_outputs, nframes):
        """
        Assumes text_embeddings is a (batchsize, embedding_dim) tensor
        Assumes audio_outputs is a (batchsize, embedding_dim, time) tensor
        Returns similarity matrix S where images are rows and audios are along the columns
        S[i][j] is computed as the dot product between the meanpooled embeddings of
        the ith image output and jth audio output
        """
        assert (audio_outputs.dim() == 3)
        audio_pool = torch.nn.AdaptiveAvgPool1d(1)
        pooled_audio_outputs_list = []
        for idx in range(len(audio_outputs)):
            nF = max(1, nframes[idx])
            pooled_audio_outputs_list.append(audio_pool(audio_outputs[idx][:, 0:nF]).unsqueeze(0))
        pooled_audio_outputs = torch.cat(pooled_audio_outputs_list).squeeze(2)
        return pooled_audio_outputs

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        enc_out = net_output["encoder_out"]
        # L2 normalize
        audio_embed = F.normalize(torch.tanh(enc_out), p=2, dim=1)
        # for sentences text embeddings are already normalized but for word embedding its not
        text_embed = sample["text_embedding"]
        if text_embed.shape[1] == 2:
            # for ST datasets we have the labse embedding
            # for both source and target
            text_embed = text_embed[:, 1]
        elif text_embed.shape[1] == 1:
            # for ASR datasets we have labse embedding
            # for the normalized transcript only
            text_embed = text_embed[:, 0]
        else:
            raise Exception
        loss = F.l1_loss(audio_embed.float(), text_embed.float(), reduction="sum") * 10.0

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "nsentences": sample["id"].numel(),
        }

        if not model.training:
            with torch.no_grad():
                S = torch.mm(text_embed.float(), audio_embed.float().t())
                recalls = calc_recalls(S)
                # A_r10 = recalls['A_r10']
                # I_r10 = recalls['I_r10']
                # A_r5 = recalls['A_r5']
                # I_r5 = recalls['I_r5']
                A_r1 = recalls['A_r1']
                I_r1 = recalls['I_r1']
                # logging_output["A_R10_sum"] = A_r10.sum
                # logging_output["A_R10_count"] = A_r10.count
                # logging_output["T_R10_sum"] = I_r10.sum
                # logging_output["T_R10_count"] = I_r10.count
                # logging_output["A_R5_sum"] = A_r5.sum
                # logging_output["A_R5_count"] = A_r5.count
                # logging_output["T_R5_sum"] = I_r5.sum
                # logging_output["T_R5_count"] = I_r5.count
                logging_output["A_R1_sum"] = A_r1.sum
                logging_output["A_R1_count"] = A_r1.count
                logging_output["T_R1_sum"] = I_r1.sum
                logging_output["T_R1_count"] = I_r1.count

        return loss, len(audio_embed), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        # A_r10_sum = utils.item(sum(log.get("A_R10_sum", 0) for log in logging_outputs))
        # A_r10_count = utils.item(sum(log.get("A_R10_count", 0) for log in logging_outputs))
        # T_r10_sum = utils.item(sum(log.get("T_R10_sum", 0) for log in logging_outputs))
        # T_r10_count = utils.item(sum(log.get("T_R10_count", 0) for log in logging_outputs))
        # A_r5_sum = utils.item(sum(log.get("A_R5_sum", 0) for log in logging_outputs))
        # A_r5_count = utils.item(sum(log.get("A_R5_count", 0) for log in logging_outputs))
        # T_r5_sum = utils.item(sum(log.get("T_R5_sum", 0) for log in logging_outputs))
        # T_r5_count = utils.item(sum(log.get("T_R5_count", 0) for log in logging_outputs))
        A_r1_sum = utils.item(sum(log.get("A_R1_sum", 0) for log in logging_outputs))
        A_r1_count = utils.item(sum(log.get("A_R1_count", 0) for log in logging_outputs))
        T_r1_sum = utils.item(sum(log.get("T_R1_sum", 0) for log in logging_outputs))
        T_r1_count = utils.item(sum(log.get("T_R1_count", 0) for log in logging_outputs))
        # avg_r10 = 0.
        # avg_r5 = 0.
        avg_r1 = 0.
        if A_r1_sum > 0:
            # A_r10 = A_r10_sum / A_r10_count
            # T_r10 = T_r10_sum / T_r10_count
            # avg_r10 = (A_r10+T_r10)/2.
            # A_r5 = A_r5_sum / A_r5_count
            # T_r5 = T_r5_sum / T_r5_count
            # avg_r5 = (A_r5+T_r5)/2.
            A_r1 = A_r1_sum / A_r1_count
            T_r1 = T_r1_sum / T_r1_count
            avg_r1 = (A_r1 + T_r1) / 2.
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / nsentences, nsentences, round=3
        )
        metrics.log_scalar("nsentences", nsentences)
        # if avg_r10 > 0.:
        #     metrics.log_scalar("avg_r10", avg_r10)
        # if avg_r5 > 0.:
        #     metrics.log_scalar("avg_r5", avg_r5)
        if avg_r1 > 0.:
            metrics.log_scalar("avg_r1", avg_r1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
