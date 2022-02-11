# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import AudioVisualDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.data.audio.multi_modality_dataset import ModalityDatasetItem, MultiModalityDataset

from . import FairseqTask, register_task


logger = logging.getLogger(__name__)


@dataclass
class InferredW2vConfig:
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward.
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")


@dataclass
class AudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )

    enable_padding: bool = field(
        default=True, metadata={"help": "pad shorter samples instead of cropping"}
    )

    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )

    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )

    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )

    inferred_w2v_config: Optional[InferredW2vConfig] = field(
        default=None,
        metadata={
            "help": "wav2vec 2.0 masking arguments used to pre-compute masks (required for TPU)",
        },
    )

    tpu: bool = II("common.tpu")
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
                    "target texts): none/low/high (default: none). "
        }
    )


@register_task("audio_visual", dataclass=AudioPretrainingConfig)
class AudioVisualTask(FairseqTask):
    """ """

    cfg: AudioPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices or self.cfg.tpu:
            assert (
                cfg.inferred_w2v_config is not None
            ), "inferred_w2v_config must be set"
            return OmegaConf.to_container(
                cfg.inferred_w2v_config, resolve=True, enum_to_str=True
            )
        else:
            return {}

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        self.datasets[split] = AudioVisualDataset(
            manifest_path=manifest_path,
            sample_rate=None,
            max_sample_size=None,
            min_sample_size=None,
            pad=None,
            normalize=None,
            compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            **self._get_mask_precompute_kwargs(task_cfg),
        )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            # if "w2v_args" in actualized_cfg:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model