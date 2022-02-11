import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from .raw_audio_dataset import RawAudioDataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

logger = logging.getLogger(__name__)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px=32):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class AudioVisualDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        text_compression_level=TextCompressionLevel.none,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.fnames = []
        self.fnames_img = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                self.fnames.append(self.text_compressor.compress(items[0]))
                self.fnames_img.append(self.text_compressor.compress(items[-1]))
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        import soundfile as sf
        import kaldiio
        fn = self.fnames[index]
        fn = self.text_compressor.decompress(fn)
        audio_embedding = kaldiio.load_mat(fn).copy()
        fn_img = self.fnames_img[index]
        fn_img = self.text_compressor.decompress(fn_img)
        img_preprocess = _transform()
        feats_img = img_preprocess(fn_img)
        feats = torch.from_numpy(audio_embedding).float()
        return {"id": index, "source": feats, "feats_img": feats_img}

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}
        collated_sources = self.get_collated(samples, "source")
        collated_feats_labse, _ = self.get_collated(samples, "feats_img")
        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        out["net_input"] = input
        out["text_embedding"] = collated_feats_labse
        return out

    def get_collated(self, samples, key):
        sources = [s[key] for s in samples]
        return torch.stack(sources, dim=0)