# Below is ported from open-muse. It assumes we are using finegrained dino image encoder

# This script is used to pre encode coyo, laion 6a, and laion 5a.
#
# It can be run as both a standalone job or via slurm. When run via slurm, be
# sure to pass `--slurm` so the script can split shards amongst workers based on
# the env vars `$SLURM_NTASKS` and `$SLURM_PROCID`. It is intended that one copy
# of the script is launched per gpu, and cpu access is controlled implicitly
# through slurm setting `$CUDA_VISIBLE_DEVICES`. See
# ../slurm_scrips/{pre_encoded_laion_6, pre_encode_laion_5,
# pre_encode_coyo}.slurm for example sbatch scripts.
#
# Benchmarks:
# COYO) 64.1 GPU * sec / shard
# laion) 75 GPU * sec / shard
#
# To convert a time per shard into a time to convert the
# whole dataset, use
# X (GPU * sec / shard) * Y shards * 1/8 (nodes/GPU) * 1/Z nodes = seconds to encode Y shards
#
# Shard counts:
# COYO) 74,752 shards (0-74,751)
# laion 6a) 1,211 shards (0 - 1,210)
# laion 5a) 60,581 shards (0 - 60,580)
#
# Encoding times using 8 nodes:
# COYO) 20h48m
# laion 6a) 23.4 minutes
# laion 5a) 19h43m

import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Callable
import os
import datasets
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from jaxtyping import Float
from torch import (
    tensor,
    Tensor,
    cat,
    device as Device,
    dtype as DType,
    randn,
    zeros_like,
    exp,
    ones_like,
    stack,
    randn_like,
    no_grad,
    randint,
    float32,
    zeros,
    ones,
    norm,
    multinomial
)
from torch.cuda import empty_cache
from torch.distributions import Beta
from torch.nn import Module, Linear, Embedding, LayerNorm
from torch.nn.init import trunc_normal_
from torch.nn.functional import mse_loss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, Resize
from refiners.foundationals.dinov2 import (
    DINOv2_small,
    DINOv2_small_reg,
    DINOv2_base,
    DINOv2_base_reg,
    DINOv2_large,
    DINOv2_large_reg,
    ViT,
)
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.fluxion.utils import image_to_tensor, normalize

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion.stable_diffusion_1.image_prompt import SD1IPAdapter, get_sd1_image_proj
from refiners.foundationals.latent_diffusion.solvers.ddpm import DDPM
from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, SD1UNet, StableDiffusion_1
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import TextEncoderWithPoolingL
from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.foundationals.latent_diffusion.image_prompt import ImageProjection, PerceiverResampler
import argparse
import concurrent.futures
import logging
import os
import re
from collections import OrderedDict
from threading import Lock

import numpy as np
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode


torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
finegrained = True

CLIP_TEXT_WITH_PROJECTION = "tests/weights/CLIPLWithProjection.safetensors"
DINO_IMAGE_ENCODER = "tests/weights/dinov2_vitl14_reg4_pretrain.safetensors"
LDA = "tests/weights/lda.safetensors"

CLIP_TEXT_EXT = f"CLIPL.pth"
CLIP_TEXT_POOLED_EXT = f"CLIPLPool.pth"
DINO_IMAGE_ENCODER_EXT = f"dinov2_vitl14_reg4_pretrain.pth"
DINO_IMAGE_ENCODER_NO_NORM_EXT = f"dinov2_vitl14_reg4_pretrain_no_norm.pth"
LDA_EXT = f"sd15_lda.pth"

PHOTO_CONCEPT = "gs://bounty-program-data/photo-concept-bucket-webdataset"
PHOTO_CONCEPT_PREENCODED = "gs://bounty-program-data/photo-concept-bucket-webdataset_preencoded"
logger = logging.getLogger(__name__)

tar_regex = r"\/([^\/]+\.tar)"


def get_tar_file_name(url):
    match = re.search(tar_regex, url)
    assert match is not None, url
    tar_file_name = match.group(1)
    return tar_file_name


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


class Uploads:
    """
    Uploads manages the post encoding steps, both CUDA -> cpu and the s3 upload.

    In order to avoid an expensive cuda sync event of the encode for every batch,
    instead "submit" the entirety of the post processing to a thread pool. Once the
    thread pool is full, we hand over the entirety of the thread pool to the python
    interpreter. This effectively allows multiple encoding batches to execute at once.
    At a 160 batch size, this uses <40 GB VRAM.

    TODO - probably would be better to wait until the thread pool is full and then
    execute just the least recent post processing? This could even be done without a
    thread pool or with a single thread, since it's executing one job at a time. Hmmm.

    The class must manage
    1) the thread pool
    2) the list of pending futures that have been submitted
    3) a list of tar writers to upload results

    For the list of tar writers, we keep at most 5 open at a time. When we need to
    open an additional writer, we close the earliest opened one assuming that we have
    finished writing to it as the archives are read sequentially. This is an assumption
    but 5 is a safe buffer as we realistically will never be writing to more than 2 at a time
    for a reasonably sized thread pool.

    The list of tar writers is managed with a global lock because it opens a sub process and
    iirc Popen is not thread safe. Additionally each tar writer is managed with its own lock
    because writes are not thread safe and can corrupt the archive.
    """

    def __init__(self, skip_upload, upload_to, num_writing_threads):
        self.open_lock = Lock()
        self.uploads = OrderedDict()
        self.skip_upload = skip_upload
        self.upload_to = upload_to
        self.futures = []
        self.num_writing_threads = num_writing_threads
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_writing_threads)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finish all pending encodings
        [x.result() for x in concurrent.futures.as_completed(self.futures)]

        self.executor.shutdown(wait=True)

        # Close all unclosed file writes
        for tar_file_name, tar_writer in self.uploads.items():
            tar_writer["writer"].close()

        return False

    def submit(
        self,
        __key__,
        __url__,
        encoder_hidden_states,
        pooled_text_embedding,
        encoded_image_dino,
        encoded_image_dino_no_norm,
        encoded_image_lda,
        metadata,
    ):
        future = self.executor.submit(
            self._upload_thread_entrypoint,
            __key__,
            __url__,
            encoder_hidden_states,
            pooled_text_embedding,
            encoded_image_dino,
            encoded_image_dino_no_norm,
            encoded_image_lda,
            metadata,
        )

        self.futures.append(future)

        # Give cuda some time to complete the encodings before moving to cpu and uploading
        if len(self.futures) == self.num_writing_threads:
            [x.result() for x in concurrent.futures.as_completed(self.futures)]
            self.futures = []

    def _upload_thread_entrypoint(
        self,
        __key__,
        __url__,
        encoder_hidden_states,
        pooled_text_embeddings,
        encoded_image_dinos,
        encoded_image_dino_no_norms,
        encoded_image_ldas,
        metadata,
    ):
        encoder_hidden_states = torch.unbind(encoder_hidden_states)
        pooled_text_embeddings = torch.unbind(pooled_text_embeddings)
        encoded_image_dinos = torch.unbind(encoded_image_dinos)
        encoded_image_dino_no_norms = torch.unbind(encoded_image_dino_no_norms)
        encoded_image_ldas = torch.unbind(encoded_image_ldas)

        for (
            __key__,
            __url__,
            encoded_image_dino,
            encoded_image_dino_no_norm,
            encoded_image_lda,
            encoder_hidden_state,
            pooled_text_embedding,
            metadata,
        ) in zip(
            __key__,
            __url__,
            encoded_image_dinos,
            encoded_image_dino_no_norms,
            encoded_image_ldas,
            encoder_hidden_states,
            pooled_text_embeddings,
            metadata,
        ):
            encoded_image_dino = encoded_image_dino.clone().to("cpu")
            encoded_image_dino_no_norm = encoded_image_dino_no_norm.clone().to("cpu")
            encoded_image_lda = encoded_image_lda.clone().to("cpu")
            encoder_hidden_state = encoder_hidden_state.clone().to("cpu")
            pooled_text_embedding = pooled_text_embedding.clone().to("cpu")

            if self.skip_upload:
                continue

            tar_file_name = get_tar_file_name(__url__)

            # It is not strictly clear to me if it is necessary to lock this whole block or
            # just part(s) of the kickout/create new writer. Just lock the whole function to be
            # safe.
            self.open_lock.acquire()

            if tar_file_name not in self.uploads:
                if len(self.uploads) == 5:
                    # kick out the earliest one
                    key = next(iter(self.uploads.keys()))
                    self.uploads[key]["writer"].close()
                    del self.uploads[key]

                upload_command = f"pipe:gsutil cp - {self.upload_to}/{tar_file_name}"
                logger.warning(f"opening new writer for {upload_command}")

                self.uploads[tar_file_name] = {
                    "writer": wds.TarWriter(upload_command),
                    "lock": Lock(),
                }

            upload = self.uploads[tar_file_name]

            self.open_lock.release()

            metadata = dict(metadata)

            sample = {
                "__key__": __key__,
                DINO_IMAGE_ENCODER_EXT: encoded_image_dino,
                DINO_IMAGE_ENCODER_NO_NORM_EXT: encoded_image_dino_no_norm,
                LDA_EXT: encoded_image_lda,
                CLIP_TEXT_EXT: encoder_hidden_state,
                CLIP_TEXT_POOLED_EXT: pooled_text_embedding,
                "json": metadata,
            }

            # Not locking around the write will corrupt the tar file
            upload["lock"].acquire()
            upload["writer"].write(sample)
            upload["lock"].release()


def distribute_shards(start_shard_all, end_shard_all, slurm_ntasks):
    total_shards = end_shard_all - start_shard_all + 1
    shards_per_task = total_shards // slurm_ntasks
    shards_per_task = [shards_per_task] * slurm_ntasks

    # to distribute the remainder of tasks for non-evenly divisible number of shards
    left_over_shards = total_shards % slurm_ntasks

    for slurm_procid in range(left_over_shards):
        shards_per_task[slurm_procid] += 1

    assert sum(shards_per_task) == total_shards

    distributed_shards = []

    for slurm_procid in range(len(shards_per_task)):
        if slurm_procid == 0:
            start_shard = start_shard_all
        else:
            start_shard = distributed_shards[slurm_procid - 1][1] + 1

        end_shard = start_shard + shards_per_task[slurm_procid] - 1
        distributed_shards.append((start_shard, end_shard))

    assert sum([end_shard - start_shard + 1 for start_shard, end_shard in distributed_shards]) == total_shards

    return distributed_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to pre-encode",
        choices=["photo_concept"],
        required=True,
    )
    parser.add_argument(
        "--start_shard",
        type=int,
        help="The starting shard to pre-encode.",
        required=True,
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        help="The ending shard to pre-encode, inclusive. If not given, defaults to `--start_shard`.",
        required=False,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "If set, this process is running under a batch of slurm tasks."
            "`--start_shard` and `--end_shard` must be set for the entirety of shards over all slurm tasks."
            " The shards that will be encoded in each instance of the task will be determined via"
            " the env vars `$SLURM_NTASKS` and `$SLURM_PROCID`."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, help="The batch size to encode at a time", required=False, default=160
    )
    parser.add_argument(
        "--resolution", type=int, help="The resolution to convert the image to.", required=False, default=518
    )
    parser.add_argument(
        "--lda_resolution", type=int, help="The resolution to convert the image to.", required=False, default=512
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Set to not actually upload results, helpful for only testing encoding.",
    )
    parser.add_argument(
        "--num_writing_threads",
        type=int,
        required=False,
        default=40,
    )

    args = parser.parse_args()

    if args.slurm and args.end_shard is None:
        raise ValueError("`--end_shard` must be set when `--slurm` is set")

    if args.end_shard is None:
        args.end_shard = args.start_shard

    if args.end_shard < args.start_shard:
        raise ValueError("`--end_shard` must be >= `--start_shard`")

    if args.batch_size < 1:
        raise ValueError("`--batch_size` must be >= 1")

    if args.resolution < 1:
        raise ValueError("`--resolution` must be >= 1")

    if args.dataset == "photo_concept":
        args.dataset = PHOTO_CONCEPT
        upload_to = PHOTO_CONCEPT_PREENCODED
    else:
        assert False


    logger.warning("********************")
    logger.warning("Pre-encoding dataset")
    logger.warning(f"dataset: {args.dataset}")
    logger.warning(f"start_shard: {args.start_shard}")
    logger.warning(f"end_shard: {args.end_shard}")
    logger.warning(f"upload_to: {upload_to}")
    logger.warning(f"batch_size: {args.batch_size}")
    logger.warning("********************")

    if args.slurm:
        slurm_procid = int(os.environ["SLURM_PROCID"])
        slurm_ntasks = int(os.environ["SLURM_NTASKS"])

        distributed_shards = distribute_shards(args.start_shard, args.end_shard, slurm_ntasks)

        start_shard_task, end_shard_task = distributed_shards[slurm_procid]

        args.start_shard = start_shard_task
        args.end_shard = end_shard_task

        logger.warning("************")
        logger.warning("Running as slurm task")
        logger.warning(f"SLURM_NTASKS: {slurm_ntasks}")
        logger.warning(f"SLURM_PROCID: {slurm_procid}")
        logger.warning(f"start_shard: {start_shard_task}, end_shard: {end_shard_task}")
        logger.warning("************")
        logger.warning(f"all slurm processes")
        for slurm_proc_id_, (start_shard, end_shard) in enumerate(distributed_shards):
            logger.warning(f"slurm process: {slurm_proc_id_}, start_shard: {start_shard}, end_shard: {end_shard}")
        logger.warning("************")
    lda = SD1Autoencoder(
        device="cuda",
    )
    lda.requires_grad_(False)
    lda.load_from_safetensors(LDA)

    text_encoder = CLIPTextEncoderL(
        device="cuda",
    )
    text_encoder = TextEncoderWithPoolingL(target=text_encoder)
    text_encoder.inject()
    text_encoder.requires_grad_(False)
    text_encoder.load_from_safetensors(CLIP_TEXT_WITH_PROJECTION)

    image_encoder_no_norm = DINOv2_large_reg().to("cuda")
    image_encoder_no_norm.requires_grad_(False)
    image_encoder_no_norm.load_from_safetensors(DINO_IMAGE_ENCODER)
    # remove final layernorm for finegrained
    image_encoder_no_norm.pop()

    image_encoder = DINOv2_large_reg().to("cuda")
    image_encoder.requires_grad_(False)
    image_encoder.load_from_safetensors(DINO_IMAGE_ENCODER)

    shard_range = "{" + format_shard_number(args.start_shard) + ".." + format_shard_number(args.end_shard) + "}"
    download_shards = f"pipe:gsutil cp {args.dataset}/{shard_range}.tar -"

    logger.warning(f"downloading shards {download_shards}")

    src = (
        wds.WebDataset(
            download_shards,
        )
        .decode("pil", handler=wds.warn_and_continue)
        .rename(image="jpg;png;jpeg;webp", prompt="text;txt;caption", metadata="json")
        .map(
            lambda dict: {
                "__key__": dict["__key__"],
                "__url__": dict["__url__"],
                "image": dict["image"],
                "prompt": dict["prompt"],
                "metadata": dict["metadata"],
            }
        )
        .to_tuple("__key__", "__url__", "image", "prompt", "metadata")
        .batched(args.batch_size)
    )
    src = DataLoader(
        src,
        batch_size=None,
        shuffle=False,
        num_workers=0,
    )

    with Uploads(args.skip_upload, upload_to, args.num_writing_threads) as uploads:
        for __key__, __url__, image, prompt, metadata in src:
            logger.warning(f"Encoding {len(__key__)} examples: {__key__[0]} to {__key__[-1]}.")

            encoder_hidden_states, pooled_text_embedding = text_encoder(prompt)

            all_images = []
            lda_images = []


            for image_ in image:
                # The following is minorly more efficient than the default
                # torchvision to_tensor and lets use move to cuda earlier :P
                mode = image_.mode
                if mode != "RGB":
                    image_ = image_.convert("RGB")

                height = image_.height
                width = image_.width

                if hasattr(image_, "getbands"):
                    channels = len(image_.getbands())
                else:
                    channels = image_.channels

                nptype = np.uint8

                image_ = np.array(image_, nptype)
                image_ = torch.from_numpy(image_)
                image_: torch.Tensor = image_.to("cuda")

                image_ = image_.view(height, width, channels)
                image_ = image_.permute((2, 0, 1)).contiguous()

                image_ = image_.to(dtype=torch.float32).div(255)

                lda_image_ = TF.resize(
                    image_, size=args.lda_resolution, interpolation=InterpolationMode.BILINEAR, antialias=True
                )

                lda_image_ = TF.center_crop(image_, args.lda_resolution)
                lda_images.append(2*lda_image_-1)
                image_ = TF.resize(
                    image_, size=args.resolution, interpolation=InterpolationMode.BILINEAR, antialias=True
                )

                image_ = TF.center_crop(image_, args.resolution)
                image_ = normalize(image_, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                all_images.append(image_)

            image = torch.stack(all_images)
            lda_image = torch.stack(lda_images)
            encoded_image_dino = image_encoder(image)
            encoded_image_dino_no_norm = image_encoder_no_norm(image)
            encoded_image_lda = lda.encode(lda_image)


            uploads.submit(
                __key__,
                __url__,
                encoder_hidden_states,
                pooled_text_embedding,
                encoded_image_dino,
                encoded_image_dino_no_norm,
                encoded_image_lda,
                metadata,
            )


if __name__ == "__main__":
    main()