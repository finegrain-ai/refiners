from refiners.training_utils.latent_diffusion import FinetuneLatentDiffusionConfig, LatentDiffusionTrainer
from torch import device as Device
from warnings import warn
import pytest

DEFAULT_LATENT_DICT = dict(
    script = "foo.py",
    wandb = dict(
        mode = "offline",
        entity = "acme",
        project = "test-ldm-training"
    ),
    latent_diffusion = dict(
        unconditional_sampling_probability = 0.2,
        offset_noise = 0.1
    ),
    optimizer = dict(
        optimizer = "AdamW",
        learning_rate = 1e-5,
        betas = [0.9, 0.999],
        eps = 1e-8,
        weight_decay = 1e-2
    ),
    scheduler = dict(),
    dropout = dict(dropout_probability = 0.2),
    checkpointing=dict(save_interval = "1:epoch"),
    test_diffusion=dict(prompts = [
        "A cute cat",
    ]),
    models = dict(
        lda = dict(
            checkpoint = "tests/weights/stable-diffusion-1-5/lda.safetensors",
            train = False,
            gpu_index= 0
        ),
        text_encoder = dict(
            checkpoint = "tests/weights/stable-diffusion-1-5/CLIPTextEncoderL.safetensors",
            train = True,
            gpu_index= 0
        ),
        unet= dict(
            checkpoint = "tests/weights/stable-diffusion-1-5/unet.safetensors",
            train = False,
            gpu_index= 1
        ),
    ),
    training = dict(
        duration= "1:epoch",
        gpu_index= 0
    ),
    dataset = dict(
        hf_repo= "1aurent/unsplash-lite-palette",
        revision= "main",
        caption_key = "ai_description"
    )
)

from lightning import Fabric
from refiners.foundationals.latent_diffusion import (
    SD1UNet
)
from lightning.fabric.strategies import FSDPStrategy
from accelerate import Accelerator, DistributedType

def test_ldm_trainer_text_encoder_on_two_devices(test_device: Device, test_second_device: Device):
    
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()
        
    if test_second_device.type == "cpu":
        warn("Running with only one GPU, skipping")
        pytest.skip()
    
    config = FinetuneLatentDiffusionConfig.load_from_dict(
        dict(DEFAULT_LATENT_DICT)
    )

    trainer = LatentDiffusionTrainer(config=config)
    trainer.train()
    
    assert trainer.lda.device == test_device
    assert trainer.text_encoder.device.type == test_second_device