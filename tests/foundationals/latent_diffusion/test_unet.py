from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.fluxion import manual_seed
import torch


def test_unet_context_flush():
    manual_seed(0)
    text_embedding = torch.randn(1, 77, 768)
    timestep = torch.randint(0, 999, size=(1, 1))
    x = torch.randn(1, 4, 32, 32)

    unet = SD1UNet(in_channels=4, clip_embedding_dim=768)
    unet.set_clip_text_embedding(clip_text_embedding=text_embedding)  # not flushed between forward-s

    with torch.no_grad():
        unet.set_timestep(timestep=timestep)
        y_1 = unet(x.clone())

    with torch.no_grad():
        unet.set_timestep(timestep=timestep)
        y_2 = unet(x.clone())

    assert torch.equal(y_1, y_2)
