import torch

from refiners.fluxion.adapters.lora import Lora, auto_attach_loras
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.model import StableDiffusion_XL


def _check_validity(debug_map: list[tuple[str, str]]):
    # Check things are in the right block.
    prefix_map = {
        "down_blocks_0": ["DownBlocks.Chain_1", "DownBlocks.Chain_2", "DownBlocks.Chain_3", "DownBlocks.Chain_4"],
        "down_blocks_1": ["DownBlocks.Chain_5", "DownBlocks.Chain_6", "DownBlocks.Chain_7"],
        "down_blocks_2": ["DownBlocks.Chain_8", "DownBlocks.Chain_9"],
        "mid_block": ["MiddleBlock"],
        "up_blocks_0": ["UpBlocks.Chain_1", "UpBlocks.Chain_2", "UpBlocks.Chain_3"],
        "up_blocks_1": ["UpBlocks.Chain_4", "UpBlocks.Chain_5", "UpBlocks.Chain_6"],
        "up_blocks_2": ["UpBlocks.Chain_7", "UpBlocks.Chain_8", "UpBlocks.Chain_9"],
    }
    for key, path in debug_map:
        for key_pfx, paths_pfxs in prefix_map.items():
            if key.startswith(f"lora_unet_{key_pfx}"):
                assert any(path.startswith(f"SDXLUNet.{x}") for x in paths_pfxs), f"bad mapping: {key} {path}"


def add_lcm_lora(
    manager: SDLoraManager,
    tensors: dict[str, torch.Tensor],
    name: str = "lcm",
    scale: float = 1.0 / 8.0,
    check_validity: bool = True,
) -> None:
    """Add a LCM LoRA to SDXLUNet.

    This is a complex LoRA so [SDLoraManager.add_loras()][refiners.foundationals.latent_diffusion.lora.SDLoraManager.add_loras]
    is not enough. Instead, we add the LoRAs to the UNet in several iterations, using the filtering mechanism of
    [auto_attach_loras][refiners.fluxion.adapters.lora.auto_attach_loras].

    This LoRA can be used with or without CFG in SD.
    If you use CFG, typical values range from 1.0 (same as no CFG) to 2.0.

    Args:
        manager: A SDLoraManager for SDXL
        tensors: The `state_dict` of the LCM LoRA
        name: The name of the LoRA.
        scale: The scale to use for the LoRA (should generally not be changed).
        check_validity: Perform additional checks, raise an exception if they fail.
    """

    assert isinstance(manager.target, StableDiffusion_XL)
    unet = manager.target.unet

    loras = Lora.from_dict(name, {k: v.to(unet.device, unet.dtype) for k, v in tensors.items()})
    assert all(k.startswith("lora_unet_") for k in loras.keys())
    loras = {k: loras[k] for k in sorted(loras.keys(), key=SDLoraManager.sort_keys)}

    debug_map: list[tuple[str, str]] | None = [] if check_validity else None

    # Projections are in `SDXLCrossAttention` but not in `CrossAttentionBlock`.
    loras_projs = {k: v for k, v in loras.items() if k.endswith("proj_in") or k.endswith("proj_out")}
    auto_attach_loras(
        loras_projs,
        unet,
        exclude=["CrossAttentionBlock"],
        include=["SDXLCrossAttention"],
        debug_map=debug_map,
    )

    # Do *not* check for time because some keys include both `resnets` and `time_emb_proj`.
    exclusions = {
        "res": "ResidualBlock",
        "downsample": "Downsample",
        "upsample": "Upsample",
    }
    loras_excluded = {k: v for k, v in loras.items() if any(x in k for x in exclusions.keys())}
    loras_remaining = {k: v for k, v in loras.items() if k not in loras_excluded and k not in loras_projs}

    auto_attach_loras(
        loras_remaining,
        unet,
        exclude=[*exclusions.values(), "TimestepEncoder"],
        debug_map=debug_map,
    )

    # Process exclusions one by one to avoid mixing them up.
    for exc, v in exclusions.items():
        ls = {k: v for k, v in loras_excluded.items() if exc in k}
        auto_attach_loras(ls, unet, include=[v], debug_map=debug_map)

    if debug_map is not None:
        _check_validity(debug_map)

    # LoRAs are finally injected, set the scale with the manager.
    manager.set_scale(name, scale)
