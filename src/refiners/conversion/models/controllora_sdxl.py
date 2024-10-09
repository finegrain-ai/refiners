import logging

import requests
import torch
from huggingface_hub import hf_hub_download  # type: ignore
from torch import Tensor
from torch.nn import Parameter as TorchParameter

from refiners.conversion.utils import Conversion, Hub
from refiners.fluxion.adapters.lora import Lora, LoraAdapter, auto_attach_loras
from refiners.fluxion.layers import Conv2d
from refiners.fluxion.layers.linear import Linear
from refiners.fluxion.utils import load_from_safetensors, save_to_safetensors
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.control_lora import (
    ConditionEncoder,
    ControlLora,
    ControlLoraAdapter,
    ZeroConvolution,
)
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.model import StableDiffusion_XL


def sort_keys(key: str, /) -> tuple[str, int]:
    """Compute the score of a key, relatively to its suffix.

    When used by [`sorted`][sorted], the keys will only be sorted "at the suffix level".

    Args:
        key: The key to sort.

    Returns:
        The padded suffix of the key.
        The score of the key's suffix.
    """
    if "time_embed" in key:  # HACK: will place the "time_embed" layers at very start of the list
        return ("", -2)

    if "label_emb" in key:  # HACK: will place the "label_emb" layers right after "time_embed"
        return ("", -1)

    if "proj_out" in key:  # HACK: will place the "proj_out" layers at the end of each "transformer_blocks"
        return (key.removesuffix("proj_out") + "transformer_blocks.99.ff.net.2", 10)

    return SDLoraManager.sort_keys(key)


def load_lora_layers(
    name: str,
    state_dict: dict[str, Tensor],
    control_lora: ControlLora,
) -> dict[str, Lora[Linear | Conv2d]]:
    """Load the LoRA layers from the state_dict into the ControlLora.

    Args:
        name: The name of the LoRA.
        state_dict: The state_dict of the LoRA.
        control_lora: The ControlLora to load the LoRA layers into.
    """
    # filter from the state_dict the layers that will be used for the LoRA layers
    lora_weights = {f"{key}.weight": value for key, value in state_dict.items() if ".up" in key or ".down" in key}

    # move the tensors to the device and dtype of the ControlLora
    lora_weights = {
        key: value.to(
            dtype=control_lora.dtype,
            device=control_lora.device,
        )
        for key, value in lora_weights.items()
    }

    # load every LoRA layers from the filtered state_dict
    lora_layers = Lora.from_dict(name, state_dict=lora_weights)

    # sort all the LoRA's keys using the `sort_keys` method
    lora_layers = {
        key: lora_layers[key]
        for key in sorted(
            lora_layers.keys(),
            key=sort_keys,
        )
    }

    # auto-attach the LoRA layers to the U-Net
    auto_attach_loras(lora_layers, control_lora, exclude=["ZeroConvolution", "ConditionEncoder"])

    # eject all the LoRA adapters from the U-Net
    # because we need each target path as if the adapter wasn't injected
    for lora_layer in lora_layers.values():
        lora_adapter = lora_layer.parent
        assert isinstance(lora_adapter, LoraAdapter)
        lora_adapter.eject()

    return lora_layers


def load_condition_encoder(
    state_dict: dict[str, Tensor],
    control_lora: ControlLora,
) -> None:
    """Load the ConditionEncoder's Conv2d layers from the state_dict into the ControlLora.

    Args:
        state_dict: The state_dict of the ConditionEncoder.
        control_lora: The control_lora to load the ConditionEncoder's Conv2d layers into.
    """
    # filter from the state_dict the layers that will be used for the ConditionEncoder
    condition_encoder_tensors = {key: value for key, value in state_dict.items() if "input_hint_block" in key}

    # move the tensors to the device and dtype of the ControlLora
    condition_encoder_tensors = {
        key: value.to(
            dtype=control_lora.dtype,
            device=control_lora.device,
        )
        for key, value in condition_encoder_tensors.items()
    }

    # find the ConditionEncoder's Conv2d layers
    condition_encoder_layer = control_lora.ensure_find(ConditionEncoder)
    condition_encoder_conv2ds = list(condition_encoder_layer.layers(Conv2d))

    # replace the Conv2d layers' weights and biases with the ones from the state_dict
    for i, layer in enumerate(condition_encoder_conv2ds):
        layer.weight = TorchParameter(condition_encoder_tensors[f"input_hint_block.{i*2}.weight"])
        layer.bias = TorchParameter(condition_encoder_tensors[f"input_hint_block.{i*2}.bias"])


def load_zero_convolutions(
    state_dict: dict[str, Tensor],
    control_lora: ControlLora,
) -> None:
    """Load the ZeroConvolution's Conv2d layers from the state_dict into the ControlLora.

    Args:
        state_dict: The state_dict of the ZeroConvolution.
        control_lora: The ControlLora to load the ZeroConvolution's Conv2d layers into.
    """
    # filter from the state_dict the layers that will be used for the ZeroConvolution layers
    zero_convolution_tensors = {key: value for key, value in state_dict.items() if "zero_convs" in key}
    n = len(zero_convolution_tensors) // 2
    zero_convolution_tensors[f"zero_convs.{n}.0.weight"] = state_dict["middle_block_out.0.weight"]
    zero_convolution_tensors[f"zero_convs.{n}.0.bias"] = state_dict["middle_block_out.0.bias"]

    # move the tensors to the device and dtype of the ControlLora
    zero_convolution_tensors = {
        key: value.to(
            dtype=control_lora.dtype,
            device=control_lora.device,
        )
        for key, value in zero_convolution_tensors.items()
    }

    # find the ZeroConvolution's Conv2d layers
    zero_convolution_layers = list(control_lora.layers(ZeroConvolution))
    zero_convolution_conv2ds = [layer.ensure_find(Conv2d) for layer in zero_convolution_layers]

    # replace the Conv2d layers' weights and biases with the ones from the state_dict
    for i, layer in enumerate(zero_convolution_conv2ds):
        layer.weight = TorchParameter(zero_convolution_tensors[f"zero_convs.{i}.0.weight"])
        layer.bias = TorchParameter(zero_convolution_tensors[f"zero_convs.{i}.0.bias"])


def simplify_key(key: str, prefix: str, index: int | None = None) -> str:
    """Simplify a key by stripping everything to the left of the prefix.

    Also optionally add a zero-padded index to the prefix.

    Example:
        >>> simplify_key("foo.bar.ControlLora.something", "ControlLora", 1)
        "ControlLora_01.something"

        >>> simplify_key("foo.bar.ControlLora.DownBlocks.something", "ControlLora")
        "ControlLora.DownBlocks.something"

    Args:
        key: The key to simplify.
        prefix: The prefix to remove.
        index: The index to add.
    """
    _, right = key.split(prefix, maxsplit=1)
    if index:
        return f"{prefix}_{index:02d}{right}"
    else:
        return f"{prefix}{right}"


def convert_lora_layers(
    lora_layers: dict[str, Lora[Linear | Conv2d]],
    control_lora: ControlLora,
    refiners_state_dict: dict[str, Tensor],
) -> None:
    """Convert the LoRA layers to the refiners format.

    Args:
        lora_layers: The LoRA layers to convert.
        control_lora: The ControlLora to convert the LoRA layers from.
        refiners_state_dict: The refiners state dict to update with the converted LoRA layers.
    """
    for lora_layer in lora_layers.values():
        # get the adapter associated with the LoRA layer
        lora_adapter = lora_layer.parent
        assert isinstance(lora_adapter, LoraAdapter)

        # get the path of the adapter's target in the ControlLora
        target = lora_adapter.target
        path = target.get_path(parent=control_lora.ensure_find_parent(target))

        state_dict = {
            f"{path}.down": lora_layer.down.weight,
            f"{path}.up": lora_layer.up.weight,
        }
        state_dict = {simplify_key(key, "ControlLora."): param for key, param in state_dict.items()}
        refiners_state_dict.update(state_dict)


def convert_zero_convolutions(
    control_lora: ControlLora,
    refiners_state_dict: dict[str, Tensor],
) -> None:
    """Convert the ZeroConvolution layers to the refiners format.

    Args:
        control_lora: The ControlLora to convert the ZeroConvolution layers from.
        refiners_state_dict: The refiners state dict to update with the converted ZeroConvolution layers.
    """
    zero_convolution_layers = list(control_lora.layers(ZeroConvolution))
    for i, zero_convolution_layer in enumerate(zero_convolution_layers):
        state_dict = zero_convolution_layer.state_dict()
        path = zero_convolution_layer.get_path()
        state_dict = {f"{path}.{key}": param for key, param in state_dict.items()}
        state_dict = {simplify_key(key, "ZeroConvolution", i + 1): param for key, param in state_dict.items()}
        refiners_state_dict.update(state_dict)


def convert_condition_encoder(
    control_lora: ControlLora,
    refiners_state_dict: dict[str, Tensor],
) -> None:
    """Convert the ConditionEncoder to the refiners format.

    Args:
        control_lora: The ControlLora to convert the ConditionEncoder from.
        refiners_state_dict: The refiners state dict to update with the converted ConditionEncoder.
    """
    condition_encoder_layer = control_lora.ensure_find(ConditionEncoder)
    path = condition_encoder_layer.get_path()
    state_dict = condition_encoder_layer.state_dict()
    state_dict = {f"{path}.{key}": param for key, param in state_dict.items()}
    state_dict = {simplify_key(key, "ConditionEncoder"): param for key, param in state_dict.items()}
    refiners_state_dict.update(state_dict)


class ControlLoraConversion(Conversion):
    def __init__(
        self,
        original: Hub,
        converted: Hub,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the weight structure.

        Args:
            original_weight_hub: A HubPath object representing the original weight.
            converted_weight_hub: A HubPath object representing the converted weight.
        """
        self.original = original
        self.converted = converted
        self.dtype = dtype

    def convert(self) -> None:
        """Convert the weights from the original to the converted weights."""
        logging.info(
            f"Converting {self.original.repo_id}/{self.original.filename} to {self.converted.repo_id}/{self.converted.filename}"
        )

        # check if the converted file already exists
        if self.converted.local_path.is_file():
            logging.warning(f"{self.converted.local_path} already exists")
            if self.converted.check_local_hash():
                try:
                    assert self.converted.check_remote_hash()
                except requests.exceptions.HTTPError:
                    logging.error(f"{self.converted.local_path} couldn't verify remote hash")
                return

        # get the original state_dict
        self.original.download()

        # load the original state_dict
        original_state_dict = load_from_safetensors(self.original.local_path)

        # convert the state_dict
        sdxl = StableDiffusion_XL()
        name = self.original.local_path.stem
        control_lora_adapter = ControlLoraAdapter(target=sdxl.unet, name=name).inject()
        control_lora = control_lora_adapter.control_lora
        lora_layers = load_lora_layers(name, original_state_dict, control_lora)
        load_zero_convolutions(original_state_dict, control_lora)
        load_condition_encoder(original_state_dict, control_lora)

        converted_state_dict: dict[str, Tensor] = {}
        convert_lora_layers(lora_layers, control_lora, converted_state_dict)
        convert_zero_convolutions(control_lora, converted_state_dict)
        convert_condition_encoder(control_lora, converted_state_dict)

        # save the converted state_dict
        self.converted.local_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_safetensors(self.converted.local_path, converted_state_dict)

        # check the converted state_dict
        assert self.converted.check_local_hash()
        try:
            assert self.converted.check_remote_hash()
        except requests.exceptions.HTTPError:
            logging.warning(f"{self.converted.local_path} couldn't verify remote hash")


canny = ControlLoraConversion(
    original=Hub(
        repo_id="lllyasviel/misc",
        filename="control-lora-canny-rank128.safetensors",
        revision="71f7a66a7affe631c64af469fe647217d422cac0",
        expected_sha256="56389dbb245ca44de91d662529bd4298abc55ce2318f60bc19454fb72ff68247",
    ),
    converted=Hub(
        repo_id="refiners/sdxl.controllora.canny",
        filename="model.safetensors",
        expected_sha256="6edfa742e2b5191ce357fb559e236652b004feea490c4f1277b30abc9804321f",
    ),
)

cpds = ControlLoraConversion(
    original=Hub(
        repo_id="lllyasviel/misc",
        filename="fooocus_xl_cpds_128.safetensors",
        revision="71f7a66a7affe631c64af469fe647217d422cac0",
        expected_sha256="eec3fd8209a65b41341ea9f415de66909c97b30fb4d20965b3304e8e5251c2f1",
    ),
    converted=Hub(
        repo_id="refiners/sdxl.controllora.cpds",
        filename="model.safetensors",
        expected_sha256="9a3b2a86f32e4747e98531b0af8b59a804391b538949a0dd85263722b6e64db0",
    ),
)
