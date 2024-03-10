# pyright: reportPrivateUsage=false
import argparse
from pathlib import Path

import torch
from diffusers import ControlNetModel  # type: ignore
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import no_grad, save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    SD1ControlnetAdapter,
    SD1UNet,
)


class Args(argparse.Namespace):
    source_path: str
    output_path: str | None


@no_grad()
def convert(args: Args) -> dict[str, torch.Tensor]:
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    controlnet_src: nn.Module = ControlNetModel.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=args.source_path,
        low_cpu_mem_usage=False,
    )
    unet = SD1UNet(in_channels=4)
    adapter = SD1ControlnetAdapter(unet, name="mycn").inject()
    controlnet = adapter.controlnet

    condition = torch.randn(1, 3, 512, 512)
    adapter.set_controlnet_condition(condition=condition)

    clip_text_embedding = torch.rand(1, 77, 768)
    unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)

    solver = DPMSolver(num_inference_steps=10)
    timestep = solver.timesteps[0].unsqueeze(dim=0)
    unet.set_timestep(timestep=timestep.unsqueeze(dim=0))

    x = torch.randn(1, 4, 64, 64)

    # We need the hack below because our implementation is not strictly equivalent
    # to diffusers in order, since we compute the residuals inline instead of
    # in a separate step.

    converter = ModelConverter(
        source_model=controlnet_src, target_model=controlnet, skip_output_check=True, verbose=False
    )

    source_order = converter._trace_module_execution_order(
        module=controlnet_src, args=(x, timestep, clip_text_embedding, condition), keys_to_skip=[]
    )
    target_order = converter._trace_module_execution_order(module=controlnet, args=(x,), keys_to_skip=[])

    broken_k = (nn.Conv2d, (torch.Size([320, 320, 1, 1]), torch.Size([320])))

    expected_source_order = [
        "down_blocks.0.attentions.0.proj_in",
        "down_blocks.0.attentions.0.proj_out",
        "down_blocks.0.attentions.1.proj_in",
        "down_blocks.0.attentions.1.proj_out",
        "controlnet_down_blocks.0",
        "controlnet_down_blocks.1",
        "controlnet_down_blocks.2",
        "controlnet_down_blocks.3",
    ]

    expected_target_order = [
        "DownBlocks.Chain_1.Passthrough.Conv2d",
        "DownBlocks.Chain_2.CLIPLCrossAttention.Chain_1.Conv2d",
        "DownBlocks.Chain_2.CLIPLCrossAttention.Chain_3.Conv2d",
        "DownBlocks.Chain_2.Passthrough.Conv2d",
        "DownBlocks.Chain_3.CLIPLCrossAttention.Chain_1.Conv2d",
        "DownBlocks.Chain_3.CLIPLCrossAttention.Chain_3.Conv2d",
        "DownBlocks.Chain_3.Passthrough.Conv2d",
        "DownBlocks.Chain_4.Passthrough.Conv2d",
    ]

    fixed_source_order = [
        "controlnet_down_blocks.0",
        "down_blocks.0.attentions.0.proj_in",
        "down_blocks.0.attentions.0.proj_out",
        "controlnet_down_blocks.1",
        "down_blocks.0.attentions.1.proj_in",
        "down_blocks.0.attentions.1.proj_out",
        "controlnet_down_blocks.2",
        "controlnet_down_blocks.3",
    ]

    assert source_order[broken_k] == expected_source_order
    assert target_order[broken_k] == expected_target_order
    source_order[broken_k] = fixed_source_order

    broken_k = (nn.Conv2d, (torch.Size([640, 640, 1, 1]), torch.Size([640])))

    expected_source_order = [
        "down_blocks.1.attentions.0.proj_in",
        "down_blocks.1.attentions.0.proj_out",
        "down_blocks.1.attentions.1.proj_in",
        "down_blocks.1.attentions.1.proj_out",
        "controlnet_down_blocks.4",
        "controlnet_down_blocks.5",
        "controlnet_down_blocks.6",
    ]

    expected_target_order = [
        "DownBlocks.Chain_5.CLIPLCrossAttention.Chain_1.Conv2d",
        "DownBlocks.Chain_5.CLIPLCrossAttention.Chain_3.Conv2d",
        "DownBlocks.Chain_5.Passthrough.Conv2d",
        "DownBlocks.Chain_6.CLIPLCrossAttention.Chain_1.Conv2d",
        "DownBlocks.Chain_6.CLIPLCrossAttention.Chain_3.Conv2d",
        "DownBlocks.Chain_6.Passthrough.Conv2d",
        "DownBlocks.Chain_7.Passthrough.Conv2d",
    ]

    fixed_source_order = [
        "down_blocks.1.attentions.0.proj_in",
        "down_blocks.1.attentions.0.proj_out",
        "controlnet_down_blocks.4",
        "down_blocks.1.attentions.1.proj_in",
        "down_blocks.1.attentions.1.proj_out",
        "controlnet_down_blocks.5",
        "controlnet_down_blocks.6",
    ]

    assert source_order[broken_k] == expected_source_order
    assert target_order[broken_k] == expected_target_order
    source_order[broken_k] = fixed_source_order

    broken_k = (nn.Conv2d, (torch.Size([1280, 1280, 1, 1]), torch.Size([1280])))

    expected_source_order = [
        "down_blocks.2.attentions.0.proj_in",
        "down_blocks.2.attentions.0.proj_out",
        "down_blocks.2.attentions.1.proj_in",
        "down_blocks.2.attentions.1.proj_out",
        "mid_block.attentions.0.proj_in",
        "mid_block.attentions.0.proj_out",
        "controlnet_down_blocks.7",
        "controlnet_down_blocks.8",
        "controlnet_down_blocks.9",
        "controlnet_down_blocks.10",
        "controlnet_down_blocks.11",
        "controlnet_mid_block",
    ]

    expected_target_order = [
        "DownBlocks.Chain_8.CLIPLCrossAttention.Chain_1.Conv2d",
        "DownBlocks.Chain_8.CLIPLCrossAttention.Chain_3.Conv2d",
        "DownBlocks.Chain_8.Passthrough.Conv2d",
        "DownBlocks.Chain_9.CLIPLCrossAttention.Chain_1.Conv2d",
        "DownBlocks.Chain_9.CLIPLCrossAttention.Chain_3.Conv2d",
        "DownBlocks.Chain_9.Passthrough.Conv2d",
        "DownBlocks.Chain_10.Passthrough.Conv2d",
        "DownBlocks.Chain_11.Passthrough.Conv2d",
        "DownBlocks.Chain_12.Passthrough.Conv2d",
        "MiddleBlock.CLIPLCrossAttention.Chain_1.Conv2d",
        "MiddleBlock.CLIPLCrossAttention.Chain_3.Conv2d",
        "MiddleBlock.Passthrough.Conv2d",
    ]

    fixed_source_order = [
        "down_blocks.2.attentions.0.proj_in",
        "down_blocks.2.attentions.0.proj_out",
        "controlnet_down_blocks.7",
        "down_blocks.2.attentions.1.proj_in",
        "down_blocks.2.attentions.1.proj_out",
        "controlnet_down_blocks.8",
        "controlnet_down_blocks.9",
        "controlnet_down_blocks.10",
        "controlnet_down_blocks.11",
        "mid_block.attentions.0.proj_in",
        "mid_block.attentions.0.proj_out",
        "controlnet_mid_block",
    ]

    assert source_order[broken_k] == expected_source_order
    assert target_order[broken_k] == expected_target_order
    source_order[broken_k] = fixed_source_order

    assert converter._assert_shapes_aligned(source_order=source_order, target_order=target_order), "Shapes not aligned"

    mapping: dict[str, str] = {}
    for model_type_shape in source_order:
        source_keys = source_order[model_type_shape]
        target_keys = target_order[model_type_shape]
        mapping.update(zip(target_keys, source_keys))

    state_dict = converter._convert_state_dict(
        source_state_dict=controlnet_src.state_dict(),
        target_state_dict=controlnet.state_dict(),
        state_dict_mapping=mapping,
    )

    return {k: v.half() for k, v in state_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a diffusers ControlNet model to a Refiners ControlNet model")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="lllyasviel/sd-controlnet-depth",
        help=(
            "Can be a path to a .bin, a .safetensors file, or a model identifier from Hugging Face Hub. Defaults to"
            " lllyasviel/sd-controlnet-depth"
        ),
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        required=False,
        default=None,
        help=(
            "Output path (.safetensors) for converted model. If not provided, the output path will be the same as the"
            " source path."
        ),
    )
    args = parser.parse_args(namespace=Args())
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-controlnet.safetensors"
    state_dict = convert(args=args)
    save_to_safetensors(path=args.output_path, tensors=state_dict)


if __name__ == "__main__":
    main()
