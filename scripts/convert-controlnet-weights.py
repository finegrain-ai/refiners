import torch
from diffusers import ControlNetModel
from safetensors.torch import save_file
from refiners.fluxion.utils import (
    forward_order_of_execution,
    verify_shape_match,
    convert_state_dict,
)
from refiners.foundationals.latent_diffusion.controlnet import Controlnet
from refiners.foundationals.latent_diffusion.schedulers.dpm_solver import DPMSolver
from refiners.foundationals.latent_diffusion import UNet


@torch.no_grad()
def convert(controlnet_src: ControlNetModel) -> dict[str, torch.Tensor]:
    controlnet = Controlnet(name="mycn")

    condition = torch.randn(1, 3, 512, 512)
    controlnet.set_controlnet_condition(condition)

    unet = UNet(4, clip_embedding_dim=768)
    unet.insert(0, controlnet)
    clip_text_embedding = torch.rand(1, 77, 768)
    unet.set_clip_text_embedding(clip_text_embedding)

    scheduler = DPMSolver(num_inference_steps=10)
    timestep = scheduler.timesteps[0].unsqueeze(0)
    unet.set_timestep(timestep.unsqueeze(0))

    x = torch.randn(1, 4, 64, 64)

    # We need the hack below because our implementation is not strictly equivalent
    # to diffusers in order, since we compute the residuals inline instead of
    # in a separate step.

    source_order = forward_order_of_execution(controlnet_src, (x, timestep, clip_text_embedding, condition))
    target_order = forward_order_of_execution(controlnet, (x,))

    broken_k = ("Conv2d", (torch.Size([320, 320, 1, 1]), torch.Size([320])))

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
        "DownBlocks.Chain_2.CLIPLCrossAttention.Chain.Chain_1.Conv2d",
        "DownBlocks.Chain_2.CLIPLCrossAttention.Chain.Chain_3.Conv2d",
        "DownBlocks.Chain_2.Passthrough.Conv2d",
        "DownBlocks.Chain_3.CLIPLCrossAttention.Chain.Chain_1.Conv2d",
        "DownBlocks.Chain_3.CLIPLCrossAttention.Chain.Chain_3.Conv2d",
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

    broken_k = ("Conv2d", (torch.Size([640, 640, 1, 1]), torch.Size([640])))

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
        "DownBlocks.Chain_5.CLIPLCrossAttention.Chain.Chain_1.Conv2d",
        "DownBlocks.Chain_5.CLIPLCrossAttention.Chain.Chain_3.Conv2d",
        "DownBlocks.Chain_5.Passthrough.Conv2d",
        "DownBlocks.Chain_6.CLIPLCrossAttention.Chain.Chain_1.Conv2d",
        "DownBlocks.Chain_6.CLIPLCrossAttention.Chain.Chain_3.Conv2d",
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

    broken_k = ("Conv2d", (torch.Size([1280, 1280, 1, 1]), torch.Size([1280])))

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
        "DownBlocks.Chain_8.CLIPLCrossAttention.Chain.Chain_1.Conv2d",
        "DownBlocks.Chain_8.CLIPLCrossAttention.Chain.Chain_3.Conv2d",
        "DownBlocks.Chain_8.Passthrough.Conv2d",
        "DownBlocks.Chain_9.CLIPLCrossAttention.Chain.Chain_1.Conv2d",
        "DownBlocks.Chain_9.CLIPLCrossAttention.Chain.Chain_3.Conv2d",
        "DownBlocks.Chain_9.Passthrough.Conv2d",
        "DownBlocks.Chain_10.Passthrough.Conv2d",
        "DownBlocks.Chain_11.Passthrough.Conv2d",
        "DownBlocks.Chain_12.Passthrough.Conv2d",
        "MiddleBlock.CLIPLCrossAttention.Chain.Chain_1.Conv2d",
        "MiddleBlock.CLIPLCrossAttention.Chain.Chain_3.Conv2d",
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

    assert verify_shape_match(source_order, target_order)

    mapping: dict[str, str] = {}
    for model_type_shape in source_order:
        source_keys = source_order[model_type_shape]
        target_keys = target_order[model_type_shape]
        mapping.update(zip(target_keys, source_keys))

    state_dict = convert_state_dict(controlnet_src.state_dict(), controlnet.state_dict(), state_dict_mapping=mapping)

    return {k: v.half() for k, v in state_dict.items()}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        type=str,
        dest="source",
        required=True,
        help="Source model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default="output.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    controlnet_src = ControlNetModel.from_pretrained(args.source)
    tensors = convert(controlnet_src)
    save_file(tensors, args.output_file)


if __name__ == "__main__":
    main()
