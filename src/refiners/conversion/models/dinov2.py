import logging

import requests
import torch

from refiners.conversion.utils import Conversion, Hub
from refiners.fluxion.utils import load_tensors, save_to_safetensors


def convert_dinov2_facebook(weights: dict[str, torch.Tensor]) -> None:
    """Convert a DINOv2 weights from facebook to refiners."""
    # get depth from "blocks" keys
    depth = max([int(k.split(".")[1]) for k in weights.keys() if k.startswith("blocks.")]) + 1

    # only needed when pre-training
    del weights["mask_token"]

    # squeeze cls_token and position_embeddings
    weights["cls_token"] = weights["cls_token"].squeeze(0)
    weights["pos_embed"] = weights["pos_embed"].squeeze(0)

    # rename "w12" to "fc1" and "w3" to "fc2", only for giant model
    for key in list(weights.keys()):
        if "w3" in key:
            new_key = key.replace("w3", "fc2")
            weights[new_key] = weights.pop(key)
        elif "w12" in key:
            # we swap w1 and w2 because of the difference between our GLU implementation and theirs
            # see https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/layers/swiglu_ffn.py#L31-L34
            # and https://github.com/finegrain-ai/refiners/blob/a2ee70578361e4d84a65a8708564480a9b0ec67e/src/refiners/fluxion/layers/activations.py#L158-L160
            weight = weights.pop(key)
            w1, w2 = weight.chunk(2, dim=0)
            w21 = torch.cat([w2, w1], dim=0)
            new_key = key.replace("w12", "fc1")
            weights[new_key] = w21

    rename_keys: list[tuple[str, str]] = [
        ("cls_token", "Concatenate.ClassToken.Parameter.weight"),
        ("pos_embed", "PositionalEncoder.PositionalEmbedding.Parameter.weight"),
        ("patch_embed.proj.weight", "Concatenate.PatchEncoder.Conv2d.weight"),
        ("patch_embed.proj.bias", "Concatenate.PatchEncoder.Conv2d.bias"),
        ("norm.weight", "LayerNorm.weight"),
        ("norm.bias", "LayerNorm.bias"),
    ]
    for i in range(depth):
        rename_keys.append(
            (
                f"blocks.{i}.norm1.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.LayerNorm.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm1.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.LayerNorm.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.attn.proj.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Linear.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.attn.proj.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Linear.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.ls1.gamma",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.LayerScale.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm2.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.LayerNorm.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm2.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.LayerNorm.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc1.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc1.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc2.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc2.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.ls2.gamma",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.LayerScale.weight",
            ),
        )

    if "register_tokens" in weights:
        weights["register_tokens"] = weights["register_tokens"].squeeze(0)
        rename_keys.append(("register_tokens", "Registers.Parameter.weight"))

    # rename keys
    for old_key, new_key in rename_keys:
        weights[new_key] = weights.pop(old_key)

    # split the qkv weights and biases
    for i in range(depth):
        qkv_weight = weights.pop(f"blocks.{i}.attn.qkv.weight")
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_1.weight"] = q_weight
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_2.weight"] = k_weight
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_3.weight"] = v_weight

        qkv_bias = weights.pop(f"blocks.{i}.attn.qkv.bias")
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_1.bias"] = q_bias
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_2.bias"] = k_bias
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_3.bias"] = v_bias


class DinoV2Conversion(Conversion):
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

    def convert(self) -> None:  # type: ignore
        """Convert the weights from the original to the converted weights."""
        logging.info(f"Converting {self.original.repo_id}/{self.original.filename} to {self.converted.repo_id}")

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
        original_weights = load_tensors(self.original.local_path)

        # convert the state_dict
        convert_dinov2_facebook(original_weights)  # FIXME: this is inplace
        original_weights = self.change_dtype(original_weights, self.dtype)

        # save the converted state_dict
        self.converted.local_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_safetensors(self.converted.local_path, original_weights)

        # check the converted state_dict
        assert self.converted.check_local_hash()
        try:
            assert self.converted.check_remote_hash()
        except requests.exceptions.HTTPError:
            logging.warning(f"{self.converted.local_path} couldn't verify remote hash")


small = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vits14.pth",
        expected_sha256="b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.small.patch_14",
        filename="model.safetensors",
        expected_sha256="56a4b77856e20bbb5c4f0ce135089d4cd72da344dcdb278ba0c1376c8545e543",
    ),
)
small_reg = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vits14_reg4.pth",
        expected_sha256="f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.small.patch_14.reg_4",
        filename="model.safetensors",
        expected_sha256="beee454507762018616635099c0ac30c7a6e4e08fbd9363c5e5d2a8f1935c3f2",
    ),
)
base = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vitb14.pth",
        expected_sha256="0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.base.patch_14",
        filename="model.safetensors",
        expected_sha256="59b778ed980bc02843456d3fbe1893943922ac7759a9a706ca286dd45d10db1f",
    ),
)
base_reg = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vitb14_reg4.pth",
        expected_sha256="73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.base.patch_14.reg_4",
        filename="model.safetensors",
        expected_sha256="7f91aa7cd5aa51d665949ba328a938967164b363ebaacb8cae914143a7e004e7",
    ),
)
large = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vitl14.pth",
        expected_sha256="d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.large.patch_14",
        filename="model.safetensors",
        expected_sha256="2ba79218d37482455db0d9967dfad024c3ad525499f8de0e3db5ff83faf80414",
    ),
)
large_reg = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vitl14_reg4.pth",
        expected_sha256="36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.large.patch_14.reg_4",
        filename="model.safetensors",
        expected_sha256="e1d5a183a0ec15c5ac0a9e388038a07f8e90dd19e001b7bd4f7ffe3c5761667c",
    ),
)
giant = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vitg14.pth",
        expected_sha256="baf8467e50af277596bbbafa06887c177ee899ab46033649c383577d7e9309d3",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.giant.patch_14",
        filename="model.safetensors",
        expected_sha256="5a2d6088f4fd4aa1bf527ce0edf2ae3e76eee70c900b90716c18ad7daa4a1f2f",
    ),
)
giant_reg = DinoV2Conversion(
    original=Hub(
        repo_id="facebook/github_dinov2",
        filename="vitg14_reg4.pth",
        expected_sha256="746ecb8c6301c645c5c855be91687d274587d6e48fdaec4a729753160b34a283",
        download_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
    ),
    converted=Hub(
        repo_id="refiners/dinov2.giant.patch_14.reg_4",
        filename="model.safetensors",
        expected_sha256="d5f7f0917926d4fe72cd33408f79562c5d524c3e8aee999830129eecabda56a2",
    ),
)
