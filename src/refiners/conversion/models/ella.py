import logging

import requests
import torch

from refiners.conversion.utils import Conversion, Hub, TensorDict
from refiners.fluxion.utils import load_from_safetensors, save_to_safetensors


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> TensorDict:
    new_state_dict: TensorDict = {}

    for key in list(state_dict.keys()):
        if "latents" in key:
            new_key = "PerceiverResampler.Latents.ParameterInitialized.weight"
            new_state_dict[new_key] = state_dict.pop(key)
        elif "time_embedding" in key:
            new_key = key.replace("time_embedding", "TimestepEncoder.RangeEncoder").replace("linear", "Linear")
            new_state_dict[new_key] = state_dict.pop(key)
        elif "proj_in" in key:
            new_key = f"PerceiverResampler.Linear.{key.split('.')[-1]}"
            new_state_dict[new_key] = state_dict.pop(key)
        elif "time_aware" in key:
            new_key = f"PerceiverResampler.Residual.Linear.{key.split('.')[-1]}"
            new_state_dict[new_key] = state_dict.pop(key)
        elif "attn.in_proj" in key:
            layer_num = int(key.split(".")[2])
            query_param, key_param, value_param = state_dict.pop(key).chunk(3, dim=0)
            param_type = "weight" if "weight" in key else "bias"
            for i, param in enumerate([query_param, key_param, value_param]):
                new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_1.PerceiverAttention.Attention.Distribute.Linear_{i+1}.{param_type}"
                new_state_dict[new_key] = param
        elif "attn.out_proj" in key:
            layer_num = int(key.split(".")[2])
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_1.PerceiverAttention.Attention.Linear.{key.split('.')[-1]}"
            new_state_dict[new_key] = state_dict.pop(key)
        elif "ln_ff" in key:
            layer_num = int(key.split(".")[2])
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_2.AdaLayerNorm.Parallel.Chain.Linear.{key.split('.')[-1]}"
            new_state_dict[new_key] = state_dict.pop(key)
        elif "ln_1" in key or "ln_2" in key:
            layer_num = int(key.split(".")[2])
            n = 1 if int(key.split(".")[3].split("_")[-1]) == 2 else 2
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_1.PerceiverAttention.Distribute.AdaLayerNorm_{n}.Parallel.Chain.Linear.{key.split('.')[-1]}"
            new_state_dict[new_key] = state_dict.pop(key)
        elif "mlp" in key:
            layer_num = int(key.split(".")[2])
            n = 1 if "c_fc" in key else 2
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_2.FeedForward.Linear_{n}.{key.split('.')[-1]}"
            new_state_dict[new_key] = state_dict.pop(key)

    return new_state_dict


class ELLAConversion(Conversion):
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

    # TODO: use WeightRecipe instead
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
        original_weights = load_from_safetensors(self.original.local_path)

        # convert the state_dict
        converted_state_dict = convert_state_dict(original_weights)
        original_weights = self.change_dtype(converted_state_dict, self.dtype)

        # save the converted state_dict
        self.converted.local_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_safetensors(self.converted.local_path, original_weights)

        # check the converted state_dict
        assert self.converted.check_local_hash()
        try:
            assert self.converted.check_remote_hash()
        except requests.exceptions.HTTPError:
            logging.warning(f"{self.converted.local_path} couldn't verify remote hash")


sd15_t5xl = ELLAConversion(
    original=Hub(
        repo_id="QQGYLab/ELLA",
        filename="ella-sd1.5-tsc-t5xl.safetensors",
        revision="c07675dea7873abe24a4152e1140cf0131c217d2",
        expected_sha256="ca2018e325170d622389b531c0a061eea9d856b80e58e359ed54ade881517417",
    ),
    converted=Hub(
        repo_id="refiners/sd15.ella.tsc_t5xl",
        filename="model.safetensors",
        expected_sha256="ffc368afb97b93792f581d4a75275f4195cf76c225961cce61c3e1ef687df7da",
    ),
    dtype=torch.float16,
)
