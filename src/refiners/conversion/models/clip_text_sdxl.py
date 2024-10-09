import logging
from typing import NamedTuple, cast

import requests
import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection  # pyright: ignore[reportMissingTypeStubs]

import refiners.fluxion.layers as fl
from refiners.conversion.model_converter import ModelConverter
from refiners.conversion.utils import Conversion, Hub
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder, CLIPTextEncoderG, CLIPTextEncoderL
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import DoubleTextEncoder


class CLIPTextEncoderConfig(NamedTuple):
    architectures: list[str]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    hidden_act: str
    layer_norm_eps: float
    projection_dim: int


class ModelConverterHubDuo(Conversion):
    def __init__(
        self,
        original_repo_id: str,
        converted: Hub,
        dtype: torch.dtype,
    ) -> None:
        self.original = Hub(repo_id=original_repo_id, filename="", expected_sha256="")
        self.converted = converted
        self.dtype = dtype

    @staticmethod
    def setup_converter(source_path: str, subfolder: str, with_projection: bool) -> ModelConverter:
        # instantiate the transformers clip model
        cls = CLIPTextModelWithProjection if with_projection else CLIPTextModel
        source: nn.Module = cls.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            pretrained_model_name_or_path=source_path,
            subfolder=subfolder,
            low_cpu_mem_usage=False,
        )
        assert isinstance(source, nn.Module), "Source model is not a nn.Module"

        # get the model config from the transformers clip model
        config = cast(CLIPTextEncoderConfig, source.config)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

        # instantiate the refiners clip model
        target = CLIPTextEncoder(
            embedding_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            feedforward_dim=config.intermediate_size,
            use_quick_gelu=config.hidden_act == "quick_gelu",
        )
        if with_projection:
            target.append(
                module=fl.Linear(
                    in_features=config.hidden_size,
                    out_features=config.projection_dim,
                    bias=False,
                )
            )

        # initialize the inputs
        text = "What a nice cat you have there!"
        tokenizer = target.ensure_find(CLIPTokenizer)
        tokens = tokenizer(text)

        # run the converter
        converter = ModelConverter(
            source_model=source,
            target_model=target,
            skip_output_check=True,
            verbose=False,
        )
        if not converter.run(source_args=(tokens,), target_args=(text,)):
            raise RuntimeError("Model conversion failed")

        return converter

    def convert(self) -> None:
        logging.info(f"Converting {self.original.repo_id} to {self.converted.repo_id}")

        # initialize the model converters, find the mappings
        converter1 = self.setup_converter(
            source_path=self.original.repo_id,
            subfolder="text_encoder",
            with_projection=False,
        )
        converter2 = self.setup_converter(
            source_path=self.original.repo_id,
            subfolder="text_encoder_2",
            with_projection=True,
        )

        # load the CLIPTextEncoderL model
        text_encoder_l = CLIPTextEncoderL()
        text_encoder_l.load_state_dict(state_dict=converter1.get_state_dict())

        # load the CLIPTextEncoderG (with projection) model
        projection = cast(CLIPTextEncoder, converter2.target_model)[-1]
        assert isinstance(projection, fl.Linear)
        text_encoder_g_with_projection = CLIPTextEncoderG()
        text_encoder_g_with_projection.append(module=projection)
        text_encoder_g_with_projection.load_state_dict(state_dict=converter2.get_state_dict())

        # build DoubleTextEncoder from previous two models
        projection = text_encoder_g_with_projection.pop(index=-1)
        assert isinstance(projection, fl.Linear)
        double_text_encoder = DoubleTextEncoder(
            text_encoder_l=text_encoder_l,
            text_encoder_g=text_encoder_g_with_projection,
            projection=projection,
        )

        # extract the state_dict from the DoubleTextEncoder model
        state_dict = double_text_encoder.state_dict()
        state_dict = self.change_dtype(state_dict, self.dtype)

        # save the converted state_dict
        self.converted.local_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_safetensors(self.converted.local_path, state_dict)

        # check the converted state_dict
        assert self.converted.check_local_hash()
        try:
            assert self.converted.check_remote_hash()
        except requests.exceptions.HTTPError:
            logging.warning(f"{self.converted.local_path} couldn't verify remote hash")


stability = ModelConverterHubDuo(
    original_repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    converted=Hub(
        repo_id="refiners/sdxl.text_encoder",
        filename="model.safetensors",
        expected_sha256="238685accd000683e937085fb3a9c147675f5a1d7775a6810696131e93ddb147",
    ),
    dtype=torch.float16,
)
juggernautXL_v10 = ModelConverterHubDuo(
    original_repo_id="RunDiffusion/Juggernaut-X-v10",  # TODO(laurent): use file from civitai instead
    converted=Hub(
        repo_id="refiners/sdxl.juggernaut.v10.text_encoder",
        filename="model.safetensors",
        expected_sha256="50dde9c171e31d1c9dcd0539ba052e4fe69d90f126c812b0145da40a0a2c4361",
    ),
    dtype=torch.float16,
)
