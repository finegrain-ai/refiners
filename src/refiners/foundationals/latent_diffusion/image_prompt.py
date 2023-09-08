from enum import IntEnum
from functools import partial
from typing import Generic, TypeVar, Any

from torch import Tensor, as_tensor, cat, zeros_like, device as Device, dtype as DType
from PIL import Image

from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.clip.image_encoder import CLIPImageEncoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.fluxion.utils import image_to_tensor
import refiners.fluxion.layers as fl

T = TypeVar("T", bound=SD1UNet | SDXLUNet)
TIPAdapter = TypeVar("TIPAdapter", bound="IPAdapter[Any]")  # Self (see PEP 673)


class ImageProjection(fl.Chain):
    structural_attrs = ["clip_image_embedding_dim", "clip_text_embedding_dim", "sequence_length"]

    def __init__(
        self,
        clip_image_embedding_dim: int = 1024,
        clip_text_embedding_dim: int = 768,
        sequence_length: int = 4,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.clip_image_embedding_dim = clip_image_embedding_dim
        self.clip_text_embedding_dim = clip_text_embedding_dim
        self.sequence_length = sequence_length
        super().__init__(
            fl.Linear(
                in_features=clip_image_embedding_dim,
                out_features=clip_text_embedding_dim * sequence_length,
                device=device,
                dtype=dtype,
            ),
            fl.Reshape(sequence_length, clip_text_embedding_dim),
            fl.LayerNorm(normalized_shape=clip_text_embedding_dim, device=device, dtype=dtype),
        )


class _CrossAttnIndex(IntEnum):
    TXT_CROSS_ATTN = 0  # text cross-attention
    IMG_CROSS_ATTN = 1  # image cross-attention


# Fluxion's Attention layer drop-in replacement implementing Decoupled Cross-Attention
class IPAttention(fl.Chain):
    structural_attrs = [
        "embedding_dim",
        "text_sequence_length",
        "image_sequence_length",
        "scale",
        "num_heads",
        "heads_dim",
        "key_embedding_dim",
        "value_embedding_dim",
        "inner_dim",
        "use_bias",
        "is_causal",
    ]

    def __init__(
        self,
        embedding_dim: int,
        text_sequence_length: int = 77,
        image_sequence_length: int = 4,
        scale: float = 1.0,
        num_heads: int = 1,
        key_embedding_dim: int | None = None,
        value_embedding_dim: int | None = None,
        inner_dim: int | None = None,
        use_bias: bool = True,
        is_causal: bool | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        assert (
            embedding_dim % num_heads == 0
        ), f"embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
        self.embedding_dim = embedding_dim
        self.text_sequence_length = text_sequence_length
        self.image_sequence_length = image_sequence_length
        self.scale = scale
        self.num_heads = num_heads
        self.heads_dim = embedding_dim // num_heads
        self.key_embedding_dim = key_embedding_dim or embedding_dim
        self.value_embedding_dim = value_embedding_dim or embedding_dim
        self.inner_dim = inner_dim or embedding_dim
        self.use_bias = use_bias
        self.is_causal = is_causal
        super().__init__(
            fl.Distribute(
                # Note: the same query is used for image cross-attention as for text cross-attention
                fl.Linear(
                    in_features=self.embedding_dim,
                    out_features=self.inner_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),  # Wq
                fl.Parallel(
                    fl.Chain(
                        fl.Slicing(dim=1, start=0, length=text_sequence_length),
                        fl.Linear(
                            in_features=self.key_embedding_dim,
                            out_features=self.inner_dim,
                            bias=self.use_bias,
                            device=device,
                            dtype=dtype,
                        ),  # Wk
                    ),
                    fl.Chain(
                        fl.Slicing(dim=1, start=text_sequence_length, length=image_sequence_length),
                        fl.Linear(
                            in_features=self.key_embedding_dim,
                            out_features=self.inner_dim,
                            bias=self.use_bias,
                            device=device,
                            dtype=dtype,
                        ),  # Wk'
                    ),
                ),
                fl.Parallel(
                    fl.Chain(
                        fl.Slicing(dim=1, start=0, length=text_sequence_length),
                        fl.Linear(
                            in_features=self.key_embedding_dim,
                            out_features=self.inner_dim,
                            bias=self.use_bias,
                            device=device,
                            dtype=dtype,
                        ),  # Wv
                    ),
                    fl.Chain(
                        fl.Slicing(dim=1, start=text_sequence_length, length=image_sequence_length),
                        fl.Linear(
                            in_features=self.key_embedding_dim,
                            out_features=self.inner_dim,
                            bias=self.use_bias,
                            device=device,
                            dtype=dtype,
                        ),  # Wv'
                    ),
                ),
            ),
            fl.Sum(
                fl.Chain(
                    fl.Lambda(func=partial(self.select_qkv, index=_CrossAttnIndex.TXT_CROSS_ATTN)),
                    ScaledDotProductAttention(num_heads=num_heads, is_causal=is_causal),
                ),
                fl.Chain(
                    fl.Lambda(func=partial(self.select_qkv, index=_CrossAttnIndex.IMG_CROSS_ATTN)),
                    ScaledDotProductAttention(num_heads=num_heads, is_causal=is_causal),
                    fl.Lambda(func=self.scale_outputs),
                ),
            ),
            fl.Linear(
                in_features=self.inner_dim,
                out_features=self.embedding_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )

    def select_qkv(
        self, query: Tensor, keys: tuple[Tensor, Tensor], values: tuple[Tensor, Tensor], index: _CrossAttnIndex
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (query, keys[index.value], values[index.value])

    def scale_outputs(self, x: Tensor) -> Tensor:
        return x * self.scale


class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    structural_attrs = ["text_sequence_length", "image_sequence_length", "scale"]

    def __init__(
        self,
        target: fl.Attention,
        text_sequence_length: int = 77,
        image_sequence_length: int = 4,
        scale: float = 1.0,
    ) -> None:
        self.text_sequence_length = text_sequence_length
        self.image_sequence_length = image_sequence_length
        self.scale = scale
        with self.setup_adapter(target):
            super().__init__(
                IPAttention(
                    embedding_dim=target.embedding_dim,
                    text_sequence_length=text_sequence_length,
                    image_sequence_length=image_sequence_length,
                    scale=scale,
                    num_heads=target.num_heads,
                    key_embedding_dim=target.key_embedding_dim,
                    value_embedding_dim=target.value_embedding_dim,
                    inner_dim=target.inner_dim,
                    use_bias=target.use_bias,
                    is_causal=target.is_causal,
                    device=target.device,
                    dtype=target.dtype,
                )
            )

    def get_parameter_name(self, matrix: str, bias: bool = False) -> str:
        match matrix:
            case "wq":
                index = 0
            case "wk":
                index = 1
            case "wk_prime":
                index = 2
            case "wv":
                index = 3
            case "wv_prime":
                index = 4
            case "proj":
                index = 5
            case _:
                raise ValueError(f"Unexpected matrix name {matrix}")

        linear = list(self.IPAttention.layers(fl.Linear))[index]
        param = getattr(linear, "bias" if bias else "weight")
        name = next((n for n, p in self.named_parameters() if id(p) == id(param)), None)
        assert name is not None
        return name


class IPAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(
        self,
        target: T,
        clip_image_encoder: CLIPImageEncoder,
        scale: float = 1.0,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self.clip_image_encoder = clip_image_encoder
        self.image_proj = ImageProjection(device=target.device, dtype=target.dtype)

        self.sub_adapters = [
            CrossAttentionAdapter(target=cross_attn, scale=scale)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

        if weights is not None:
            image_proj_state_dict: dict[str, Tensor] = {
                k.removeprefix("image_proj."): v for k, v in weights.items() if k.startswith("image_proj.")
            }
            self.image_proj.load_state_dict(image_proj_state_dict)

            for i, cross_attn in enumerate(self.sub_adapters):
                cross_attn_state_dict: dict[str, Tensor] = {}
                for k, v in weights.items():
                    prefix = f"ip_adapter.{i:03d}."
                    if not k.startswith(prefix):
                        continue
                    cross_attn_state_dict[k.removeprefix(prefix)] = v

                # Retrieve original (frozen) cross-attention weights
                # Note: this assumes the target UNet has already loaded weights
                cross_attn_linears = list(cross_attn.target.layers(fl.Linear))
                assert len(cross_attn_linears) == 4  # Wq, Wk, Wv and Proj

                cross_attn_state_dict[cross_attn.get_parameter_name("wq")] = cross_attn_linears[0].weight
                cross_attn_state_dict[cross_attn.get_parameter_name("wk")] = cross_attn_linears[1].weight
                cross_attn_state_dict[cross_attn.get_parameter_name("wv")] = cross_attn_linears[2].weight
                cross_attn_state_dict[cross_attn.get_parameter_name("proj")] = cross_attn_linears[3].weight
                cross_attn_state_dict[cross_attn.get_parameter_name("proj", bias=True)] = cross_attn_linears[3].bias

                cross_attn.load_state_dict(state_dict=cross_attn_state_dict)

    def inject(self: "TIPAdapter", parent: fl.Chain | None = None) -> "TIPAdapter":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()

    # These should be concatenated to the CLIP text embedding before setting the UNet context
    def compute_clip_image_embedding(self, image_prompt: Tensor | None) -> Tensor:
        clip_embedding = self.clip_image_encoder(image_prompt)
        conditional_embedding = self.image_proj(clip_embedding)
        negative_embedding = self.image_proj(zeros_like(clip_embedding))
        return cat((negative_embedding, conditional_embedding))

    def preprocess_image(
        self,
        image: Image.Image,
        size: tuple[int, int] = (224, 224),
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> Tensor:
        # Default mean and std are parameters from https://github.com/openai/CLIP
        return self._normalize(
            image_to_tensor(image.resize(size), device=self.target.device, dtype=self.target.dtype),
            mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
            std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
        )

    # Adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
    @staticmethod
    def _normalize(tensor: Tensor, mean: list[float], std: list[float], inplace: bool = False) -> Tensor:
        assert tensor.is_floating_point()
        assert tensor.ndim >= 3

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype

        mean_tensor = as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std_tensor = as_tensor(std, dtype=tensor.dtype, device=tensor.device)

        if (std_tensor == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

        if mean_tensor.ndim == 1:
            mean_tensor = mean_tensor.view(-1, 1, 1)

        if std_tensor.ndim == 1:
            std_tensor = std_tensor.view(-1, 1, 1)

        return tensor.sub_(mean_tensor).div_(std_tensor)
