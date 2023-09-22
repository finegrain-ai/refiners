from enum import IntEnum
from functools import partial
from typing import Generic, TypeVar, Any, Callable, TYPE_CHECKING

from torch import Tensor, cat, zeros_like, device as Device, dtype as DType
from PIL import Image

from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.adapters.lora import Lora
from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.fluxion.utils import image_to_tensor, normalize
import refiners.fluxion.layers as fl

if TYPE_CHECKING:
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TIPAdapter = TypeVar("TIPAdapter", bound="IPAdapter[Any]")  # Self (see PEP 673)


class ImageProjection(fl.Chain):
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


class InjectionPoint(fl.Chain):
    pass


class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
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
                fl.Distribute(
                    # Note: the same query is used for image cross-attention as for text cross-attention
                    InjectionPoint(),  # Wq
                    fl.Parallel(
                        fl.Chain(
                            fl.Slicing(dim=1, start=0, length=text_sequence_length),
                            InjectionPoint(),  # Wk
                        ),
                        fl.Chain(
                            fl.Slicing(dim=1, start=text_sequence_length, length=image_sequence_length),
                            fl.Linear(
                                in_features=self.target.key_embedding_dim,
                                out_features=self.target.inner_dim,
                                bias=self.target.use_bias,
                                device=target.device,
                                dtype=target.dtype,
                            ),  # Wk'
                        ),
                    ),
                    fl.Parallel(
                        fl.Chain(
                            fl.Slicing(dim=1, start=0, length=text_sequence_length),
                            InjectionPoint(),  # Wv
                        ),
                        fl.Chain(
                            fl.Slicing(dim=1, start=text_sequence_length, length=image_sequence_length),
                            fl.Linear(
                                in_features=self.target.key_embedding_dim,
                                out_features=self.target.inner_dim,
                                bias=self.target.use_bias,
                                device=target.device,
                                dtype=target.dtype,
                            ),  # Wv'
                        ),
                    ),
                ),
                fl.Sum(
                    fl.Chain(
                        fl.Lambda(func=partial(self.select_qkv, index=_CrossAttnIndex.TXT_CROSS_ATTN)),
                        ScaledDotProductAttention(num_heads=target.num_heads, is_causal=target.is_causal),
                    ),
                    fl.Chain(
                        fl.Lambda(func=partial(self.select_qkv, index=_CrossAttnIndex.IMG_CROSS_ATTN)),
                        ScaledDotProductAttention(num_heads=target.num_heads, is_causal=target.is_causal),
                        fl.Lambda(func=self.scale_outputs),
                    ),
                ),
                InjectionPoint(),  # proj
            )

    def select_qkv(
        self, query: Tensor, keys: tuple[Tensor, Tensor], values: tuple[Tensor, Tensor], index: _CrossAttnIndex
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (query, keys[index.value], values[index.value])

    def scale_outputs(self, x: Tensor) -> Tensor:
        return x * self.scale

    def _predicate(self, k: type[fl.Module]) -> Callable[[fl.Module, fl.Chain], bool]:
        def f(m: fl.Module, _: fl.Chain) -> bool:
            if isinstance(m, Lora):  # do not adapt LoRAs
                raise StopIteration
            return isinstance(m, k)

        return f

    def _target_linears(self) -> list[fl.Linear]:
        return [m for m, _ in self.target.walk(self._predicate(fl.Linear)) if isinstance(m, fl.Linear)]

    def inject(self: "CrossAttentionAdapter", parent: fl.Chain | None = None) -> "CrossAttentionAdapter":
        linears = self._target_linears()
        assert len(linears) == 4  # Wq, Wk, Wv and Proj

        injection_points = list(self.layers(InjectionPoint))
        assert len(injection_points) == 4

        for linear, ip in zip(linears, injection_points):
            ip.append(linear)
            assert len(ip) == 1

        return super().inject(parent)

    def eject(self) -> None:
        injection_points = list(self.layers(InjectionPoint))
        assert len(injection_points) == 4

        for ip in injection_points:
            ip.pop()
            assert len(ip) == 0

        super().eject()


class IPAdapter(Generic[T], fl.Chain, Adapter[T]):
    # Prevent PyTorch module registration
    _clip_image_encoder: list[CLIPImageEncoderH]
    _image_proj: list[ImageProjection]

    def __init__(
        self,
        target: T,
        clip_image_encoder: CLIPImageEncoderH | None = None,
        scale: float = 1.0,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        cross_attn_2d = target.ensure_find(CrossAttentionBlock2d)

        self._clip_image_encoder = [clip_image_encoder or CLIPImageEncoderH(device=target.device, dtype=target.dtype)]
        self._image_proj = [
            ImageProjection(
                clip_image_embedding_dim=self.clip_image_encoder.output_dim,
                clip_text_embedding_dim=cross_attn_2d.context_embedding_dim,
                device=target.device,
                dtype=target.dtype,
            )
        ]

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

                cross_attn.load_state_dict(state_dict=cross_attn_state_dict)

    @property
    def clip_image_encoder(self) -> CLIPImageEncoderH:
        return self._clip_image_encoder[0]

    @property
    def image_proj(self) -> ImageProjection:
        return self._image_proj[0]

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
        return normalize(
            image_to_tensor(image.resize(size), device=self.target.device, dtype=self.target.dtype),
            mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
            std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
        )
