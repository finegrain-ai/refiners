from refiners.fluxion.layers import (
    Passthrough,
    Lambda,
    Chain,
    Concatenate,
    UseContext,
    SelfAttention,
    SetContext,
    Identity,
    Parallel,
)
from refiners.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock
from torch import Tensor


class SaveLayerNormAdapter(Chain, Adapter[SelfAttention]):
    def __init__(self, target: SelfAttention, context: str) -> None:
        self.context = context
        with self.setup_adapter(target):
            super().__init__(SetContext(self.context, "norm"), target)


class ReferenceOnlyControlAdapter(Chain, Adapter[SelfAttention]):
    def __init__(
        self,
        target: SelfAttention,
        context: str,
        sai: "SelfAttentionInjection",
    ) -> None:
        self.context = context
        self._sai = [sai]  # only to support setting `style_cfg` dynamically

        sa_guided = target.structural_copy()
        assert isinstance(sa_guided[0], Parallel)
        sa_guided.replace(
            sa_guided[0],
            Parallel(
                Identity(),
                Concatenate(Identity(), UseContext(self.context, "norm"), dim=1),
                Concatenate(Identity(), UseContext(self.context, "norm"), dim=1),
            ),
        )

        with self.setup_adapter(target):
            super().__init__(
                Parallel(sa_guided, Chain(Lambda(lambda x: x[:1]), target)),
                Lambda(self.compute_averaged_unconditioned_x),
            )

    def compute_averaged_unconditioned_x(self, x: Tensor, unguided_unconditioned_x: Tensor) -> Tensor:
        style_cfg = self._sai[0].style_cfg
        x[0] = style_cfg * x[0] + (1.0 - style_cfg) * unguided_unconditioned_x
        return x


class SelfAttentionInjection(Passthrough):
    # TODO: Does not support batching yet. Assumes concatenated inputs for classifier-free guidance

    def __init__(self, unet: SD1UNet, style_cfg: float = 0.5) -> None:
        # the style_cfg is the weight of the guide in unconditionned diffusion.
        # This value is recommended to be 0.5 on the sdwebui repo.
        self.style_cfg = style_cfg
        self._adapters: list[ReferenceOnlyControlAdapter] = []
        self._unet = [unet]

        guide_unet = unet.structural_copy()
        for i, attention_block in enumerate(guide_unet.layers(CrossAttentionBlock)):
            sa = attention_block.find(SelfAttention)
            assert sa is not None and sa.parent is not None
            SaveLayerNormAdapter(sa, context=f"self_attention_context_{i}").inject()

        for i, attention_block in enumerate(unet.layers(CrossAttentionBlock)):
            unet.set_context(f"self_attention_context_{i}", {"norm": None})

            sa = attention_block.find(SelfAttention)
            assert sa is not None and sa.parent is not None

            self._adapters.append(ReferenceOnlyControlAdapter(sa, context=f"self_attention_context_{i}", sai=self))

        super().__init__(
            Lambda(self.copy_diffusion_context),
            UseContext("self_attention_injection", "guide"),
            guide_unet,
            Lambda(self.restore_diffusion_context),
        )

    @property
    def unet(self):
        return self._unet[0]

    def inject(self) -> None:
        assert self not in self._unet[0], f"{self} is already injected"
        for adapter in self._adapters:
            adapter.inject()
        self.unet.insert(0, self)

    def eject(self) -> None:
        assert self.unet[0] == self, f"{self} is not the first element of target UNet"
        for adapter in self._adapters:
            adapter.eject()
        self.unet.pop(0)

    def set_controlnet_condition(self, condition: Tensor) -> None:
        self.set_context("self_attention_injection", {"guide": condition})

    def copy_diffusion_context(self, x: Tensor) -> Tensor:
        # This function allows to not disrupt the accumulation of residuals in the unet (if controlnet are used)
        self.set_context(
            "self_attention_residuals_buffer",
            {"buffer": self.use_context("unet")["residuals"]},
        )
        self.set_context(
            "unet",
            {"residuals": [0.0] * 13},
        )
        return x

    def restore_diffusion_context(self, x: Tensor) -> Tensor:
        self.set_context(
            "unet",
            {
                "residuals": self.use_context("self_attention_residuals_buffer")["buffer"],
            },
        )
        return x

    def structural_copy(self: "SelfAttentionInjection") -> "SelfAttentionInjection":
        raise RuntimeError("SelfAttentionInjection cannot be copied, eject it first.")
