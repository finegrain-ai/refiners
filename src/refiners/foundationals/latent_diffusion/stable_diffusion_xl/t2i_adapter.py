from typing import cast, Iterable

from torch import Tensor

from refiners.foundationals.latent_diffusion.t2i_adapter import T2IAdapter, T2IFeatures, ConditionEncoderXL
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import SDXLUNet
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import ResidualAccumulator
import refiners.fluxion.layers as fl


class SDXLT2IAdapter(T2IAdapter[SDXLUNet]):
    def __init__(
        self,
        target: SDXLUNet,
        name: str,
        condition_encoder: ConditionEncoderXL | None = None,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        self.residual_indices = (3, 5, 8)  # the UNet's middle block is handled separately (see `inject` and `eject`)
        super().__init__(
            target=target,
            name=name,
            condition_encoder=condition_encoder or ConditionEncoderXL(device=target.device, dtype=target.dtype),
            weights=weights,
        )

    def inject(self: "SDXLT2IAdapter", parent: fl.Chain | None = None) -> "SDXLT2IAdapter":
        def sanity_check_t2i(block: fl.Module) -> None:
            for t2i_layer in block.layers(layer_type=T2IFeatures):
                assert t2i_layer.name != self.name, f"T2I-Adapter named {self.name} is already injected"

        for n, block in enumerate(cast(Iterable[fl.Chain], self.target.DownBlocks)):
            if n not in self.residual_indices:
                continue
            sanity_check_t2i(block)
            block.insert_before_type(
                ResidualAccumulator, T2IFeatures(name=self.name, index=self.residual_indices.index(n))
            )
        sanity_check_t2i(self.target.MiddleBlock)
        # Special case: the MiddleBlock has no ResidualAccumulator (this is done via a subsequent layer) so just append
        self.target.MiddleBlock.append(T2IFeatures(name=self.name, index=-1))
        return super().inject(parent)

    def eject(self: "SDXLT2IAdapter") -> None:
        def eject_t2i(block: fl.Module) -> None:
            t2i_layers = [
                t2i_layer for t2i_layer in block.layers(layer_type=T2IFeatures) if t2i_layer.name == self.name
            ]
            assert len(t2i_layers) == 1
            block.remove(t2i_layers.pop())

        for n, block in enumerate(cast(Iterable[fl.Chain], self.target.DownBlocks)):
            if n not in self.residual_indices:
                continue
            eject_t2i(block)
        eject_t2i(self.target.MiddleBlock)
        super().eject()
