from typing import cast, Iterable

from torch import Tensor

from refiners.foundationals.latent_diffusion.t2i_adapter import T2IAdapter, T2IFeatures, ConditionEncoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet, ResidualAccumulator
import refiners.fluxion.layers as fl


class SD1T2IAdapter(T2IAdapter[SD1UNet]):
    def __init__(
        self,
        target: SD1UNet,
        name: str,
        condition_encoder: ConditionEncoder | None = None,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        self.residual_indices = (2, 5, 8, 11)
        super().__init__(
            target=target,
            name=name,
            condition_encoder=condition_encoder or ConditionEncoder(device=target.device, dtype=target.dtype),
            weights=weights,
        )

    def inject(self: "SD1T2IAdapter", parent: fl.Chain | None = None) -> "SD1T2IAdapter":
        for n, block in enumerate(cast(Iterable[fl.Chain], self.target.DownBlocks)):
            if n not in self.residual_indices:
                continue
            for t2i_layer in block.layers(layer_type=T2IFeatures):
                assert t2i_layer.name != self.name, f"T2I-Adapter named {self.name} is already injected"
            block.insert_before_type(
                ResidualAccumulator, T2IFeatures(name=self.name, index=self.residual_indices.index(n))
            )
        return super().inject(parent)

    def eject(self: "SD1T2IAdapter") -> None:
        for n, block in enumerate(cast(Iterable[fl.Chain], self.target.DownBlocks)):
            if n not in self.residual_indices:
                continue
            t2i_layers = [
                t2i_layer for t2i_layer in block.layers(layer_type=T2IFeatures) if t2i_layer.name == self.name
            ]
            assert len(t2i_layers) == 1
            block.remove(t2i_layers.pop())
        super().eject()
