from refiners.adapters.lora import Lora
from refiners.foundationals.latent_diffusion.lora import apply_loras_to_target, LoraTarget
import refiners.fluxion.layers as fl


def test_lora_target_self() -> None:
    chain = fl.Chain(
        fl.Chain(
            fl.Linear(in_features=1, out_features=1),
            fl.Linear(in_features=1, out_features=1),
        ),
        fl.Linear(in_features=1, out_features=2),
    )
    apply_loras_to_target(chain, LoraTarget.Self, 1, 1.0)

    assert len(list(chain.layers(Lora))) == 3
