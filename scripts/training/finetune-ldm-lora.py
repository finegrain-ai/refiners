import random
from typing import Any
from pydantic import BaseModel
from loguru import logger
from refiners.adapters.lora import LoraAdapter, Lora
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion.lora import LoraTarget
import refiners.fluxion.layers as fl
from torch import Tensor
from torch.utils.data import Dataset

from refiners.training_utils.callback import Callback
from refiners.training_utils.latent_diffusion import (
    FinetuneLatentDiffusionConfig,
    TextEmbeddingLatentsBatch,
    TextEmbeddingLatentsDataset,
    LatentDiffusionTrainer,
    LatentDiffusionConfig,
)


class LoraConfig(BaseModel):
    rank: int = 32
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    unet_targets: list[LoraTarget]
    text_encoder_targets: list[LoraTarget]
    lda_targets: list[LoraTarget]

    def apply_loras_to_target(self, module: fl.Chain, target: LoraTarget) -> None:
        for layer in module.layers(layer_type=target.get_class()):
            for linear, parent in layer.walk(fl.Linear):
                adapter = LoraAdapter(
                    target=linear,
                    rank=self.rank,
                )
                adapter.inject(parent)
                for linear in adapter.Lora.layers(fl.Linear):
                    linear.requires_grad_(requires_grad=True)


class TriggerPhraseDataset(TextEmbeddingLatentsDataset):
    def __init__(
        self,
        trainer: "LoraLatentDiffusionTrainer",
    ) -> None:
        super().__init__(trainer=trainer)
        self.trigger_phrase = trainer.config.lora.trigger_phrase
        self.use_only_trigger_probability = trainer.config.lora.use_only_trigger_probability
        logger.info(f"Trigger phrase: {self.trigger_phrase}")

    def process_caption(self, caption: str) -> str:
        caption = super().process_caption(caption=caption)
        if self.trigger_phrase:
            caption = (
                f"{self.trigger_phrase} {caption}"
                if random.random() < self.use_only_trigger_probability
                else self.trigger_phrase
            )
        return caption


class LoraLatentDiffusionConfig(FinetuneLatentDiffusionConfig):
    latent_diffusion: LatentDiffusionConfig
    lora: LoraConfig

    def model_post_init(self, __context: Any) -> None:
        """Pydantic v2 does post init differently, so we need to override this method too."""
        logger.info("Freezing models to train only the loras.")
        self.models["unet"].train = False
        self.models["text_encoder"].train = False
        self.models["lda"].train = False


class LoraLatentDiffusionTrainer(LatentDiffusionTrainer[LoraLatentDiffusionConfig]):
    def __init__(
        self,
        config: LoraLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadLoras(), SaveLoras()))

    def load_dataset(self) -> Dataset[TextEmbeddingLatentsBatch]:
        return TriggerPhraseDataset(trainer=self)


class LoadLoras(Callback[LoraLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: LoraLatentDiffusionTrainer) -> None:
        lora_config = trainer.config.lora
        for target in lora_config.unet_targets:
            lora_config.apply_loras_to_target(module=trainer.unet, target=target)
        for target in lora_config.text_encoder_targets:
            lora_config.apply_loras_to_target(module=trainer.text_encoder, target=target)
        for target in lora_config.lda_targets:
            lora_config.apply_loras_to_target(module=trainer.lda, target=target)


class SaveLoras(Callback[LoraLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: LoraLatentDiffusionTrainer) -> None:
        lora_config = trainer.config.lora

        def get_weight(linear: fl.Linear) -> Tensor:
            assert linear.bias is None
            return linear.state_dict()["weight"]

        def build_loras_safetensors(module: fl.Chain, key_prefix: str) -> dict[str, Tensor]:
            weights: list[Tensor] = []
            for lora in module.layers(layer_type=Lora):
                linears = list(lora.layers(fl.Linear))
                assert len(linears) == 2
                # See `load_lora_weights` in refiners.adapters.lora
                weights.extend((get_weight(linears[1]), get_weight(linears[0])))  # aka (up_weight, down_weight)
            return {f"{key_prefix}{i:03d}": w for i, w in enumerate(weights)}

        tensors: dict[str, Tensor] = {}
        metadata: dict[str, str] = {}

        if lora_config.unet_targets:
            tensors |= build_loras_safetensors(trainer.unet, key_prefix="unet.")
            metadata |= {"unet_targets": ",".join(lora_config.unet_targets)}

        if lora_config.text_encoder_targets:
            tensors |= build_loras_safetensors(trainer.text_encoder, key_prefix="text_encoder.")
            metadata |= {"text_encoder_targets": ",".join(lora_config.text_encoder_targets)}

        if lora_config.lda_targets:
            tensors |= build_loras_safetensors(trainer.lda, key_prefix="lda.")
            metadata |= {"lda_targets": ",".join(lora_config.lda_targets)}

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = LoraLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = LoraLatentDiffusionTrainer(config=config)
    trainer.train()
