import random
from typing import Any
from pydantic import BaseModel
from loguru import logger
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion.lora import LoraTarget, LoraAdapter, MODELS, lora_targets
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

        for model_name in MODELS:
            model = getattr(trainer, model_name)
            model_targets: list[LoraTarget] = getattr(lora_config, f"{model_name}_targets")
            adapter = LoraAdapter[type(model)](
                model,
                sub_targets=[x for target in model_targets for x in lora_targets(model, target)],
                rank=lora_config.rank,
            )
            for sub_adapter, _ in adapter.sub_adapters:
                for linear in sub_adapter.Lora.layers(fl.Linear):
                    linear.requires_grad_(requires_grad=True)
            adapter.inject()


class SaveLoras(Callback[LoraLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: LoraLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        metadata: dict[str, str] = {}

        for model_name in MODELS:
            model = getattr(trainer, model_name)
            adapter = model.parent
            tensors |= {f"{model_name}.{i:03d}": w for i, w in enumerate(adapter.weights)}
            metadata |= {f"{model_name}_targets": ",".join(adapter.sub_targets)}

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
