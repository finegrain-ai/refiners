import random
from typing import Any

from loguru import logger
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset

from refiners.fluxion.utils import save_to_safetensors
from refiners.training_utils.callback import Callback
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.latent_diffusion import (
    FinetuneLatentDiffusionConfig,
    LatentDiffusionConfig,
    LatentDiffusionTrainer,
    TextEmbeddingLatentsBatch,
    TextEmbeddingLatentsDataset,
)

from refiners.training_utils. import 

class ColorPaletteConfig(BaseModel):
    max_colors: int = 8
    model_dim: int = 256
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    
class ColorPaletteDataset(TextEmbeddingLatentsDataset):
    def __init__(
        self,
        trainer: "ColorPaletteLatentDiffusionTrainer",
    ) -> None:
        super().__init__(trainer=trainer)
        self.trigger_phrase = trainer.config.color_palette.trigger_phrase
        self.use_only_trigger_probability = trainer.config.color_palette.use_only_trigger_probability
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


class ColorPaletteLatentDiffusionConfig(FinetuneLatentDiffusionConfig):
    dataset: HuggingfaceDatasetConfig
    latent_diffusion: LatentDiffusionConfig
    color_palette: ColorPaletteConfig

    def model_post_init(self, __context: Any) -> None:
        """Pydantic v2 does post init differently, so we need to override this method too."""
        logger.info("Freezing models to train only the color palette.")
        self.models["unet"].train = False
        self.models["text_encoder"].train = False
        self.models["lda"].train = False


class ColorPaletteLatentDiffusionTrainer(LatentDiffusionTrainer[ColorPaletteLatentDiffusionConfig]):
    def __init__(
        self,
        config: ColorPaletteLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadColorPalette(), SaveColorPalette()))

    def load_dataset(self) -> Dataset[TextEmbeddingLatentsBatch]:
        return ColorPaletteDataset(trainer=self)


class LoadColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        color_palette_config = trainer.config.color_palette
        
        adapter = SD1ColorPaletteAdapter(
            target=trainer.unet,
            model_dim=color_palette_config.model_dim,
            max_colors=color_palette_config.max_colors,
        )
        
        adapter.inject()


class SaveColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        metadata: dict[str, str] = {}
        
        model = trainer.unet
        adapter = model.parent
        
        tensors = {f"unet.{i:03d}": w for i, w in enumerate(adapter.weights)}
        metadata = {f"unet_targets": ",".join(adapter.sub_targets)}
        
        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = ColorPaletteLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = ColorPaletteLatentDiffusionTrainer(config=config)
    trainer.train()
