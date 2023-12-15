import random
from typing import Any

from loguru import logger
from pydantic import BaseModel
from torch import Tensor, randn
from torch.utils.data import Dataset

from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.clip.concepts import ConceptExtender, EmbeddingExtender
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder, TokenEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.training_utils.callback import Callback
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.latent_diffusion import (
    FinetuneLatentDiffusionConfig,
    LatentDiffusionConfig,
    LatentDiffusionTrainer,
    TextEmbeddingLatentsBatch,
    TextEmbeddingLatentsDataset,
)

IMAGENET_TEMPLATES_SMALL = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

IMAGENET_STYLE_TEMPLATES_SMALL = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(TextEmbeddingLatentsDataset):
    templates: list[str] = []
    placeholder_token: str = ""

    def __init__(self, trainer: "LatentDiffusionTrainer[Any]") -> None:
        super().__init__(trainer)
        self.templates = (
            IMAGENET_STYLE_TEMPLATES_SMALL if self.config.textual_inversion.style_mode else IMAGENET_TEMPLATES_SMALL
        )
        self.placeholder_token = self.config.textual_inversion.placeholder_token

    def get_caption(self, index: int) -> str:
        # Ignore the dataset caption, if any: use a template instead
        return random.choice(self.templates).format(self.placeholder_token)


class TextualInversionConfig(BaseModel):
    # The new token to be learned
    placeholder_token: str = "*"
    # The token to be used as initializer; if None, a random vector is used
    initializer_token: str | None = None
    style_mode: bool = False

    def apply_textual_inversion_to_target(self, text_encoder: CLIPTextEncoder) -> None:
        adapter = ConceptExtender(target=text_encoder)
        tokenizer = text_encoder.ensure_find(CLIPTokenizer)
        token_encoder = text_encoder.ensure_find(TokenEncoder)
        if self.initializer_token is not None:
            bpe = tokenizer.byte_pair_encoding(token=self.initializer_token)
            assert " " not in bpe, "This initializer_token is not a single token."
            token = Tensor([tokenizer.token_to_id_mapping[bpe]]).int().to(text_encoder.device)
            init_embedding = token_encoder(token).squeeze(0)
        else:
            token_encoder = text_encoder.ensure_find(TokenEncoder)
            init_embedding = randn(token_encoder.embedding_dim)
        adapter.add_concept(self.placeholder_token, init_embedding)
        adapter.inject()


class TextualInversionLatentDiffusionConfig(FinetuneLatentDiffusionConfig):
    dataset: HuggingfaceDatasetConfig
    latent_diffusion: LatentDiffusionConfig
    textual_inversion: TextualInversionConfig

    def model_post_init(self, __context: Any) -> None:
        # Pydantic v2 does post init differently, so we need to override this method too.
        logger.info("Freezing models to train only the new embedding.")
        self.models["unet"].train = False
        self.models["text_encoder"].train = False
        self.models["lda"].train = False


class TextualInversionLatentDiffusionTrainer(LatentDiffusionTrainer[TextualInversionLatentDiffusionConfig]):
    def __init__(
        self,
        config: TextualInversionLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadTextualInversion(), SaveTextualInversion()))

    def load_dataset(self) -> Dataset[TextEmbeddingLatentsBatch]:
        return TextualInversionDataset(trainer=self)


class LoadTextualInversion(Callback[TextualInversionLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: TextualInversionLatentDiffusionTrainer) -> None:
        trainer.config.textual_inversion.apply_textual_inversion_to_target(text_encoder=trainer.text_encoder)


class SaveTextualInversion(Callback[TextualInversionLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: TextualInversionLatentDiffusionTrainer) -> None:
        embedding_extender = trainer.text_encoder.ensure_find(EmbeddingExtender)
        tensors = {trainer.config.textual_inversion.placeholder_token: embedding_extender.new_weight.squeeze(0)}

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors", tensors=tensors
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = TextualInversionLatentDiffusionConfig.load_from_toml(toml_path=config_path)
    trainer = TextualInversionLatentDiffusionTrainer(config=config)
    trainer.train()
