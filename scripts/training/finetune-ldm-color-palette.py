from refiners.training_utils.color_palette import (
    ColorPaletteLatentDiffusionConfig,
    ColorPaletteLatentDiffusionTrainer
)

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = ColorPaletteLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = ColorPaletteLatentDiffusionTrainer(config=config)
    trainer.train()
