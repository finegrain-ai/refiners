from refiners.training_utils.histogram import HistogramLatentDiffusionConfig, HistogramLatentDiffusionTrainer

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = HistogramLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = HistogramLatentDiffusionTrainer(config=config)
    trainer.train()
