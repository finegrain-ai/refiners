from refiners.training_utils.latent_diffusion import FinetuneLatentDiffusionConfig, LatentDiffusionTrainer

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = FinetuneLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = LatentDiffusionTrainer(config=config)
    trainer.train()
