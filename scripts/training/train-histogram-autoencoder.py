from refiners.training_utils.trainers.histogram_auto_encoder import HistogramAutoEncoderTrainer, TrainHistogramAutoEncoderConfig

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1]
    config = TrainHistogramAutoEncoderConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = HistogramAutoEncoderTrainer(config=config)
    trainer.train()
