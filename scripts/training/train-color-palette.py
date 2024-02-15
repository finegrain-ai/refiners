from matplotlib.pyplot import hist
from refiners.training_utils.trainers.color_palette import ColorPaletteLatentDiffusionConfig, ColorPaletteLatentDiffusionTrainer
from refiners.training_utils.trainers.histogram import HistogramLatentDiffusionConfig, HistogramLatentDiffusionTrainer
from refiners.training_utils.trainers.histogram_auto_encoder import HistogramAutoEncoderTrainer, TrainHistogramAutoEncoderConfig
from refiners.training_utils.trainers.latent_diffusion import FinetuneLatentDiffusionBaseConfig


def adapt_config(config_path: str, config: FinetuneLatentDiffusionBaseConfig) -> FinetuneLatentDiffusionBaseConfig:
    if 'local' in config_path:
        config.training.batch_size = 1
        config.training.num_workers = min(4,config.training.num_workers)
        config.wandb.tags = config.wandb.tags + ['local']

        if 'unet' in config.models:
            config.models['unet'].gpu_index = 1
        
        if 'text_encoder' in config.models:
            config.models['text_encoder'].gpu_index = 0
        
        if 'lda' in config.models:
            config.models['lda'].gpu_index = 0
            
        if 'color_palette_encoder' in config.models:
            config.models['color_palette_encoder'].gpu_index = 1

        return config
    else:
        config.wandb.tags = config.wandb.tags + ['remote']
        return config

def train_histogram_auto_encoder(config_path: str) -> None:
    config = TrainHistogramAutoEncoderConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = HistogramAutoEncoderTrainer(config=adapt_config(config_path, config))
    trainer.train()

def train_color_palette(config_path: str) -> None:
    config = ColorPaletteLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = ColorPaletteLatentDiffusionTrainer(config=adapt_config(config_path, config))
    trainer.train()

def train_histogram(config_path: str) -> None:
    config = HistogramLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = HistogramLatentDiffusionTrainer(config=adapt_config(config_path, config))
    trainer.train()

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1]
    
    if 'histogram-autoencoder' in config_path:
        train_histogram_auto_encoder(config_path)
    elif 'color-palette' in config_path:
        train_color_palette(config_path)
    elif 'histogram' in config_path:
        train_histogram(config_path)  
    else:
        raise ValueError(f"Invalid config path: {config_path}")