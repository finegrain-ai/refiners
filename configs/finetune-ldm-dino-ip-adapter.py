script = "finetune-ldm-palette-adapter.py" # not used for now

[wandb]
mode = "offline"                         # "online", "offline", "disabled"
entity = "acme"
project = "finetune-ldm-palette-adapter"

[models]
unet = { checkpoint = "weights/unet.safetensors", train = false }
text_encoder = { checkpoint = "weights/clip.safetensors", train = false }
lda = { checkpoint = "weights/lda.safetensors", train = false }
color_encoder = { train = true }
adapter = { train = true }

[ldm]
unconditional_sampling_probability = 0.05
offset_noise = 0.1

[adapter]
scale = 1.0

[adapter.color_encoder]
dim_sinusoids = 64
dim_hidden = 256

[training]
duration = "2000:step"
seed = 0
gpu_index = 0
batch_size = 2
gradient_accumulation = "1:step"
evaluation_interval = "250:step"
evaluation_seed = 1

[optimizer]
optimizer = "AdamW"  # "SGD", "Adam", "AdamW", "AdamW8bit", "Lion8bit"
learning_rate = 5e-4
betas = [0.9, 0.999]
eps = 1e-8
weight_decay = 1e-2

[scheduler]
scheduler_type = "ConstantLR"
update_interval = "1:step"

[dropout]
dropout_probability = 0
use_gyro_dropout = false

[dataset]
hf_repo = "1aurent/unsplash-lite-palette"
revision = "main"
split = "train"
horizontal_flip_probability = 0.5
random_crop_size = 384                    # (256 + 128, quick math)
resize_image_min_size = 384
resize_image_max_size = 416               # (384 + 32, quick math)

[checkpointing]
# save_folder = "/path/to/ckpts"
save_interval = "250:step"

[test_ldm]
num_inference_steps = 30
prompts = [
    "The Starry Night, Vincent van Gogh",
    "Girl with a Pearl Earring, Johannes Vermeer, low poly",
    "a cute cat, pointillism",
]
palettes = [
    [[220, 20, 60]], # crimson
    [[127, 0, 255]], # violet
    [[0, 191, 255]], # deepskyblue
]