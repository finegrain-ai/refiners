#!/bin/sh
python /home/isamu/refiners/scripts/generate_benchmark_images.py --checkpoint_path="/home/isamu/refiners/tests/weights/ip-adapter-plus_sd15.safetensors" --generation_path="/home/isamu/generations/default" --clip_image_encoder
python /home/isamu/refiners/scripts/generate_benchmark_images.py --checkpoint_path="/home/isamu/checkpoints_32/step110000.safetensors" --generation_path="/home/isamu/generations/fixed_uncond_init_gen_110000_steps" --use_timestep_embedding --image_embedding_div_factor=3
python /home/isamu/refiners/scripts/generate_benchmark_images.py --checkpoint_path="/home/isamu/checkpoints_rescaler_fixed/step110000.safetensors" --generation_path="/home/isamu/generations/init_gen_110000_steps" --use_timestep_embedding
python /home/isamu/refiners/scripts/generate_benchmark_images.py --checkpoint_path="/home/isamu/checkpoints_rescaler_fixed/ step500000.safetensors" --generation_path="/home/isamu/generations/init_gen_500000_steps" --use_timestep_embedding

