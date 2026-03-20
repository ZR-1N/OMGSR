python infer/infer_omgsr_s.py \
    --input_image dataset/train_rgb/CCPs/LQ \
    --output_dir rgb_same_output \
    --sd_path stabilityai/stable-diffusion-2-1-base \
    --lora_path omgsr_trainings/rgb_same_version/weight-6000 \
    --process_size 256 \
    --upscale 2 \
    --mid_timestep 254 \
    --align_method adain
