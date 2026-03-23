python infer/infer_omgsr_s.py \
    --input_image dataset/test_rgb/CCPs/LQ \
    --output_dir infer_results/rgb_test_zero \
    --sd_path stabilityai/stable-diffusion-2-1-base \
    --lora_path adapters/omgsr-s-512-adapter \
    --process_size 256 \
    --upscale 2 \
    --mid_timestep 254 \
    --align_method adain
