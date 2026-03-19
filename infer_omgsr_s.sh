python infer/infer_omgsr_s.py \
    --input_image my_tests \
    --output_dir rgb_same_output \
    --sd_path stabilityai/stable-diffusion-2-1-base \
    --lora_path adapters/omgsr-s-512-adapter \
    --process_size 256 \
    --upscale 2 \
    --mid_timestep 273 \
    --align_method adain
