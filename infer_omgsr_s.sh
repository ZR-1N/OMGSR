python infer/infer_omgsr_s.py \
    --input_image dataset/test_rgb/ER/LR \
    --output_dir infer_results/rgb_test_ER \
    --sd_path stabilityai/stable-diffusion-2-1-base \
    --lora_path omgsr_trainings/ER/weight-41000 \
    --process_size 256 \
    --upscale 2 \
    --mid_timestep 276 \
    --align_method adain
