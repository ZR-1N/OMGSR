import os
import cv2
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

# 配置你的文件夹路径
SR_DIR = "infer_results/rgb_same_output_v3"  # 你的 49000 步模型输出的文件夹
GT_DIR = "dataset/train_rgb/CCPs/HQ"
SAVE_FILE = "results_omgsr_nature_standard.txt"
SCALE = 2

# Nature Comm 官方的线性对齐函数（双精度防溢出版）
def linear_transform(img_true, img_test):
    img_true = img_true.astype(np.float64)
    img_test = img_test.astype(np.float64)

    mean_true = np.mean(img_true)
    mean_test = np.mean(img_test)

    denominator = np.mean(np.square(img_test - mean_test)) + 1e-8
    b = np.mean((img_test - mean_test) * (img_true - mean_true)) / denominator
    a = mean_true - b * mean_test

    img_test_transform = a + b * img_test
    return np.clip(img_test_transform, 0, None).astype(np.float32)

def main():
    sr_files = sorted([f for f in os.listdir(SR_DIR) if f.endswith(('.png', '.tif', '.jpg'))])
    
    psnr_list = []
    ssim_list = []

    print(f"总计找到 {len(sr_files)} 个生成图像，开始执行 Nature Comm 标准评测...")

    with open(SAVE_FILE, "w") as f:
        for name in tqdm(sr_files):
            sr_path = os.path.join(SR_DIR, name)
            gt_path = os.path.join(GT_DIR, name)

            if not os.path.exists(gt_path):
                continue

            # 1. 读取并转换为灰度浮点数 [0, 1] 空间
            sr_bgr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            gt_gray = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if sr_bgr is None or gt_gray is None:
                continue
                
            sr_gray = np.mean(sr_bgr, axis=2)
            
            if SCALE > 0:
                sr_gray = sr_gray[SCALE:-SCALE, SCALE:-SCALE]
                gt_gray = gt_gray[SCALE:-SCALE, SCALE:-SCALE]

            sr_float = sr_gray.astype(np.float32) / 255.0
            gt_float = gt_gray.astype(np.float32) / 255.0

            # 2. 核心大招：卸下 VAE 的包袱，线性对齐亮度！
            sr_aligned = linear_transform(gt_float, sr_float)

            # 3. 计算指标
            d_range = float(gt_float.max() - gt_float.min())
            if d_range <= 1e-8:
                d_range = 1.0

            try:
                psnr_val = compare_psnr(gt_float, sr_aligned, data_range=d_range)
                ssim_val = compare_ssim(gt_float, sr_aligned, data_range=d_range, 
                                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                
                if not math.isnan(psnr_val) and not math.isinf(psnr_val):
                    psnr_list.append(psnr_val)
                if not math.isnan(ssim_val) and not math.isinf(ssim_val):
                    ssim_list.append(ssim_val)
                    
            except Exception:
                pass

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        summary = f"\n{'='*50}\n【OMGSR 模型成绩单】 线性对齐评测\nPSNR: {avg_psnr:.4f}  |  SSIM: {avg_ssim:.4f}\n有效样本: {len(ssim_list)}\n{'='*50}"
        print(summary)
        f.write(summary + "\n")

if __name__ == "__main__":
    main()