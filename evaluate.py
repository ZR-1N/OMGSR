import os
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# =======================
# 配置路径
SR_DIR = "rgb_same_output"
GT_DIR = "dataset/train_rgb/CCPs/HQ"
SAVE_FILE = "results_bio_standard.txt"
SCALE = 2  # 如果生成的图像边缘有伪影，设为2裁掉边缘；否则设为0
# =======================

def main():
    sr_files = sorted([f for f in os.listdir(SR_DIR) if f.endswith(('.png', '.tif', '.jpg'))])

    psnr_list = []
    ssim_list = []

    with open(SAVE_FILE, "w") as f:
        for name in sr_files:
            sr_path = os.path.join(SR_DIR, name)
            gt_path = os.path.join(GT_DIR, name)

            if not os.path.exists(gt_path):
                print(f"找不到对应的 GT 图像: {name}")
                continue

            # ---------------------------------------------------------
            # 1. 读取生成的伪 RGB 图像 (SR)
            # ---------------------------------------------------------
            sr_bgr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            if sr_bgr is None:
                continue
            
            # 核心修正：求三个通道的平均值，变回纯粹的生物信号强度 [H, W]
            sr_gray = np.mean(sr_bgr, axis=2)
            
            # ---------------------------------------------------------
            # 2. 读取真实的 GT 图像
            # ---------------------------------------------------------
            # GT 原本就是单通道转出来的，我们直接以灰度模式读取即可 [H, W]
            gt_gray = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # ---------------------------------------------------------
            # 3. 边缘裁切 (SR常规操作)
            # ---------------------------------------------------------
            if SCALE > 0:
                sr_gray = sr_gray[SCALE:-SCALE, SCALE:-SCALE]
                gt_gray = gt_gray[SCALE:-SCALE, SCALE:-SCALE]

            # ---------------------------------------------------------
            # 4. 转换到 [0, 1] 浮点数，模拟物理光子强度
            # ---------------------------------------------------------
            sr_float = sr_gray.astype(np.float32) / 255.0
            gt_float = gt_gray.astype(np.float32) / 255.0

            # ---------------------------------------------------------
            # 5. 使用 skimage 官方接口计算 (严格标准)
            # ---------------------------------------------------------
            # data_range=1.0 极其重要，因为我们的像素值在 0~1 之间
            psnr_val = compare_psnr(gt_float, sr_float, data_range=1.0)
            
            # 生物图像常使用 gaussian_weights=True 匹配显微镜的光学点扩散(PSF)特性
            ssim_val = compare_ssim(gt_float, sr_float, data_range=1.0, gaussian_weights=True)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            line = f"{name} | PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}"
            print(line)
            f.write(line + "\n")

        # 汇总统计
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        summary = f"\n===========================\nFinal Average | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}\nTotal Images: {len(psnr_list)}"
        print(summary)
        f.write(summary + "\n")

if __name__ == "__main__":
    main()