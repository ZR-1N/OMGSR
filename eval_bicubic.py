import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# =======================
# 配置路径
LQ_DIR = "dataset/train_rgb/CCPs/LQ"  # 你的 Raw/低分辨率输入文件夹
GT_DIR = "dataset/train_rgb/CCPs/HQ"  # 你的 真实高清 GT 文件夹
SAVE_FILE = "results_bicubic_baseline.txt"
SCALE = 2  # 为了和你的 SR 评测绝对公平，保持相同的边缘裁切
# =======================

def main():
    lq_files = sorted([f for f in os.listdir(LQ_DIR) if f.endswith(('.png', '.tif', '.jpg'))])

    psnr_list = []
    ssim_list = []

    print("开始进行 Bicubic 基线测试...")

    with open(SAVE_FILE, "w") as f:
        for name in lq_files:
            lq_path = os.path.join(LQ_DIR, name)
            gt_path = os.path.join(GT_DIR, name)

            if not os.path.exists(gt_path):
                continue

            # 1. 以灰度图模式读取 LQ 和 GT
            lq_gray = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE)
            gt_gray = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if lq_gray is None or gt_gray is None:
                continue

            # 2. 核心：使用双三次插值 (Bicubic) 将 LQ 放大到 GT 的尺寸
            h, w = gt_gray.shape
            # 注意 cv2.resize 的 dsize 参数是 (width, height)
            bicubic_gray = cv2.resize(lq_gray, (w, h), interpolation=cv2.INTER_CUBIC)

            # 3. 边缘裁切 (严格保持与扩散模型评测一致)
            if SCALE > 0:
                bicubic_gray = bicubic_gray[SCALE:-SCALE, SCALE:-SCALE]
                gt_gray = gt_gray[SCALE:-SCALE, SCALE:-SCALE]

            # 4. 转换到 [0, 1] 浮点数
            # Bicubic 不会像 VAE 那样引入全局底噪，所以不需要 Min-Max 归一化，直接除以 255 即可
            bicubic_float = bicubic_gray.astype(np.float32) / 255.0
            gt_float = gt_gray.astype(np.float32) / 255.0

            # 5. 计算指标
            psnr_val = compare_psnr(gt_float, bicubic_float, data_range=1.0)
            ssim_val = compare_ssim(gt_float, bicubic_float, data_range=1.0, gaussian_weights=True)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

        # 汇总统计
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        summary = f"\n===========================\nBicubic Baseline | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}\nTotal Images: {len(psnr_list)}"
        print(summary)
        f.write(summary + "\n")

if __name__ == "__main__":
    main()