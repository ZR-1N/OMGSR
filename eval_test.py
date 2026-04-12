import os
import cv2
import math
import re
import numpy as np
import tifffile
import torch
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse
from tqdm import tqdm

# =======================
# 路径配置区
# =======================
GT_DIR = "dataset/test/ER/GT"          # 高清 TIF GT 文件夹
LR_DIR = "dataset/test/ER/LR"          # 宽场 TIF LR 文件夹
SR_DIR = "infer_results/SwinIRER"             # 你的模型输出 PNG 文件夹
SAVE_FILE = "results_SwinIRER.txt"
SCALE = 2                                # 边缘裁切像素

# =======================
# 核心算法区
# =======================
def linear_transform(img_true, img_test):
    """Nature Comm 标准：线性亮度对齐 (双精度计算 + 物理上限保护)"""
    img_true = img_true.astype(np.float64)
    img_test = img_test.astype(np.float64)

    mean_true = np.mean(img_true)
    mean_test = np.mean(img_test)

    denominator = np.mean(np.square(img_test - mean_test)) + 1e-8
    b = np.mean((img_test - mean_test) * (img_true - mean_true)) / denominator
    a = mean_true - b * mean_test

    img_test_transform = a + b * img_test
    return np.clip(img_test_transform, 0, img_true.max()).astype(np.float32)

def calc_zncc(img_true, img_test):
    """零均值归一化交叉相关"""
    mu_true = np.mean(img_true)
    mu_test = np.mean(img_test)
    sigma_true = np.std(img_true)
    sigma_test = np.std(img_test)
    if sigma_true == 0 or sigma_test == 0:
        return 0.0
    return np.mean((img_true - mu_true) * (img_test - mu_test) / (sigma_true * sigma_test))

def read_scientific_tif(path):
    """强制读取 TIF 并转换为 [0,1] 的单通道浮点数"""
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = np.mean(img, axis=-1)
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    else:
        return np.clip(img.astype(np.float32), 0, 1)

def main():
    sr_files = sorted([f for f in os.listdir(SR_DIR) if f.endswith(('.png', '.jpg'))])
    
    metrics = {
        'Bicubic': {'PSNR': [], 'SSIM': [], 'MS_SSIM': [], 'NRMSE': [], 'ZNCC': []},
        'OMGSR':   {'PSNR': [], 'SSIM': [], 'MS_SSIM': [], 'NRMSE': [], 'ZNCC': []}
    }

    print(f"✅ 开始多维基准评测，在 SR_DIR 中找到 {len(sr_files)} 个生成图像...")
    matched_count = 0

    with open(SAVE_FILE, "w") as f:
        for name in tqdm(sr_files):
            sr_path = os.path.join(SR_DIR, name)
            
            # --- 智能文件名匹配 ---
            base_name = os.path.splitext(name)[0]
            # 用正则剔除 SR 文件名中可能带有的后缀，提取核心编号 (例如 im1_LR -> im1)
            core_name = re.sub(r'(_LR|_GT|_SR|_output|output)$', '', base_name, flags=re.IGNORECASE)

            # 构建多种可能存在的 TIF 命名规则
            gt_candidates = [f"{core_name}_GT.tif", f"{core_name}.tif", f"{base_name}_GT.tif"]
            lr_candidates = [f"{core_name}_LR.tif", f"{core_name}.tif", f"{base_name}_LR.tif"]

            gt_path, lr_path = None, None
            for cand in gt_candidates:
                if os.path.exists(os.path.join(GT_DIR, cand)):
                    gt_path = os.path.join(GT_DIR, cand)
                    break
            for cand in lr_candidates:
                if os.path.exists(os.path.join(LR_DIR, cand)):
                    lr_path = os.path.join(LR_DIR, cand)
                    break

            if not gt_path or not lr_path:
                # 打印日志帮助你定位是哪个文件没匹配上
                # print(f"⚠️ 跳过 {name}: 找不到对应的 .tif 格式 GT 或 LR 文件。")
                continue
            
            matched_count += 1

            # ---------------------------------------------------------
            # 1. 数据加载与预处理 (只读取 TIF)
            # ---------------------------------------------------------
            gt_float = read_scientific_tif(gt_path)
            lr_float = read_scientific_tif(lr_path)

            sr_bgr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            if sr_bgr is None: continue
            sr_float = (np.mean(sr_bgr, axis=2) / 255.0).astype(np.float32)

            # ---------------------------------------------------------
            # 2. 边缘裁切与 Bicubic 放大
            # ---------------------------------------------------------
            if SCALE > 0:
                gt_float = gt_float[SCALE:-SCALE, SCALE:-SCALE]
                lr_float = lr_float[SCALE:-SCALE, SCALE:-SCALE]
                sr_float = sr_float[SCALE:-SCALE, SCALE:-SCALE]

            h_gt, w_gt = gt_float.shape
            bicubic_float = cv2.resize(lr_float, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)

            # ---------------------------------------------------------
            # 3. 核心：Nature Comm 线性对齐
            # ---------------------------------------------------------
            bicubic_aligned = linear_transform(gt_float, bicubic_float)
            sr_aligned = linear_transform(gt_float, sr_float)

            # ---------------------------------------------------------
            # 4. 计算指标
            # ---------------------------------------------------------
            d_range = float(gt_float.max() - gt_float.min())
            if d_range <= 1e-8: d_range = 1.0

            gt_t = torch.from_numpy(gt_float).unsqueeze(0).unsqueeze(0).float()
            bicubic_t = torch.from_numpy(bicubic_aligned).unsqueeze(0).unsqueeze(0).float()
            sr_t = torch.from_numpy(sr_aligned).unsqueeze(0).unsqueeze(0).float()

            for method, img_aligned, img_t in zip(['Bicubic', 'OMGSR'], 
                                                  [bicubic_aligned, sr_aligned], 
                                                  [bicubic_t, sr_t]):
                try:
                    psnr_val = compare_psnr(gt_float, img_aligned, data_range=d_range)
                    ssim_val = compare_ssim(gt_float, img_aligned, data_range=d_range, 
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    nrmse_val = compare_nrmse(gt_float, img_aligned, normalization='euclidean')
                    zncc_val = calc_zncc(gt_float, img_aligned)

                    try:
                        ms_ssim_val = ms_ssim(gt_t, img_t, data_range=d_range, size_average=True).item()
                    except:
                        ms_ssim_val = ssim_val 

                    if not (math.isnan(psnr_val) or math.isinf(psnr_val)):
                        metrics[method]['PSNR'].append(psnr_val)
                        metrics[method]['SSIM'].append(ssim_val)
                        metrics[method]['MS_SSIM'].append(ms_ssim_val)
                        metrics[method]['NRMSE'].append(nrmse_val)
                        metrics[method]['ZNCC'].append(zncc_val)
                except Exception:
                    pass

        # ---------------------------------------------------------
        # 5. 生成最终报告
        # ---------------------------------------------------------
        report = f"\n{'='*60}\n"
        report += f"🔬 BioSR 终极对冲测试报告 (Nature Comm 标准)\n"
        report += f"有效匹配并测试样本: {matched_count}\n"
        report += f"{'-'*60}\n"
        report += f"{'Metric':<10} | {'Bicubic Baseline':<20} | {'OMGSR (Ours)':<20}\n"
        report += f"{'-'*60}\n"
        
        if matched_count > 0:
            for m in ['PSNR', 'SSIM', 'MS_SSIM', 'ZNCC', 'NRMSE']:
                if len(metrics['Bicubic'][m]) > 0 and len(metrics['OMGSR'][m]) > 0:
                    bic_avg = np.mean(metrics['Bicubic'][m])
                    omg_avg = np.mean(metrics['OMGSR'][m])
                    better_omg = omg_avg < bic_avg if m == 'NRMSE' else omg_avg > bic_avg
                    indicator = "🚀" if better_omg else "⚠️"
                    report += f"{m:<10} | {bic_avg:<20.4f} | {omg_avg:<15.4f} {indicator}\n"
                else:
                    report += f"{m:<10} | {'N/A':<20} | {'N/A':<15}\n"
        else:
            report += "🚨 匹配失败！请检查文件名格式是否如 im1_LR.png, im1_GT.tif 形式。\n"
            
        report += f"{'='*60}\n"
        
        print(report)
        f.write(report)

if __name__ == "__main__":
    main()