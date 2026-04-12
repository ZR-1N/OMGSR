import os
import cv2
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

# =======================
# 路径配置区
# =======================
GT_DIR = "dataset/test/ER/GT"          # 高清 TIF GT
LR_DIR = "dataset/test/ER/LR"          # 宽场 TIF LR
SR_DIR = "infer_results/SwinIRER"             # 你的模型输出 PNG 文件夹 (请确保这里是你6000步或最新的输出文件夹)
SAVE_PLOT = "FRC/SwinIRER.png"
SCALE = 2

# FRC 计算阈值 (冷冻电镜和超分领域常用的 1/7 阈值)
THRESHOLD = 0.143

# =======================
# 核心算法区
# =======================
def linear_transform(img_true, img_test):
    """Nature Comm 标准：线性亮度对齐"""
    img_true = img_true.astype(np.float64)
    img_test = img_test.astype(np.float64)
    mean_true = np.mean(img_true)
    mean_test = np.mean(img_test)
    denominator = np.mean(np.square(img_test - mean_test)) + 1e-8
    b = np.mean((img_test - mean_test) * (img_true - mean_true)) / denominator
    a = mean_true - b * mean_test
    img_test_transform = a + b * img_test
    return np.clip(img_test_transform, 0, img_true.max()).astype(np.float32)

def read_scientific_tif(path):
    """读取 TIF"""
    img = tifffile.imread(path)
    if img.ndim == 3: img = np.mean(img, axis=-1)
    if img.dtype == np.uint16: return img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8: return img.astype(np.float32) / 255.0
    else: return np.clip(img.astype(np.float32), 0, 1)

def calculate_frc_curve(img1, img2):
    """
    计算两张图像的 FRC (Fourier Ring Correlation) 曲线。
    返回: 半径 bins (空间频率), FRC 值
    """
    assert img1.shape == img2.shape, "Images must have the same shape"
    h, w = img1.shape
    
    # 1. 添加 Hanning 窗，减少 FFT 的边缘伪影 (学术界计算 FRC 的标配)
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window = np.outer(window_y, window_x)
    i1 = img1 * window
    i2 = img2 * window
    
    # 2. 傅里叶变换
    f1 = np.fft.fftshift(np.fft.fft2(i1))
    f2 = np.fft.fftshift(np.fft.fft2(i2))
    
    # 3. 计算互功率谱和自功率谱
    c = np.real(f1 * np.conjugate(f2))
    p1 = np.abs(f1)**2
    p2 = np.abs(f2)**2
    
    # 4. 创建极坐标网格 (环)
    center = (int(h/2), int(w/2))
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = np.round(r).astype(int)
    
    max_radius = int(min(h, w) / 2)
    frc_curve = np.zeros(max_radius)
    
    # 5. 积分每一环 (Ring)
    for i in range(max_radius):
        mask = (r == i)
        num = np.sum(c[mask])
        den = np.sqrt(np.sum(p1[mask]) * np.sum(p2[mask]))
        if den > 0:
            frc_curve[i] = num / den
            
    # 频率坐标 (归一化到 0 ~ 1 Nyquist)
    spatial_freqs = np.linspace(0, 1, max_radius)
    return spatial_freqs, frc_curve

def find_resolution(freqs, frc_curve, threshold=THRESHOLD):
    """找到 FRC 曲线跌破阈值的空间频率"""
    for i in range(len(frc_curve)):
        if frc_curve[i] < threshold:
            # 线性插值找到精确过零点
            if i == 0: return freqs[0]
            x1, x2 = freqs[i-1], freqs[i]
            y1, y2 = frc_curve[i-1], frc_curve[i]
            exact_freq = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            return exact_freq
    return freqs[-1]

def main():
    sr_files = sorted([f for f in os.listdir(SR_DIR) if f.endswith(('.png', '.jpg'))])
    
    all_frc_bicubic = []
    all_frc_omgsr = []
    
    print(f"✅ 开始计算 FRC (Fourier Ring Correlation)，共 {len(sr_files)} 个样本...")
    
    for name in tqdm(sr_files):
        sr_path = os.path.join(SR_DIR, name)
        base_name = os.path.splitext(name)[0]
        core_name = re.sub(r'(_LR|_GT|_SR|_output|output)$', '', base_name, flags=re.IGNORECASE)

        # 智能匹配
        gt_cand = [f"{core_name}_GT.tif", f"{core_name}.tif", f"{base_name}_GT.tif"]
        lr_cand = [f"{core_name}_LR.tif", f"{core_name}.tif", f"{base_name}_LR.tif"]
        
        gt_path = next((os.path.join(GT_DIR, c) for c in gt_cand if os.path.exists(os.path.join(GT_DIR, c))), None)
        lr_path = next((os.path.join(LR_DIR, c) for c in lr_cand if os.path.exists(os.path.join(LR_DIR, c))), None)
        
        if not gt_path or not lr_path: continue

        # 加载与裁切
        gt_float = read_scientific_tif(gt_path)
        lr_float = read_scientific_tif(lr_path)
        sr_bgr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
        if sr_bgr is None: continue
        sr_float = (np.mean(sr_bgr, axis=2) / 255.0).astype(np.float32)

        if SCALE > 0:
            gt_float = gt_float[SCALE:-SCALE, SCALE:-SCALE]
            lr_float = lr_float[SCALE:-SCALE, SCALE:-SCALE]
            sr_float = sr_float[SCALE:-SCALE, SCALE:-SCALE]

        # Bicubic 放缩
        h_gt, w_gt = gt_float.shape
        bicubic_float = cv2.resize(lr_float, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)

        # 线性对齐亮度 (至关重要)
        bicubic_aligned = linear_transform(gt_float, bicubic_float)
        sr_aligned = linear_transform(gt_float, sr_float)

        # 计算单张 FRC 曲线
        freqs, frc_bicubic = calculate_frc_curve(gt_float, bicubic_aligned)
        _, frc_omgsr = calculate_frc_curve(gt_float, sr_aligned)
        
        all_frc_bicubic.append(frc_bicubic)
        all_frc_omgsr.append(frc_omgsr)

    # 计算平均 FRC 曲线
    avg_frc_bicubic = np.mean(all_frc_bicubic, axis=0)
    avg_frc_omgsr = np.mean(all_frc_omgsr, axis=0)
    
    # 寻找物理分辨率截止点 (截止频率越高越好)
    res_bicubic = find_resolution(freqs, avg_frc_bicubic)
    res_omgsr = find_resolution(freqs, avg_frc_omgsr)

    # ==========================
    # 绘制可直接发论文的精美折线图
    # ==========================
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(freqs, avg_frc_bicubic, label=f'Bicubic (Cutoff: {res_bicubic:.3f})', color='#1f77b4', linewidth=2)
    plt.plot(freqs, avg_frc_omgsr, label=f'OMGSR [Ours] (Cutoff: {res_omgsr:.3f})', color='#d62728', linewidth=2)
    
    # 画阈值线
    plt.axhline(y=THRESHOLD, color='gray', linestyle='--', alpha=0.7, label=f'Threshold = {THRESHOLD:.3f}')
    
    # 标注 cutoff 位置
    plt.scatter([res_bicubic, res_omgsr], [THRESHOLD, THRESHOLD], color=['#1f77b4', '#d62728'], zorder=5)
    
    plt.title('Fourier Ring Correlation (FRC) Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Spatial Frequency (Nyquist)', fontsize=12)
    plt.ylabel('FRC', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1.1)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(SAVE_PLOT)
    print(f"\n✅ FRC 曲线已保存为: {SAVE_PLOT}")
    
    print("\n" + "="*50)
    print("🔬 物理分辨率评估 (Cutoff Frequency - 越高越好)")
    print(f"Bicubic Baseline:  {res_bicubic:.4f} Nyquist")
    print(f"OMGSR (Ours)    :  {res_omgsr:.4f} Nyquist 🚀")
    print(f"提升幅度        :  +{(res_omgsr - res_bicubic)/res_bicubic*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()