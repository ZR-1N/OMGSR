import numpy as np
import cv2
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

NPZ_PATH = 'dataset/train/ER/my_training_data.npz'

def linear_transform(img_true, img_test):
    # 【升级 1】内部强制使用 float64 双精度运算，彻底杜绝平方溢出
    img_true = img_true.astype(np.float64)
    img_test = img_test.astype(np.float64)

    mean_true = np.mean(img_true)
    mean_test = np.mean(img_test)

    denominator = np.mean(np.square(img_test - mean_test)) + 1e-8
    b = np.mean((img_test - mean_test) * (img_true - mean_true)) / denominator
    a = mean_true - b * mean_test

    img_test_transform = a + b * img_test
    # 算完之后，防止出现负数底噪，切除负数，并转回 float32 给 skimage
    return np.clip(img_test_transform, 0, None).astype(np.float32)

def main():
    print(f"正在加载高精度浮点数据集: {NPZ_PATH}")
    data = np.load(NPZ_PATH)
    
    X = data['X'].astype(np.float32)
    Y = data['Y'].astype(np.float32)
    num_samples = X.shape[0]
    
    # 【升级 2】核心切除术：用 99.99% 屏蔽天文数字死像素
    print("正在扫描并切除数据集异常值...")
    y_max_safe = np.percentile(Y, 99.99)
    x_max_safe = np.percentile(X, 99.99)
    y_min_safe = np.percentile(Y, 0.01)
    
    # 将所有的天文数字强行削平到安全水位
    X = np.clip(X, 0, x_max_safe)
    Y = np.clip(Y, 0, y_max_safe)
    
    print(f"数据清洗完毕，安全物理上限设定为: {y_max_safe:.4f}")
    
    psnr_list = []
    ssim_list = []
    
    print(f"总计 {num_samples} 个样本，开始 Nature Comm 标准 Bicubic 测试...")
    
    for i in tqdm(range(num_samples)):
        lq = np.squeeze(X[i])
        gt = np.squeeze(Y[i])
        
        h_gt, w_gt = gt.shape
        bicubic_raw = cv2.resize(lq, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
        
        # 核心：使用无敌的 float64 线性对齐
        bicubic_aligned = linear_transform(gt, bicubic_raw)
        
        # 动态范围设定：当前单张 GT 的 max - min
        d_range = float(gt.max() - gt.min())
        if d_range <= 1e-8:
            d_range = 1.0
            
        psnr_val = compare_psnr(gt, bicubic_aligned, data_range=d_range)
        ssim_val = compare_ssim(gt, bicubic_aligned, data_range=d_range, 
                                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        
        # 二次保险：防止个别极端纯黑图片产出 nan 污染均值
        if not math.isnan(psnr_val) and not math.isinf(psnr_val):
            psnr_list.append(psnr_val)
        if not math.isnan(ssim_val) and not math.isinf(ssim_val):
            ssim_list.append(ssim_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print("\n" + "="*50)
    print(f"【Nature Comm 官方标准】 线性对齐 Bicubic Baseline")
    print(f"PSNR: {avg_psnr:.4f}  |  SSIM: {avg_ssim:.4f}")
    print(f"有效评估样本数: {len(ssim_list)} / {num_samples}")
    print("="*50)

if __name__ == "__main__":
    main()