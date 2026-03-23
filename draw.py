import re
import os
import numpy as np
import matplotlib.pyplot as plt

# ========= 配置 =========
log_file = "train_v3.log"   # 改成你的路径
save_dir = "loss_plots_v3"
os.makedirs(save_dir, exist_ok=True)

loss_keys = [
    "loss_D_fake",
    "loss_D_real",
    "loss_Dv3D",
    "loss_L1",
    "loss_LRR",
    "loss_SSIM"
]

# ========= 工具函数 =========

def ema(y, alpha=0.95):
    """指数滑动平均"""
    if len(y) == 0:
        return y
    ema_y = []
    s = y[0]
    for v in y:
        s = alpha * s + (1 - alpha) * v
        ema_y.append(s)
    return ema_y

def downsample(x, y, step=10):
    """降采样"""
    return x[::step], y[::step]

# ========= 解析 log =========

loss_data = {k: [] for k in loss_keys}
steps = []

step_pattern = re.compile(r"Steps:\s*\d+%.*?\|\s*(\d+)/(\d+)")
loss_patterns = {k: re.compile(rf"{k}=([0-9\.eE+-]+)") for k in loss_keys}

with open(log_file, "r") as f:
    for line in f:
        if "Steps:" not in line:
            continue

        step_match = step_pattern.search(line)
        if not step_match:
            continue

        step = int(step_match.group(1))
        steps.append(step)

        for k in loss_keys:
            match = loss_patterns[k].search(line)
            if match:
                loss_data[k].append(float(match.group(1)))
            else:
                loss_data[k].append(None)

# ========= 画图 =========

for k in loss_keys:
    values = loss_data[k]

    # 去掉 None
    valid_steps = []
    valid_values = []
    for s, v in zip(steps, values):
        if v is not None:
            valid_steps.append(s)
            valid_values.append(v)

    if len(valid_values) == 0:
        continue

    # 1️⃣ 降采样
    ds_steps, ds_values = downsample(valid_steps, valid_values, step=10)

    # 2️⃣ 平滑
    smooth_values = ema(ds_values, alpha=0.95)

    # ========= 画单独图 =========
    plt.figure(figsize=(6,4))
    plt.plot(ds_steps, smooth_values)

    plt.xlabel("Step")
    plt.ylabel(k)
    plt.title(f"{k} Curve")
    plt.grid()

    # 保存
    save_path = os.path.join(save_dir, f"{k}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")

print("Done.")