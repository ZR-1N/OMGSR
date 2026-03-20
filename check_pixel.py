import cv2
import numpy as np

# 随便挑一张你生成的图和对应的 GT
sr = cv2.imread("rgb_same_output/00000.png", cv2.IMREAD_GRAYSCALE)
gt = cv2.imread("dataset/train_rgb/CCPs/HQ/00000.png", cv2.IMREAD_GRAYSCALE)

print(f"【生成的 SR 图像】 背景最暗处: {sr.min()}, 最亮点: {sr.max()}, 平均亮度: {sr.mean():.2f}")
print(f"【真实的 GT 图像】 背景最暗处: {gt.min()}, 最亮点: {gt.max()}, 平均亮度: {gt.mean():.2f}")