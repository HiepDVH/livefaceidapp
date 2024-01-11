import os

import cv2
import numpy as np
import torch
from lpips import LPIPS
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_metrics(original_img, enhanced_img):
    # Chuyển đổi ảnh từ [0, 255] sang [0, 1]
    original_img = original_img.astype(np.float32) / 255.0
    enhanced_img = enhanced_img.astype(np.float32) / 255.0
    original_img.resize(enhanced_img.shape)
    # PSNR
    psnr_value = psnr(original_img, enhanced_img, data_range=1.0)

    # SSIM
    ssim_value, _ = ssim(original_img, enhanced_img, full=True, data_range=1.0)

    # Entropy
    entropy_original = entropy(original_img.flatten())
    entropy_enhanced = entropy(enhanced_img.flatten())

    # Standard Deviation
    std_dev_original = np.std(original_img)
    std_dev_enhanced = np.std(enhanced_img)
    original_tensor = torch.from_numpy(original_img)
    enhanced_tensor = torch.from_numpy(enhanced_img)
    lpips_value = lpips_model(original_tensor, enhanced_tensor).item()
    return (
        psnr_value,
        ssim_value,
        entropy_original,
        entropy_enhanced,
        std_dev_original,
        std_dev_enhanced,
        lpips_value,
    )


if __name__ == '__main__':
    lpips_model = LPIPS(net='vgg')
    original_image_folder = (
        '/home/hiepdvh/low-light-recognition-trial/Zero-DCE++/lol_dataset/eval15/low'
    )
    enhanced_image_folder = (
        '/home/hiepdvh/low-light-recognition-trial/Zero-DCE++/lol_dataset/eval15/high'
    )

    images = os.listdir(original_image_folder)
    results = ''
    for image in images:
        original_image_path = os.path.join(original_image_folder, image)
        enhanced_image_path = os.path.join(enhanced_image_folder, image)
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        enhanced_img = cv2.imread(enhanced_image_path, cv2.IMREAD_GRAYSCALE)

        (
            psnr_value,
            ssim_value,
            entropy_original,
            entropy_enhanced,
            std_dev_original,
            std_dev_enhanced,
            lpips_value,
        ) = calculate_metrics(original_img, enhanced_img)
        result = f'{image}\npsnr: {psnr_value}, ssim: {ssim_value}, entropy: {entropy_enhanced}, std: {std_dev_enhanced}, lpips: {lpips_value}'
        results = results + result + '\n'
    print(results)
