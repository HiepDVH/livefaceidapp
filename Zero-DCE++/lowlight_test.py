import os
import time

import cv2
import model
import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image


def lowlight(data_lowlight):
    start = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 12

    data_lowlight = data_lowlight.convert('RGB')

    data_lowlight = np.asarray(data_lowlight) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))
    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)

    end_time = time.time() - start

    return enhanced_image


def lowlight_frame(frame, DCE_net, scale_factor):
    scale_factor = 12

    frame = frame / 255.0  # Chuẩn hóa frame từ [0, 255] thành [0, 1]

    # Chuyển frame thành tensor và thực hiện các bước xử lý
    frame = torch.from_numpy(frame).float().cuda()

    h = (frame.shape[0] // scale_factor) * scale_factor
    w = (frame.shape[1] // scale_factor) * scale_factor
    frame = frame[0:h, 0:w, :]
    frame = frame.permute(2, 0, 1)
    frame = frame.unsqueeze(0)

    # Thực hiện xử lí bằng mô hình DCE_net
    enhanced_frame, params_maps = DCE_net(frame)

    enhanced_frame = (
        enhanced_frame[0].permute(1, 2, 0).cpu().numpy() * 255.0
    )  # Chuyển kết quả về [0, 255]

    return enhanced_frame


def adjust_brightness_cv2(frame, brightness_factor=1.5):
    enhanced_img = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)

    return enhanced_img


def process_video(input_path, output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    scale_factor = 12

    # Load mô hình DCE_net
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(
        torch.load(
            'first-train/Epoch99.pth',
        )
    )

    # Mở video input
    cap = cv2.VideoCapture(input_path)

    # Lấy thông tin về video (số frame mỗi giây, kích thước frame)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tạo video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (1920, 1080))
            enhanced_frame = lowlight_frame(resized_frame, DCE_net, scale_factor)
            # enhanced_frame = adjust_brightness_cv2(resized_frame, brightness_factor=4)
            out.write(enhanced_frame.astype(np.uint8))

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_low_light_images = os.listdir(
        '/home/hiepdvh/low-light-recognition-trial/Zero-DCE++/lol_dataset/eval15/low'
    )

    for low_light_image in test_low_light_images:
        low_light_image_path = os.path.join(
            '/home/hiepdvh/low-light-recognition-trial/Zero-DCE++/lol_dataset/eval15/low',
            low_light_image,
        )
        original_image = Image.open(low_light_image_path)
        enhanced_image = lowlight(original_image)
        result_path = os.path.join(
            '/home/hiepdvh/low-light-recognition-trial/Zero-DCE++/lol_dataset/eval15/high',
            low_light_image,
        )
        torchvision.utils.save_image(enhanced_image, result_path)
# if __name__ == '__main__':
#     for video in os.listdir('/home/hiepdvh/low-light-recognition/Zero-DCE++/video-test'):
#         input_video_path = f'/home/hiepdvh/low-light-recognition/Zero-DCE++/video-test/{video}'
#         output_video_dir_path = '/home/hiepdvh/low-light-recognition/Zero-DCE++/video-testenhanced'
#         if not os.path.exists(output_video_dir_path):
#             os.mkdir(output_video_dir_path)
#         output_video_path = os.path.join(output_video_dir_path, f'{video}')
#         process_video(input_video_path, output_video_path)
