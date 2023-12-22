import os
import time

import cv2
import model
import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 12
    data_lowlight = Image.open(image_path)

    data_lowlight = data_lowlight.convert('RGB')
    print(type(data_lowlight))
    data_lowlight = np.asarray(data_lowlight) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch14.pth'))
    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)

    end_time = time.time() - start

    print(end_time)
    image_path = image_path.replace('test_data', 'result_Zero_DCE++')

    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split('/')[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split('/')[-1], ''))
    # import pdb;pdb.set_trace()
    torchvision.utils.save_image(enhanced_image, result_path)
    return end_time


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


def process_video(input_path, output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    scale_factor = 12

    # Load mô hình DCE_net
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))

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
            out.write(enhanced_frame.astype(np.uint8))

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     with torch.no_grad():
#         filePath = 'data/test_data/'
#         file_list = os.listdir(filePath)
#         sum_time = 0
#         for file_name in file_list:
#             test_list = glob.glob(filePath + file_name + '/*')
#             for image in test_list:
#                 print(image)
#                 sum_time = sum_time + lowlight(image)
#         print(sum_time)

if __name__ == '__main__':
    input_video_path = 'test1.mp4'
    output_video_path = 'test1_high.mp4'

    process_video(input_video_path, output_video_path)
