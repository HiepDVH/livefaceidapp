import os

import cv2
import EnhanceModel
import numpy as np
import torch
import torch.optim


def enhance_frame(frame):
    scale_factor = 12

    frame = np.asarray(frame) / 255.0

    frame = torch.from_numpy(frame).float()

    h = (frame.shape[0] // scale_factor) * scale_factor
    w = (frame.shape[1] // scale_factor) * scale_factor
    frame = frame[0:h, 0:w, :]
    frame = frame.permute(2, 0, 1)
    frame = frame.cpu().unsqueeze(0)

    DCE_net = EnhanceModel.enhance_net_nopool(scale_factor).cpu()
    DCE_net.load_state_dict(
        torch.load(
            'EnhanceModel.pth',
            map_location=torch.device('cpu'),
        )
    )
    enhanced_frame, params_maps = DCE_net(frame)

    enhanced_frame = enhanced_frame[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0

    return enhanced_frame


def process_video(input_path, output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    scale_factor = 12

    # Load mô hình DCE_net
    DCE_net = EnhanceModel.enhance_net_nopool(scale_factor).cpu()
    DCE_net.load_state_dict(
        torch.load(
            'EnhanceModel.pth',
            map_location=torch.device('cpu'),
        )
    )

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (1920, 1080))
            enhanced_frame = enhance_frame(resized_frame)
            out.write(enhanced_frame.astype(np.uint8))

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for video in os.listdir('/home/livefaceidapp/video-test'):
        if 'hung' in video:
            input_video_path = f'/home/livefaceidapp/video-test/{video}'
            output_video_dir_path = '/home/livefaceidapp/video-test-enhanced2'
            if not os.path.exists(output_video_dir_path):
                os.mkdir(output_video_dir_path)
            output_video_path = os.path.join(output_video_dir_path, f'{video}')
            process_video(input_video_path, output_video_path)
