import argparse
import pickle
import time

import cv2
import numpy as np
import recognition

with open('feature.pkl', 'rb') as file:
    gallery = pickle.load(file)
parser = argparse.ArgumentParser()

parser.add_argument(
    '--video-path',
    type=str,
    help='video path',
    required=True,
)

args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)


if not cap.isOpened():
    print('Không thể mở video đầu vào.')
    exit()


fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = 'output_video2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # frame = enhance.enhance_frame(frame)

    # frame = frame.astype('uint8')
    frame = recognition.video_frame_callback(frame, gallery)
    end_time = time.time()
    elapsed_time = end_time - start_time
    real_fps = 1 / elapsed_time

    cv2.putText(
        frame,
        f'FPS: {real_fps:.2f}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # frame.resize((height, width, 3))
    # frame = frame.astype(np.uint8)
    # frame = cv2.resize(frame, (1920, 1080))
    out.write(frame.astype(np.uint8))

    start_time = time.time()

cap.release()
out.release()
cv2.destroyAllWindows()
