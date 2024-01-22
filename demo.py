import pickle
import tempfile
import time

import cv2
import streamlit as st
from enhance import enhance_frame
from recognize import video_frame_callback

with open('lowlight.h5', 'rb') as file:
    gallery = pickle.load(file)


def main():
    st.title('Video Player with Streamlit')

    video_file = st.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov', 'm4v'])

    button_stop = st.button('Stop')
    enhance_state = st.checkbox('Enhance State')
    if video_file is not None:
        st.video(video_file)
        video_name = video_file.name

        video_bytes = video_file.read()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_bytes)

        cap = cv2.VideoCapture(temp_file.name)
        video_placeholder = st.image([])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            start = time.time()
            if enhance_state:
                frame = enhance_frame(frame)
            frame = frame = frame.astype('uint8')
            frame = video_frame_callback(frame, gallery, video_name.split('.')[0])
            frame = frame / 255
            video_placeholder.image(frame, channels='BGR', use_column_width=True)
            end = time.time()
            # print(end - start)
            if button_stop:
                break
        cap.release()

        temp_file.close()


if __name__ == '__main__':
    main()
