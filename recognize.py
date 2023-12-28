import argparse
import os
import pickle
import time
from types import SimpleNamespace
from typing import List

import cv2
import enhance
import mediapipe as mp
import numpy as np
import onnxruntime as rt
from skimage.transform import SimilarityTransform
from sklearn.metrics.pairwise import cosine_distances


# Define a class to store a detection
class Detection(SimpleNamespace):
    bbox: List[List[float]] = None
    landmarks: List[List[float]] = None


# Define a class to store an identity
class Identity(SimpleNamespace):
    detection: Detection = Detection()
    name: str = None
    embedding: np.ndarray = None
    face: np.ndarray = None


# Define a class to store a match
class Match(SimpleNamespace):
    subject_id: Identity = Identity()
    gallery_id: Identity = Identity()
    distance: float = None
    name: str = None


SIMILARITY_THRESHOLD = 0.7

FACE_RECOGNIZER = rt.InferenceSession('RecognizeModel.onnx', providers=rt.get_available_providers())
FACE_DETECTOR = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=7,
)


def detect_faces(frame: np.ndarray) -> List[Detection]:
    # Process the frame with the face detector
    result = FACE_DETECTOR.process(frame)

    # Initialize an empty list to store the detected faces
    detections = []

    # Check if any faces were detected
    if result.multi_face_landmarks:
        # Iterate over each detected face
        for count, detection in enumerate(result.multi_face_landmarks):
            # Select 5 Landmarks
            five_landmarks = np.asarray(detection.landmark)[[470, 475, 1, 57, 287]]

            # Extract the x and y coordinates of the landmarks of interest
            landmarks = np.asarray(
                [
                    [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]
                    for landmark in five_landmarks
                ]
            )

            # Extract the x and y coordinates of all landmarks
            all_x_coords = [landmark.x * frame.shape[1] for landmark in detection.landmark]
            all_y_coords = [landmark.y * frame.shape[0] for landmark in detection.landmark]

            # Compute the bounding box of the face
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            bbox = [[x_min, y_min], [x_max, y_max]]

            # Create a Detection object for the face
            detection = Detection(idx=count, bbox=bbox, landmarks=landmarks, confidence=None)

            # Add the detection to the list
            detections.append(detection)

    # Return the list of detections
    return detections


def recognize_faces(frame: np.ndarray, detections: List[Detection]) -> List[Identity]:
    if not detections:
        return []

    identities = []
    for detection in detections:
        # ALIGNMENT -----------------------------------------------------------
        # Target landmark coordinates (as used in training)
        landmarks_target = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        tform = SimilarityTransform()
        tform.estimate(detection.landmarks, landmarks_target)
        tmatrix = tform.params[0:2, :]
        face_aligned = cv2.warpAffine(frame, tmatrix, (112, 112), borderValue=0.0)
        # ---------------------------------------------------------------------

        # INFERENCE -----------------------------------------------------------
        # Inference face embeddings with onnxruntime
        input_image = (np.asarray([face_aligned]).astype(np.float32) / 255.0).clip(0.0, 1.0)
        embedding = FACE_RECOGNIZER.run(None, {'input_image': input_image})[0][0]
        # ---------------------------------------------------------------------

        # Create Identity object
        identities.append(Identity(detection=detection, embedding=embedding, face=face_aligned))

    return identities


def extract_faces(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    if not detections:
        return []

    for detection in detections:
        # ALIGNMENT -----------------------------------------------------------
        # Target landmark coordinates (as used in training)
        landmarks_target = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        tform = SimilarityTransform()
        tform.estimate(detection.landmarks, landmarks_target)
        tmatrix = tform.params[0:2, :]
        face_aligned = cv2.warpAffine(frame, tmatrix, (112, 112), borderValue=0.0)
    return face_aligned


def extract_feature(
    database_path: str = '/home/livefaceidapp/database',
    feature_path: str = 'feature.pkl',
) -> None:
    # Get a list of image files in the database
    image_files = [f for f in os.listdir(database_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Process image files and add to gallery
    gallery = []
    list_detections = []
    faces = []
    for file_name in image_files:
        # Construct the full path to the image file
        file_path = os.path.join(database_path, file_name)

        # Read file bytes
        file_bytes = np.asarray(bytearray(open(file_path, 'rb').read()), dtype=np.uint8)

        # Decode image and convert from BGR to RGB
        img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Detect faces
        detections = detect_faces(img)
        list_detections.append(detections)
        if detections:
            # Recognize faces
            subjects = recognize_faces(img, detections[:1])  # take only one face
            faces.append(extract_faces(img, detections[:1]))
            # Add subjects to gallery
            gallery.append(
                Identity(
                    name=os.path.splitext(file_name)[0],
                    embedding=subjects[0].embedding,
                    face=subjects[0].face,
                )
            )

    with open(feature_path, 'wb') as file:
        pickle.dump(gallery, file)


def match_faces(subjects: List[Identity], gallery: List[Identity]) -> List[Match]:
    if len(gallery) == 0 or len(subjects) == 0:
        return []

    # Get Embeddings
    embs_gal = np.asarray([identity.embedding for identity in gallery])
    embs_det = np.asarray([identity.embedding for identity in subjects])

    # Calculate Cosine Distances
    cos_distances = cosine_distances(embs_det, embs_gal)

    # Find Matches
    matches = []
    for ident_idx, identity in enumerate(subjects):
        dists_to_identity = cos_distances[ident_idx]
        idx_min = np.argmin(dists_to_identity)
        if dists_to_identity[idx_min] < SIMILARITY_THRESHOLD:
            matches.append(
                Match(
                    subject_id=identity,
                    gallery_id=gallery[idx_min],
                    distance=dists_to_identity[idx_min],
                )
            )

    # Sort Matches by identity_idx
    matches = sorted(matches, key=lambda match: match.gallery_id.name)

    return matches


def draw_annotations(
    frame: np.ndarray, detections: List[Detection], matches: List[Match]
) -> np.ndarray:
    shape = np.asarray(frame.shape[:2][::-1])

    # Upscale frame to 1080p for better visualization of drawn annotations
    frame = cv2.resize(frame, (1920, 1080))
    upscale_factor = np.asarray([1920 / shape[0], 1080 / shape[1]])
    shape = np.asarray(frame.shape[:2][::-1])

    # Make frame writeable (for better performance)
    frame.flags.writeable = True

    # Draw Detections
    for detection in detections:
        # Draw Landmarks
        for landmark in detection.landmarks:
            cv2.circle(frame, (landmark * upscale_factor).astype(int), 2, (255, 255, 255), -1)

        # Draw Bounding Box
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (255, 0, 0),
            2,
        )

        # Draw Index
        cv2.putText(
            frame,
            str(detection.idx),
            (
                ((detection.bbox[1][0] + 2) * upscale_factor[0]).astype(int),
                ((detection.bbox[1][1] + 2) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    # Draw Matches
    for match in matches:
        detection = match.subject_id.detection
        name = match.gallery_id.name

        # Draw Bounding Box in green
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (0, 255, 0),
            2,
        )

        # Draw Banner
        cv2.rectangle(
            frame,
            (
                (detection.bbox[0][0] * upscale_factor[0]).astype(int),
                (detection.bbox[0][1] * upscale_factor[1] - (shape[1] // 25)).astype(int),
            ),
            (
                (detection.bbox[1][0] * upscale_factor[0]).astype(int),
                (detection.bbox[0][1] * upscale_factor[1]).astype(int),
            ),
            (255, 255, 255),
            -1,
        )

        # Draw Distance and Name
        cv2.putText(
            frame,
            f' Distance: {match.distance:.2f}, name:{name}',
            (
                ((detection.bbox[0][0] + shape[0] // 400) * upscale_factor[0]).astype(int),
                ((detection.bbox[0][1] - shape[1] // 350) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    return frame


def video_frame_callback(frame: np.ndarray, gallery) -> np.ndarray:
    # Run face detection
    detections = detect_faces(frame)

    # Run face recognition
    subjects = recognize_faces(frame, detections)

    # Run face matching
    matches = match_faces(subjects, gallery)

    # Draw annotations
    frame = draw_annotations(frame, detections, matches)

    return frame


if __name__ == '__main__':
    # extract_feature()
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

    output_video_path = 'output_video/output_video4.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = enhance.enhance_frame(frame)

        frame = frame.astype('uint8')
        frame = video_frame_callback(frame, gallery)
        end_time = time.time()
        elapsed_time = end_time - start_time
        real_fps = int(1 / elapsed_time)

        cv2.putText(
            frame,
            f'FPS: {real_fps:.2f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        out.write(frame)

        start_time = time.time()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
