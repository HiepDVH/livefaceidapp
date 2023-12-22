import os

import cv2
import numpy as np
from recognition import Identity, detect_faces, extract_faces, recognize_faces

# Specify the directory where your image files are located
image_directory = '/home/livefaceidapp/database'

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('png', 'jpg', 'jpeg'))]

# Process image files and add to gallery
gallery = []
list_detections = []
faces = []
for file_name in image_files:
    # Construct the full path to the image file
    file_path = os.path.join(image_directory, file_name)

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


# with open('feature.pkl', 'wb') as file:
#     pickle.dump(gallery, file)

for i in range(len(gallery)):
    name = gallery[i].name
    path = f'/home/livefaceidapp/face/{name}.png'
    face = faces[i]
    cv2.imwrite(path, face)
