import os

import keras
import numpy as np
from model import mirnet_model
from PIL import Image

model = mirnet_model(num_rrg=3, num_mrb=2, channels=64)

model.load_weights('Epochs50-weight.h5')


def infer(original_image):
    image = keras.utils.img_to_array(original_image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image, verbose=0)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape((np.shape(output_image)[0], np.shape(output_image)[1], 3))
    output_image = Image.fromarray(np.uint8(output_image))
    original_image = Image.fromarray(np.uint8(original_image))
    return output_image


test_low_light_images = os.listdir('/home/hiepdvh/MIRNET/lol_dataset/eval15/low')

for low_light_image in test_low_light_images:
    low_light_image_path = os.path.join(
        '/home/hiepdvh/MIRNET/lol_dataset/eval15/low', low_light_image
    )
    original_image = Image.open(low_light_image_path)
    enhanced_image = infer(original_image)
    enhanced_image.save(
        os.path.join(
            '/home/hiepdvh/MIRNET/lol_dataset/eval15/high',
            low_light_image,
        )
    )
