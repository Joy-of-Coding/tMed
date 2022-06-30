import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # -1 for CPU 0, 1, etc for GPU(s)

import numpy as np
from tensorflow import keras

import config
from dataset import random_image_and_mask


# Load model
weights = r".\logs\2022-04-11 14-16-39__-gputest2\weights_20.h5"
model = keras.models.load_model(weights, compile=False)

# Load an image
img, gt_mask = random_image_and_mask(config)

# preprocess
img = img.astype(np.float32) / 255 - 0.5
result = model.predict(np.array([img]))
result = result[0]

# Now can compare gt_mask & result
