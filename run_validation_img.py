import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

import tensorflow.keras as keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json

# architecture and weights from HDF5
model = load_model('models/model.h5')

# architecture from JSON, weights from HDF5
with open('models/architecture.json') as f:
    model = model_from_json(f.read())
model.load_weights('models/weights.h5')


validation_img_paths = ["images/data/val/animal giraffe/2O5CMBPQT1J7.jpg",
                        "images/data/val/animal giraffe/4XR4KKW2BMNP.jpg",
                        "images/data/val/animal giraffe/5MLDX4EWXPNH.jpg",]
img_list = [Image.open(img_path) for img_path in validation_img_paths]

img_size = 224
validation_batch = np.stack([preprocess_input(np.array(img.resize((img_size, img_size))))
                             for img in img_list])

pred_probs = model.predict(validation_batch)

fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% chimken, {:.0f}% geraf, {:.0f}% lion".format(100*pred_probs[i,0],
                                                          100*pred_probs[i,1],
                                                          100*pred_probs[i,2]))
    ax.imshow(img)

plt.savefig("test.png")