import sys

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

def run_model(filepath):
    # architecture and weights from HDF5
    model = load_model('models/model.h5')

    # architecture from JSON, weights from HDF5
    with open('models/architecture.json') as f:
        model = model_from_json(f.read())
    model.load_weights('models/weights.h5')

    img = Image.open(filepath)
    img_size = 224
    
    batch = np.stack([preprocess_input(np.array(img.resize((img_size, img_size))))])

    preds = model.predict(batch)

    print(f"Giraffe Score: {preds[0,1]}")

if __name__=="__main__":
    filename = sys.argv[1]

    run_model(filename)