#
# train.py
# Model Training
# Dabnet
#

import os
import numpy as np
import random
from datetime import datetime
from PIL import Image
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count

import model
from data import load_data

# Performs preprocessing steps to the given PIL image
# Rescales the image to 128 by 128
# Reshapes image to a 128 by 128 by 1 numpy array
# Scales pixel values between 0 and 1
# Return the processed image as a np array
def preprocess_image(image):
    image = image.resize((128, 128), Image.NEAREST)
    image = np.asarray(image)
    # Add channel dimention to data
    image = np.expand_dims(image, axis=2)
    # scale pixel values
    image = image / 255.0
    
    return image

if __name__ == "__main__":
    # Preprocess Data
    print("Preprocessing data...")
    worker_pool = Pool(cpu_count())
    images, labels = load_data()
    images = worker_pool.map(preprocess_image, images)
    
    # Shuffle data
    dataset = list(zip(images, labels))
    random.shuffle(dataset)
    images, labels = zip(*dataset)
    images = np.asarray(images)
    labels = np.asarray(labels)

    print(labels)

    # Train model
    print("Training model...")
    model = model.build()
    # TODO: Train the model

    model.save("dabnet.hd5")
    print("Saved trained model at dabnet.hd5")
