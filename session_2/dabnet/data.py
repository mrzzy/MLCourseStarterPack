#
# data.py
# Dataset Collector for Dabnet
# 

import cv2
import os
import re
import numpy as np
from PIL import Image

from cv import capture_frame, crop_center

## Dataset Manipulation
# Count the number of entries for each label in tne dataset
# Returns n_dab, n_notdab counts for the number of dab entries, number of non dab 
# entries
def count_dataset():
    img_paths = list(filter((lambda p: re.match(r"(not)?dab_[0-9]+.jpg", p) != None),
                       os.listdir()))
    n_notdab = sum([ 1 for path in img_paths if "notdab"])
    n_dab = len(img_paths) - n_notdab

    return n_dab, n_notdab

# Load dab image data from the given data path
# Represents images as numpy arrays and labels as 1 - dab, 0 - no dab
# Returns a list of images and labels
def load_data(path="data"):
    cwd = os.getcwd()
    os.chdir(path)
    img_paths = filter((lambda p: re.match(r"(not)?dab_[0-9]+.jpg", p) != None),
                       os.listdir())

    # Collate data into image and label lists
    imgs = []
    labels = []
    for img_path in img_paths:
        # Load and collect image
        img = Image.open(img_path)
        imgs.append(img)

        # Detemine and collect label
        if "notdab" in img_path:
            labels.append(0)
        elif "dab" in img_path:
            labels.append(1)

    os.chdir(cwd)
    return imgs, labels


if __name__ == "__main__":
    if not os.path.exists("data"): os.mkdir("data")
    os.chdir("data")
    
    # Interactive dataset collector
    print("Dataset Recorder for Dabnet...")
    print("press d to record a dab, n to record not a dab.")
    print("q to quit")
    
    n_dab, n_notdab = count_dataset()
    while True:
        frame = capture_frame()
        frame = crop_center(frame)
        cv2.imshow("d - dab, n - not dab, q - quit", frame)
    
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == "q":
            break # Quit program
        elif key == "d":
            n_dab += 1
            cv2.imwrite("dab_{}.jpg".format(n_dab), frame)
            print("{} dabs, {} notdabs".format(n_dab, n_notdab))
        elif key == "n":
            n_notdab += 1
            cv2.imwrite("notdab_{}.jpg".format(n_notdab), frame)
            print("{} dabs, {} notdabs".format(n_dab, n_notdab))
