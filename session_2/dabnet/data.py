#
# data.py
# Dataset Collector for Dabnet
# 
import cv2
import os
import numpy as np
from shutil import rmtree

# OpenCV utilities
# Captures and Returns a grayscale frame from the camera
# Rescales the frame for the given scale
# Return frame is represented as a numpy array
capture = cv2.VideoCapture(0)
def capture_frame(scale=0.5):
    # Capture frame from camera 
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rescale the frame
    cv2.resize(frame, (0,0), fx=scale, fy=scale) 

    return frame

# Crop the given image repesented as a np array to a square frame of x by x
# where x is the length of the shorter side of the image
def crop_center(image):
    # Compute new dimentions for image
    # Crop a centered square from the image
    target_dim = min(image.shape)
    len_y, len_x = image.shape

    begin_y = (len_y // 2) - (target_dim // 2)
    end_y  = (len_y // 2) + (target_dim // 2)
    
    begin_x = (len_x // 2) - (target_dim // 2)
    end_x  = (len_x // 2) + (target_dim // 2)
    
    # Perform crop for computed dimentions
    image = image[begin_y:end_y, begin_x: end_x]
    return image


if __name__ == "__main__":
    # Setup data directory
    if os.path.exists("data"): rmtree("data")
    os.mkdir("data")
    os.chdir("data")
    
    # Interactive dataset collector
    print("Dataset Recorder for Dabnet...")
    print("press d to record a dab, n to record not a dab.")
    print("q to quit")
    
    n_dab, n_notdab = 0, 0
    while True:
        frame = capture_frame()
        frame = crop_center(frame)
        cv2.imshow("d - dab, n - not dab, q - quit", frame)
    
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == "q":
            capture.release()
            break # Quit program
        elif key == "d":
            n_dab += 1
            cv2.imwrite("dab_{}.jpg".format(n_dab), frame)
            print("{} dabs, {} notdabs".format(n_dab, n_notdab))
        elif key == "n":
            n_notdab += 1
            cv2.imwrite("notdab_{}.jpg".format(n_notdab), frame)
            print("{} dabs, {} notdabs".format(n_dab, n_notdab))
