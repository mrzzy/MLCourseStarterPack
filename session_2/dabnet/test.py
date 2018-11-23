#
# test.py
# Dabnet
# Model Testing Client
#

import cv2
import numpy as np
from keras import models
from PIL import Image

from cv import capture_frame, crop_center
from train import preprocess_image

if __name__ == "__main__":
    # Load model from file
    model = models.load_model("dabnet.hd5")

    # Interactive model testing client
    print("Dabnet test client...")
    print("q to quit")

    while True:
        frame = capture_frame()
        frame = crop_center(frame)
        
        # Classify frame contains dab or not
        image = Image.fromarray(frame.astype("uint8"))
        image = preprocess_image(image)
        
        # TODO: Make predictions using model
       
        # Show Results of classfication
        cv2.putText(frame, prediction,(8, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    2,(255, 255, 255), 2 ,cv2.LINE_AA)
        cv2.imshow("Detecting...", frame)
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == "q":
            break # Quit program
