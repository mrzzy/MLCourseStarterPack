#
# cv.py
# Computer Vision Utilities
#
import cv2

# OpenCV utilities
# Captures and Returns a grayscale frame from the camera
# Rescales the frame for the given scale
# Return frame is represented as a numpy array
capture = None
def capture_frame(scale=0.5):
    global capture
    if capture == None: capture = cv2.VideoCapture(0)
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

