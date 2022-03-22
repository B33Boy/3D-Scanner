import cv2 
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

import argparse

# Functions
def undistort_camera(img, mtx, new_mtx, roi, dist, w, h):

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    
    # crop the image
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]


def extract_laser(frame): 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_colour = np.array([146, 62, 0])
    upper_colour = np.array([255, 255, 255])
    
    # Threshold the HSV image to get only get red colour
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    
    # Isolate the red channel
    img = frame[...,2]
    ret,img = cv2.threshold(img,144,255,0)

    # Create emptry array of zeros of same size as img
    out = np.zeros_like(img)

    # For each row, get the position of the highest intensity
    bppr = np.argmax(img, axis=1)

    # Set the highest intensity pixel locations to 255
    out[np.arange(bppr.shape[0]), bppr] = 255
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(out,out, mask= mask)
    
    return res, bppr


parser = argparse.ArgumentParser(description='Accuracy, Brightness, Interference')
parser.add_argument('-t','--test', help='type of test (accuracy: a, brightness: b, interference: i)', required=True)
args = parser.parse_args()

print("Test:", args.test)

# Load camera calibration data from cam_out folder
with np.load('res/cal_out/cam_params.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

# Start the video capture
cam = cv2.VideoCapture(0)

# Obtain the width and height of the camera
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Undistort Camera Matrix + ROI
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

count = 0



while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    # Rotate frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # First undistort
    undist = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)

    #laser = extract_laser(undist)

    cv2.imshow("Laser isolation", undist)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break

    # Capture image if spacebar is pressed
    elif k%256 == 32:

        if args.test == 'a':
            pass

        elif args.test == 'b':
            if count%2 == 0:
                img_name = f"res/unit_tests/laser_iso/laser_on_{count}.png"
            if count%2 == 1:
                img_name = f"res/unit_tests/laser_iso/laser_off_{count}.png"

            cv2.imwrite(img_name, undist)
            print("Laser {} written!".format(img_name))
        
        elif args.test == 'i':
            pass


        count += 1

cam.release()

cv2.destroyAllWindows()



    

