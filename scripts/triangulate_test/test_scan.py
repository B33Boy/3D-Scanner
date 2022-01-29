# Code required to import from root of scripts/ folder when working in a subfolder within scripts/
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import extract_laser, undistort_camera

# External libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load theta calibration parameters
with np.load('res/cal_theta_out/theta_params.npz') as X:
    theta_coeff = X['theta_coeff']

# Load camera calibration data from cam_out folder
with np.load('res/cal_out/cam_params.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

# Start the video capture
cam = cv2.VideoCapture(0)

# Obtain the width and height of the camera
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get new camera matrix
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Camera params
centre_x = w/2
#dist b/w camera and laser is 4 in
X = 4

count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    # Rotate image 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # First undistort
    frame = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)

    # Next extract the laser + matrix containing points of interest
    frame, POI = extract_laser(frame)
    
    cv2.imshow("cam_img", frame)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break
    # Capture image if spacebar is pressed
    elif k%256 == 32:
        img_name = f"res/cal_theta_out/test_scan_{count}.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        count += 1
        
        # Get distance
        pix_dist = POI - centre_x
        theta = np.multiply(theta_coeff[86:335,0], pix_dist[86:335]) + theta_coeff[86:335,1]
        # print(theta.shape)
        D = X*np.tan(theta)
        print(D)
        

cam.release()

cv2.destroyAllWindows()


