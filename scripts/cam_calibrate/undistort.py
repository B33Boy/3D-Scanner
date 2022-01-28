import numpy as np
import cv2

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html



def undistort_camera(img, mtx, new_mtx, roi, dist, w, h):

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    
    # crop the image
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]


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


while True:
    
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    # Rotate image 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # First undistort
    undist = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)

    cv2.imshow("distorted", frame)
    cv2.imshow("undistorted", undist)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break         

cam.release()
cv2.destroyAllWindows()