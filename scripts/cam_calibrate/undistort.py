import numpy as np
import cv2

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

# Load previously saved data
with np.load('res/cal_out/cam_params.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    print(mtx.shape)
    print(dist.shape)
    print(rvecs.shape)
    print(tvecs.shape)
    print("-------------------------")

def undistort_camera(img, mtx, new_mtx, roi, dist, w, h):

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    # crop the image
    x, y, w, h = roi
    return dst#[y:y+h, x:x+w]


# Camera params
w = 640
h = 480


# Undistort Camera Matrix + ROI
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(new_mtx)
print(roi)
print("=======================")

cam = cv2.VideoCapture(0)

while True:
    
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    # First undistort
    undist_frame = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)
    print("Shape of undistorted frame: ",undist_frame.shape)

    cv2.imshow("distorted", frame)
    cv2.imshow("undistorted", undist_frame)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break         

cam.release()
cv2.destroyAllWindows()