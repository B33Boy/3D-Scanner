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


# Aruco prep
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
board = aruco.CharucoBoard_create(11, 8, 1.5, 1.2, aruco_dict)
parameters =  aruco.DetectorParameters_create()

def print_axes(undist, aruco_dict, parameters, board, mtx):
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
                                                          parameters=parameters)
    # SUB PIXEL DETECTION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

    # frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    empty_dist = np.array([0,0,0,0,0]).reshape(1,5)
    
    charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
    im_with_charuco_board = cv2.aruco.drawDetectedCornersCharuco(gray, charucoCorners, charucoIds, (0,255,0))
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, empty_dist, rvec = False, tvec = False)  # posture estimation from a charuco board
    im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, mtx, np.array([0.0,0.0,0.0,0.0,0.0]).reshape(1,5), rvec, tvec, 100)
    
    return im_with_charuco_board


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    # Rotate frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # First undistort
    undist = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)

    axes_img = print_axes(undist, aruco_dict, parameters, board, new_mtx)

    cv2.imshow("Pose estimate charuco", axes_img)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break
    # Capture image if spacebar is pressed
    elif k%256 == 32:
        if args.test == 'a':
            if count < 5:
                img_name = f"res/unit_tests/pose_est/accuracy/charuco_{count}.png"
                cv2.imwrite(img_name, axes_img)
                print("Charuco {} written!".format(img_name))
            
            if count >= 5 and count < 10:
                img_name = f"res/unit_tests/pose_est/accuracy/aruco_{count}.png"
                cv2.imwrite(img_name, axes_img)
                print("Aruco {} written!".format(img_name))                

        elif args.test == 'b':
            if count == 0:
                img_name = "res/unit_tests/pose_est/brightness/charuco_dark.png"
            elif count == 1:
                img_name = "res/unit_tests/pose_est/brightness/charuco_ambient.png"
            elif count == 2:
                img_name = "res/unit_tests/pose_est/brightness/charuco_bright.png"
            
            cv2.imwrite(img_name, axes_img)
            print("Brightness test image {} written!".format(img_name))
        
        elif args.test == 'i':
            if count == 0:
                img_name = f"res/unit_tests/pose_est/interference/interference_{count}.png"
            
            cv2.imwrite(img_name, axes_img)
            print("Brightness test image {} written!".format(img_name))


        count += 1

cam.release()

cv2.destroyAllWindows()



    

