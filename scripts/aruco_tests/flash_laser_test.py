# import the opencv library
import cv2
from cv2 import aruco
import numpy as np
import sys
from util import extract_laser, undistort_camera

def customAruco():
    # define an empty custom dictionary with 
    aruco_dict = cv2.aruco.custom_dictionary(0, 5, 1)
    # add empty bytesList array to fill with 3 markers later
    aruco_dict.bytesList = np.empty(shape = (5, 4, 4), dtype = np.uint8)
    # add new marker(s)
    mybits = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,1,1,1]], dtype = np.uint8)
    aruco_dict.bytesList[0] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,1]], dtype = np.uint8)
    aruco_dict.bytesList[1] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,1,1,0]], dtype = np.uint8)
    aruco_dict.bytesList[2] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,1,1,1],[1,0,0,0,0]], dtype = np.uint8)
    aruco_dict.bytesList[3] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    mybits = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,1,1,1],[1,0,1,1,1]], dtype = np.uint8)
    aruco_dict.bytesList[4] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
    # adjust dictionary parameters for better marker detection
    parameters =  cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = 3
    parameters.errorCorrectionRate = 0.2
    return aruco_dict, parameters

# define aruco dictionary
aruco_dict, arucoParams = customAruco()

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

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = cam.read()

    # Rotate frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    undist = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
      
    k = cv2.waitKey(1)

    
    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break

    # Capture image if spacebar is pressed
    elif k%256 == 32:
        print("PRESSED SPACEBAR")
        
        
        undist_name = f"res/pose_samples/flash_test/aruco_undist_{count}.png"
        cv2.imwrite(undist_name, gray)
        print("{} written!".format(undist_name))

        #TURN LASER ON

        laser, _ = extract_laser(undist)
        img_name = f"res/pose_samples/aruco_laser_{count}.png"
        cv2.imwrite(img_name, laser)
        print("{} written!".format(img_name))
        
        #TURN LASER OFF
        
        
        count +=1

        

        # corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
        
        # if len(corners) > 0:
        #     for i in range(0, len(ids)):  # Iterate in markers
        #         # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        #         rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.015, new_mtx, dist)
        #         print("Rvec", rvec)
        #         print("Tvec",  tvec)
        #         cv2.aruco.drawDetectedMarkers(undist, corners, ids)
        #         cv2.aruco.drawAxis(undist, new_mtx, dist, rvec, tvec, 0.01)


  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()