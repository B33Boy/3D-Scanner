# Generic skeleton script for later use

# import the opencv library
import cv2
import numpy as np
import sys
from cv2 import aruco

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

  
# define a video capture object
vid = cv2.VideoCapture(0)
aruco_dict, arucoParams = customAruco()

with np.load('res/cal_out/cam_params.npz') as X:
        camera_matrix, dist_coeffs = [X[i] for i in ('mtx','dist')]
    
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Rotate frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, 
    parameters=arucoParams)
    
    if len(corners) > 0:
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.015, camera_matrix,dist_coeffs)
            print("Rvec", rvec)
            print("Tvec",  tvec)
            #cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)


    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()