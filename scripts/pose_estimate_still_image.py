import cv2
import numpy as np
import sys
from cv2 import aruco

def read_node_matrix( reader, name ):
    node = reader.getNode( name )
    return node.mat()

def main():
    # Initialization value for image
    file_name = "Markers\Test_Images\im1.jpg"
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    arucoParams = cv2.aruco.DetectorParameters_create()
    board = cv2.aruco.CharucoBoard_create(12,
                                          9,
                                          60/1000,
                                          47/1000,
                                          aruco_dict)
    
    camera_reader = cv2.FileStorage()
    camera_reader.open("cameraParameters.xml",cv2.FileStorage_READ)

    camera_matrix = read_node_matrix( camera_reader, "cameraMatrix" )
    dist_coeffs   = read_node_matrix( camera_reader, "dist_coeffs" )
    
    frame = cv2.imread(file_name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, 
    parameters=arucoParams)
    #print(ids)
    ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board )
    #print(ch_ids)
    cv2.aruco.drawDetectedCornersCharuco(frame,ch_corners,ch_ids,(0,0,255) )
    
    print(camera_matrix)
    print(dist_coeffs)
    #retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard( ch_corners, ch_ids, board, camera_matrix, dist_coeffs)
    for i in range(0, len(ids)):  # Iterate in markers
        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.015, camera_matrix,dist_coeffs)
        print("Rvec", rvec)
        print("Tvec",  tvec)
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)



    # Draw image or quit
    cv2.imshow('Image', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


main()