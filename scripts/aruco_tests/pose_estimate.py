import cv2
import numpy as np
import sys
from cv2 import aruco

def read_node_matrix( reader, name ):
    node = reader.getNode( name )
    return node.mat()

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


def main():
    # Initialization value for image. Change to location of image
    file_name = "res\marker_test\marker_4.png"
    aruco_dict, arucoParams = customAruco()
    
    '''
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    arucoParams = cv2.aruco.DetectorParameters_create()
    board = cv2.aruco.CharucoBoard_create(12,
                                          9,
                                          60/1000,
                                          47/1000,
                                          aruco_dict)
    '''
    
    with np.load('res/cal_out/cam_params.npz') as X:
        camera_matrix, dist_coeffs = [X[i] for i in ('mtx','dist')]
    
    print(camera_matrix)
    print(dist_coeffs)
    '''
    camera_reader = cv2.FileStorage()
    camera_reader.open("cameraParameters.xml",cv2.FileStorage_READ)

    camera_matrix = read_node_matrix( camera_reader, "cameraMatrix" )
    dist_coeffs   = read_node_matrix( camera_reader, "dist_coeffs" )
    '''
    #camera_matrix = np.array(
     #                    [[focal_length, 0, center[0]],
      #                   [0, focal_length, center[1]],
       #                  [0, 0, 1]], dtype = "double"
        #                 )
    #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    
    frame = cv2.imread(file_name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, 
    parameters=arucoParams)
    #print(ids)
    #ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board )
    #print(ch_ids)
    #cv2.aruco.drawDetectedCornersCharuco(frame,ch_corners,ch_ids,(0,0,255) )

    #retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard( ch_corners, ch_ids, board, camera_matrix, dist_coeffs)
    for i in range(0, len(ids)):  # Iterate in markers
        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.015, camera_matrix,dist_coeffs)
        print("Rvec", rvec)
        print("Tvec",  tvec)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)



    # Draw image or quit
    cv2.imshow('Image', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


main()