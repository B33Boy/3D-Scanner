'''
Main Run File

Todo: Bring all functions into this file 
Sequence: 
1. Run Camera capture and pose estimation in the background
2. Check GPIO for button Press
3. If pose is available for image, do scan, else flash red light
4. Save all scan data 
5. At finish scan (press Start Button again --> change states), export all data 

'''

# Import installed libraries
from operator import truediv
import cv2 
from cv2 import aruco
from datetime import datetime
import numpy as np

'''
    Global Variables
'''
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
board = aruco.CharucoBoard_create(11, 8, 1.5, 1.2, aruco_dict)

'''
    LOAD CAMERA CALIBRATION HERE
'''
with np.load('res/cal_out/cam_params.npz') as X:
    camera_matrix, dist_coeffs = [X[i] for i in ('mtx','dist')]
    
'''
    Functions and Helpers
'''

def read_charuco(dt, image):
    '''
        Base pose estimation
    '''
    print("Start Pose Estimation Sequence...")
    allCorners = []
    allIds = []
    decimator = 0
    charuco_detected = False
    
    #SUB PIXEL CORNER DETECTION CRITERIA
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.00001)
    print("processing image{x}".format(dt))
    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.arcuco.detectMarkers(gray, aruco_dict)
    #Check corners array to see if any detected. TODO: Use Try Except
    '''
    if len(corners) > 0:
        for corner in corners:
            cv2.cornerSubPix(gray, corner, winSize= (3,3), zeroZone= (-1,-1), criteria= criteria)
        res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1 == 0:
            allCorners.append(res2[1])
            allIds.append(res2[2])
    '''
    
    try:
        if len(corners) > 0:
            print("Charuco detected with {} corners".format(len(corners)))
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize= (3,3), zeroZone= (-1,-1), criteria= criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])
            flash_green_LED()
            charuco_detected = True
    except:
        print("*Unable to get pose")
        flash_red_LED()
            
    decimator += 1
    imsize = gray.shape
    
    return allCorners, allIds, imsize, gray, charuco_detected
        

def flash_red_LED():
    pass

def flash_green_LED():
    pass




#main function
def main():
    vid = cv2.VideoCapture(0)
    
   
    
    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()
        
        #get the current timestamp of image 
        dt = datetime.now()
        
        
        # Rotate frame 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        
        #undistort the image before processing 
        img_undistort = cv2.undistort(frame, cameraMatrix= camera_matrix, distCoeffs= dist_coeffs)
        
        allCorners,allIds,imsize, gray, charuco_detected =read_charuco(dt= dt, image= img_undistort)
        
        if(charuco_detected):
            pass
            # if board is detected, run pose calculations and laser isolation and triangulation
        
        # Display the resulting frame
        cv2.imshow('frame', img_undistort) #remove later

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

#run the main function
if __name__ == '__main__':
    main()