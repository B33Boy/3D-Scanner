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
from datetime import datetime
import argparse
import os

import cv2 
from cv2 import aruco
import numpy as np
import open3d as o3d
from gpiozero import Button, LED

'''
    Command Line Arguments
'''
parser = argparse.ArgumentParser(description='Laser Scanner Parameters')
parser.add_argument('-d','--dist', help='distance between laser and camera in mm', default=101.6) # Note 101.6mm is 4 inches
parser.add_argument('-t','--test', action='store_false', help='test mode on PC') # normally false, if flag called it becomes true
args = parser.parse_args()


'''
    Global Variables
'''
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
board = aruco.CharucoBoard_create(11, 8, 1.5, 1.2, aruco_dict)
parameters =  aruco.DetectorParameters_create()

onFlag = False

'''
    Camera Calibration
'''
with np.load('res/cal_out/cam_params.npz') as X:
    camera_matrix, dist_coeffs = [X[i] for i in ('mtx','dist')]
    
'''
    Functions and Helpers
'''
def get_itf(rvec, tvec):
    """ Function that takes in 3x1 rvec and tvec representing the transformation from marker coords to camera coords, and outputs the 4x4 transformation matrix from camera coords to marker coords. 

    Args:
        rvec (ndarray): rotation vector
        tvec (ndarray): translation vector

    Returns:
        itf: inverse transformation matrix that is of dimension 4x4
    """
    dst, _ = cv2.Rodrigues(rvec) # 2nd output is jacobian
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = dst
    extrinsics[:3, 3] = tvec.flatten()
    return np.linalg.inv(extrinsics)

def get_tf(undist, aruco_dict, parameters, board, mtx):
    """ Gets the transformation from the camera to the charuco board

    Args:
        undist (_type_): _description_
        aruco_dict (_type_): _description_
        parameters (_type_): _description_
        board (_type_): _description_
        mtx (_type_): _description_

    Returns:
        _type_: _description_
    """
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
                                                          parameters=parameters)
    if ids is not None:
        # SUB PIXEL DETECTION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        for corner in corners:
            cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

        empty_dist = np.array([0,0,0,0,0]).reshape(1,5)

        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        im_with_charuco_board = cv2.aruco.drawDetectedCornersCharuco(gray, charucoCorners, charucoIds, (0,255,0))
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, empty_dist, rvec = False, tvec = False) 
        return retval, rvec, tvec
    else:
        return False, 0, 0 # 0s are to ensure three values are returned

def extract_laser(frame): 
    """_summary_

    Args:
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Isolate the red channel
    img = frame[...,2]
    
    # ret,img = cv2.threshold(img,144,255,0)

    # Create emptry array of zeros of same size as img
    out = np.zeros_like(img)

    img = cv2.GaussianBlur(img,(3,3),0)

    # For each row, get the position of the highest intensity
    bppr = np.argmax(img, axis=1)

    # Set the highest intensity pixel locations to 255
    out[np.arange(bppr.shape[0]), bppr] = 255
    
    return out, bppr

def get_laser_pts(img, POI):
    
    # Obtain the width and height of the camera
    h, w = img.shape

    # Get new camera matrix
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # dist b/w camera and laser is 4 in
    X = args.dist
    cam_angle = 30
    
    fx = new_mtx[0][0]
    fy = new_mtx[1][1]
    f = int(0.5*(fx+fy))
        
    # Scale factor for y direction
    Y_SCALE = 0.321

    # Camera params
    centre_x = int(w/2)
    centre_y = int(h/2)

    b0 = X
    C0 = cam_angle

    POI_len = POI.shape[0]
    
    # L_PTS (pixel coordinates)
    l_pts = np.zeros((POI_len,3))
    l_pts[:,0] = POI - centre_x
    l_pts[:,1] = centre_y - np.arange(POI_len)
    l_pts[:,2] = 100
    
    x = l_pts[:, 0]
    y = l_pts[:, 1]
    z = l_pts[:, 2]

    cam_pts = np.zeros((POI_len,3))
    
    delta_px_x0 = l_pts[:,0]
    A0 = 90 + np.degrees(np.arctan(delta_px_x0/f))
    B0 = 180 - (A0 + C0)
    a0 = (np.sin(np.deg2rad(A0))*b0)/np.sin(np.deg2rad(B0))    
    h1 = a0 * np.cos(np.arctan(delta_px_x0/f))
    
    cam_pts[:,2] = h1
    cam_pts[:,1] = l_pts[:,1] * Y_SCALE
    cam_pts[:,0] = 0
    
    return cam_pts

def transformed_points(undist):
    """_summary_

    Args:
        undist (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Grab the undistorted images and laser sample pair
    undist = cv2.imread(undist)
    
    # Perform pose detection on the undistorted images and obtain rvec, and tvec of the board
    retval, rvec, tvec = get_tf(undist, aruco_dict, parameters, board, mtx)
        
    if retval:
        
        # Get the 4x4 coordinate transform from the rvecs and tvecs
        tf = get_itf(rvec, tvec)

        # Perform triangulation on the laser samples and obtain mx3 matrix of points 
        laser, POI  = extract_laser(undist)
        cam_pts = get_laser_pts(laser, POI)
        
        # Add a column of ones to the mx3 matrix such that it is mx4
        cam_pts = np.hstack((cam_pts, np.ones((len(POI), 1))))
        
        # Transpose the matrix so that it is 4xm (i.e. each column is one point)
        cam_pts = cam_pts.transpose()
        
        print("Successfully performed operation!")
        
        # Multiply coordinate transofrm (4x4) for each point (4x1) to get a point that is transformed 
        return tf@cam_pts
    
    # Reject and log undected charuco board
    else:
        print("Could not perform pose detection on current frame!")
        return None

def exportPointCloud(point_cloud):
    """ Exports the point cloud into a ply file

    Args:
        point_cloud (ndarray): The final point cloud 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    dt = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    o3d.io.write_point_cloud(f"Project/point_cloud_{dt}.ply", pcd)

def displayPointCloud(point_cloud):
    from mpl_toolkits.mplot3d import proj3d

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # for out in full_pt_cloud:
        # print(out)
    ax.scatter(final_cloud[0,:], final_cloud[1,:], final_cloud[2,:], color='#ff0000')

    plt.show()

def read_charuco(dt, image):
    """Base pose estimation

    Args:
        dt (_type_): _description_
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    print("Start Pose Estimation Sequence...")
    allCorners = []
    allIds = []
    decimator = 0
    charuco_detected = False
    
    #SUB PIXEL CORNER DETECTION CRITERIA
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.00001)
    print("processing image{x}".format(dt))
    # frame = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        

def startScan():
    print("Starting Scan")
    global onFlag
    onFlag = True
    # ledRed.on()

def stopScan():
    print("Stopping Scan")
    global onFlag
    onFlag = False
    # ledRed.off()

def flash_green_LED():
    pass



#main function
def main():
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame by frame
        ret, frame = vid.read()
        
        #get the current timestamp of image 
        dt = datetime.now()
        
        # Rotate frame 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        #undistort the image before processing 
        undist = cv2.undistort(frame, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
        
        # allCorners,allIds,imsize, gray, charuco_detected = read_charuco(dt=dt, image=img_undistort)
        
        # if(charuco_detected):
        #     pass
        #     # if board is detected, run pose calculations and laser isolation and triangulation
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('undist', undist) 

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            onFlag = False
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()




#run the main function
if __name__ == '__main__':
    main()