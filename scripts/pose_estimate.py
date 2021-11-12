import cv2
import numpy as np
import glob 
from cv2 import CAP_PROP_AUTOFOCUS, aruco
# Load previously saved data
with np.load('res/calibration_output/cam_params.npz') as X:
    mtx, dist, rotation_vectors, translation_vectors = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

# print(mtx)

# print(dist)

'''
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for fname in glob.glob('res/calibration_input/*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)
    # print(corners.shape)
    # print(corners[0])
    # print("\n\n\n")

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
   
           # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
   
        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)
   
cv2.destroyAllWindows()
'''
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_50)

board = aruco.GridBoard_create(
    markersX = 6,
    markersY = 8,
    markerLength = 0.04,
    markerSeparation = 0.02,
    dictionary = ARUCO_DICT)

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    ret, img = cam.read()
    if ret == True:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        #detect markers
        corners, ids, rejectedPoints = aruco.detectMarkers(grey, ARUCO_DICT, parameters = ARUCO_PARAMETERS)
        #refine detected markers
        corners, ids, rejectedPoints, recoveredIds = aruco.refineDetectedMarkers(
            image = grey,
            board = board,
            detectedCorners = corners,
            detectedIds = ids,
            rejectedCorners = rejectedPoints,
            cameraMatrix = mtx,
            distCoeffs = dist)
        #outline markers
        img = aruco.drawDetectedMarkers (img, corners, ids, borderColor = (255,0,0))
        
        if ids != None:
           charucoretval, charucooCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, grey, board)
           charucoBoard = aruco.drawDetectedCornersCharuco(img, charucooCorners, charucoIds, (0,255,0))
           retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoBoard, charucoIds, board, mtx, dist)
           
           if retval == True:
               charucoBoard = aruco.drawAxis(charucoBoard, mtx, dist, rvec, tvec, 100) #last one is axis length to be changed
        
        else:
            charucoBoard = img
            
        cv2.imshow("charucoboard", charucoBoard)
        cv2.waitKey(0)

cv2.destroyAllWindows()

        
        