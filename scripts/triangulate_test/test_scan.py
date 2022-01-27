import numpy as np
import matplotlib.pyplot as plt
import cv2


def undistort_camera(img, mtx, new_mtx, roi, dist, w, h):

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    # crop the image
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]

# Load calibration matrix
with np.load('res/calibration_theta_output/theta_params.npz') as X:
    theta_coeff = X['theta_coeff']

# Load previously saved data
with np.load('res/cal_out/cam_params.npz') as X:
    mtx, dist, rot_vectors, trans_vectors = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

# print(mtx.shape)
# print(dist.shape)
# print(rot_vectors.shape)
# print(trans_vectors.shape)

cam = cv2.VideoCapture(0)
count = 0

# Camera params
w = 640
h = 480
centre_x = w/2
#dist b/w camera and laser is 4 in
X = 4

# Undistort Camera
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    # First undistort
    frame = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)

    # Next extract the laser + matrix containing points of interest
    frame, POI = extract_laser(frame)
    
    cv2.imshow("cam_img", frame)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break
    # Capture image if spacebar is pressed
    elif k%256 == 32:
        img_name = f"res/calibration_theta_output/test_scan_{count}.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        count += 1
        
        # Get distance
        pix_dist = POI - centre_x
        theta = np.multiply(theta_coeff[86:335,0], pix_dist[86:335]) + theta_coeff[86:335,1]
        # print(theta.shape)
        D = X*np.tan(theta)
        print(D)
    
        
        
        

cam.release()

cv2.destroyAllWindows()


