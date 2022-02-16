import cv2 
import numpy as np


def undistort_camera(img, mtx, new_mtx, roi, dist, w, h):

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    
    # crop the image
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]


def extract_laser(frame): 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_colour = np.array([146, 62, 0])
    upper_colour = np.array([255, 255, 255])
    
    # Threshold the HSV image to get only get red colour
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    
    # Isolate the red channel
    img = frame[...,2]
    ret,img = cv2.threshold(img,144,255,0)

    # Create emptry array of zeros of same size as img
    out = np.zeros_like(img)

    # For each row, get the position of the highest intensity
    bppr = np.argmax(img, axis=1)

    # Set the highest intensity pixel locations to 255
    out[np.arange(bppr.shape[0]), bppr] = 255
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(out,out, mask= mask)
    
    return res, bppr


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

# Global vars
x = 4 # dist b/w camera and laser in inches
D = np.arange(26, 6, -2) # 26 to 8 
offset = 0

count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # First undistort
    undist = undistort_camera(frame, mtx, new_mtx, roi, dist, w, h)

    frame, _ = extract_laser(undist)

    # Rotate frame 180 degrees
    cv2.imshow("Calibrate_theta", frame)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break
    # Capture image if spacebar is pressed
    elif k%256 == 32:
        # img_name = f"res/calibration_theta_input/dist_{count}_{D[count]}.png"
        img_name = f"res/pose_samples/pose_c_laser_{count}.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        undist_name = f"res/pose_samples/pose_c_undist_{count}.png"
        cv2.imwrite(undist_name, undist)
        print("{} written!".format(undist_name))

        count += 1

cam.release()

cv2.destroyAllWindows()



