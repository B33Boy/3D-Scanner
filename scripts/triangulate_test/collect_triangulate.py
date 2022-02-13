import cv2 
import numpy as np


def extract_laser(frame): 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_colour = np.array([146, 62, 0])
    upper_colour = np.array([255, 255, 255])
    
    # Threshold the HSV image to get only get red colour
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    
    # Isolate the red channel
    img = frame[...,2]
    ret,img = cv2.threshold(img,225,255,0)

    # Create emptry array of zeros of same size as img
    out = np.zeros_like(img)

    # For each row, get the position of the highest intensity
    bppr = np.argmax(img, axis=1)

    # Set the highest intensity pixel locations to 255
    out[np.arange(bppr.shape[0]), bppr] = 255
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(out,out, mask= mask)
    
    return res

# Global vars
x = 4 # dist b/w camera and laser in inches
D = np.arange(26, 6, -2) # 26 to 8 
offset = 0

count = 0

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    # Rotate frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # frame = extract_laser(frame)
    cv2.imshow("Calibrate_theta", frame)

    k = cv2.waitKey(1)

    # Exit if 'q' is pressed
    if k%256 == 113: 
        print("Escape hit, closing...")
        break
    # Capture image if spacebar is pressed
    elif k%256 == 32:
        img_name = f"res/marker_test/marker_{count}_bright.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        count += 1

cam.release()

cv2.destroyAllWindows()



