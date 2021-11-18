import cv2
import numpy as np
  
vid = cv2.VideoCapture(0)

def extract_laser(frame): 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_colour = np.array([146, 62, 0])
    upper_colour = np.array([255, 255, 255])
    
    # Threshold the HSV image to get only get red colour
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    
    # Isolate the red channel
    img = frame[...,2]
    ret,img = cv2.threshold(img,230,255,0)

    print(img.shape)
    # Create emptry array of zeros of same size as img
    out = np.zeros_like(img)

    # For each row, get the position of the highest intensity
    bppr = np.argmax(img, axis=1)

    # Set the highest intensity pixel locations to 255
    out[np.arange(bppr.shape[0]), bppr] = 255
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(out,out, mask= mask)
    
    return res
    

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()


    laser = extract_laser(frame)
    
    # Display the resulting frame
    cv2.imshow('frame', laser)
    
    # the 'q' button is set as the
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


