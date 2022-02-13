# Generic skeleton script for later use

# import the opencv library
import cv2 
  
#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# define a video capture object
vid = cv2.VideoCapture(0)


#create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls')
#create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('value','controls',0,100,nothing)


while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    value = int(cv2.getTrackbarPos('value','controls'))

    # Rotate frame 180 degrees
    #frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame = increase_brightness(frame, value=value)

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