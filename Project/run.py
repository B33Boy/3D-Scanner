# Import installed libraries
from operator import truediv
import cv2 

def detect_charuco():
    pass

def laser():
    pass



#main function
def main():
    vid = cv2.VideoCapture(0)
    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Rotate frame 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)

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

#run the main function
if __name__ == '__main__':
    main()