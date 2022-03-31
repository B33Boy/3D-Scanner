# Generic skeleton script for later use

# import the opencv library
import cv2
from gpiozero import Button, LED
from time import sleep

# Define GPIO Pin I/O
ledGreen = LED(23)
ledRed = LED(24)
btnStart = Button(17)
btnStop = Button(27)

# define a video capture object
vid = cv2.VideoCapture(0)
  
onFlag = False

def onState():
    print("STATE: ON")
    onFlag = True
    ledRed.on()

def offState():
    print("STATE: OFF")
    onFlag = False
    ledRed.off()

#TODO: MODIFY CODE TO TOGGLE WITH ONE BUTTON AND SAVE DATA WITH ANOTHER
# def onState():
#     print("SOPPING RECORDING")
#     ledRed.on()
#     if onFlag == True:
#         print("STOPPING RECORDING")
#         ledRed.on()
#         ledGreen.off()
#     else:
#         print("STARTING RECORDING")
#         ledRed.on()
#         ledGreen.off()
    
#     onFlag = not onFlag

while(True):
    btnStart.when_pressed = onState
    btnStop.when_pressed = offState

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Rotate frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    cv2.putText(frame, 
                str(onFlag), 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

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
