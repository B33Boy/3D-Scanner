import RPi.GPIO as GPIO
import time
import cv2

but_pin =2
def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(but_pin,GPIO.IN)
    cam =cv2.VideoCapture(0)
    img_counter = 0
    print('Starting Demo Now')
    try:
        while True:
            print('Waiting for the button event')
            press_stage = GPIO.wait_for_edge(but_pin,GPIO.FALLING)
            print(press_stage)
            if press_stage == True:
                print('Button Pressed')
                ret, frame = cam.read()
                img_name = ('opencv_frame_{}.jpg'.format(img_counter))
                # cv2.imwrite(img_name,frame)
                print('{} written'.format(img_name))
            else:
                print('Closing')
                break
        finally:
            GPIO.cleanup()
            cam.release()
            cv2.destroyAllWindows()
if __name__ == '__main__':
    main()