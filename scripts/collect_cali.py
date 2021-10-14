import cv2
import os

vid = cv2.VideoCapture(0)

dir = r'res/checkerboard/'

img_count = 0

print("Starting")

while(True):
      
    _, frame = vid.read()
  
    cv2.imshow('frame', frame)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord(' '):
        cv2.imwrite(dir + f'{img_count}.jpg', frame)
        img_count+=1

    elif pressedKey == ord(' '):
        if img_count > 0:
            os.remove(dir + f'{img_count}.jpg')
            img_count-=1

vid.release()
cv2.destroyAllWindows()


print("The following images were saved")
print(os.listdir(dir))
print("Finished")

