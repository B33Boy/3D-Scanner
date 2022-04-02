import cv2
import os

vid = cv2.VideoCapture(0)

output_dir = r'res/cal_in/batch_charuco/'

img_count = 0

print("Starting")

while(True):
      
    _, frame = vid.read()
    
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    cv2.imshow('frame', frame)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('c'):
        cv2.imwrite(output_dir + f'{img_count}.jpg', frame)
        img_count+=1

    elif pressedKey == ord('z'):
        if img_count > 0:
            img_count-=1
            os.remove(output_dir + f'{img_count}.jpg')
            

vid.release()
cv2.destroyAllWindows()


print("The following images were saved")
print(os.listdir(output_dir))
print("Finished")

