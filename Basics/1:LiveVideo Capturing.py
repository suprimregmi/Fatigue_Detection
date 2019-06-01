#Live video capturing with OpenCV
import cv2
import numpy as np

cap=cv2.VideoCapture(0) #0/1 is default

while True:
    ret,frame =cap.read()
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xF  #0xF is mask for 64bit machine
    if key == ord('q'):  #on pressing q key program will terminate
        break
cap.release()
cv2.destroyAllWindows()
