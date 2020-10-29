from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2
import datetime
import csv
# import os

def minute_passed(oldpoch):
    return time.time() - oldpoch >=60
last_state=True
blink_count=0
epoch=time.time()





# from fastgrab import screenshot
import pyttsx3




with open('Driver_Detail.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Driver Name ','Time','Blinkinfo','YawnInfo','No of Alerted Time'])



pygame.mixer.init()
pygame.mixer.music.load('/home/suprim/PycharmProjects/fatigue/3 (copy)/audio/alert.wav')

EYE_AR_THRESH = 0.30
EYE_A_R_CONSEC_FRAMES = 25
COUNTER = 0
COUNTER_BLINK=0


sleep_flag = 0
yawn_flag = 0
count_mouth = 0
total_yawn = 0
total = 0



SHOW_POINTS_FACE = False
SHOW_CONVEX_HULL_FACE = False
SHOW_INFO = False


face_cascade = cv2.CascadeClassifier(
    "/home/suprim/PycharmProjects/fatigue/3 (copy)/haarcascades/haarcascade_frontalface_default.xml")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# MOUTHFUN

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[3], mouth[9])
    C = distance.euclidean(mouth[4], mouth[8])
    mar = (A + B+C) /3
    return mar


video_capture = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/suprim/PycharmProjects/fatigue/3 (copy)/shape_predictor_68_face_landmarks.dat')
# Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

ret, frame = video_capture.read()



dist_coeffs = np.zeros((4, 1))

start_time = time.time()
elapsed_time=start_time
# Give some time for camera to initialize(not required)
time.sleep(2)

while (True):
    # Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect facial points through detector function
    faces = detector(gray, 0)

    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect facial points
    for face in faces:

        shape = predictor(gray, face)

        #convert facial landmark x,y to numpy array
        shape = face_utils.shape_to_np(shape)

        # Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate aspect ratio of both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0


        mouth=shape[mStart:mEnd]
        mouthAR=mouth_aspect_ratio(mouth)


        if SHOW_CONVEX_HULL_FACE:
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            mouthHull=cv2.convexHull(mouth)
            cv2.drawContours(frame,[mouthHull],-1,(0,255,0),1)

            # jawHull = cv2.convexHull(jaw)
            # cv2.drawContours(frame, [jawHull], 0, (255, 255, 255), 1)

        # Detect if eye aspect ratio is less than threshold
        if (ear< EYE_AR_THRESH):
            COUNTER+= 1
            cv2.putText(frame, "EyeState:Closed", (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_A_R_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                print("you are sleeping")
                with open('Driver_Detail_.csv', 'w') as csvfile:





                    filewriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewriter.writerow(['Driver Name ', 'Time', 'Blinkinfo', 'No of Alerted Time'])
                    filewriter.writerow(['Suprim',datetime.datetime.now().strftime("%Y-%m-%d %H:%M") ,COUNTER_BLINK])




                cv2.putText(frame, "You are Sleeping", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 40, 255), 2)

                # take a full screen screenshot
                # img = screenshot.Screenshot().capture()
        else:
            cv2.putText(frame, "State: AWAKE", (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            pygame.mixer.music.stop()
            if COUNTER>1:
                COUNTER_BLINK+=1
                COUNTER=0



        if mouthAR > 30:
            count_mouth += 1
            if count_mouth >= 20:
                if yawn_flag < 0:
                    print("You are yawning")

                    yawn_flag = 1
                    total_yawn += 1
                    cv2.putText(frame, "Yawn Detected", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

                    with open('Driver_Yawn_Detail_.csv', 'w') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        filewriter.writerow(['Driver Name ', 'Time', 'Yawn_Count'])
                        filewriter.writerow(
                            ['Suprim', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),total_yawn])
                    # text_file=open("Output.txt","w")
                    # text_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+" FATIGUE")
                    # text_file.close()
                else:
                    yawn_flag = 1
            else:
                yawn_flag = -1
        else:
            count_mouth = 0
            yawn_flag = -1




        # engine.runAndWait()
        if total + total_yawn > 4:
            engine = pyttsx3.init()
            engine.say("You are yawinning frequently. I recommend you going for a walk or listen to music")
            engine.setProperty('volume', 0.9)
            engine.runAndWait()

            # os.system("espeak 'You are yawinning frequently. I recommend you going for a walk or listen to music'")
            total = 0
            total_yawn = 0


        # cv2.putText(frame, "Total Sleeps: {}".format(total), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)q
        cv2.putText(frame, "BLINKS: {}".format(COUNTER_BLINK), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(frame, "counter: {}".format(COUNTER), (20, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mouthAR), (300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(frame, "MAR: {:.2f}".format(count_mouth), (540, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "State: {:.2f}".format(count_mouth), (540, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # #Show video feed
    cv2.imshow('Montioring Window', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('l'):
        SHOW_CONVEX_HULL_FACE = not SHOW_CONVEX_HULL_FACE
    time.sleep(0.02)

video_capture.release()
cv2.destroyAllWindows()
