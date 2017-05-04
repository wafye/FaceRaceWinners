import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        print("You have reached the end of the video.")
        break

    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('grayframe',gray)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    cv2.namedWindow('Haar Cascade Box', cv2.WINDOW_AUTOSIZE)
    
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(112,112))
    
    if len(faces) ==  0:
	    faceBoxFrame = frame
    else:
	    #print("The size of the face box is {0} by {1}".format(faces[0,2],faces[0,3]))
        for (x, y, w, h) in faces:

            faceBoxFrame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            cv2.imshow("Haar Cascade Box", faceBoxFrame)
            x1 = faces[0,0]
            y1 = faces[0,1]
            w1 = faces[0,2]
            h1 = faces[0,3]
            addition1 = int(h1//10)
            if y1-addition1 < 0:
                addition1 = 0
            
            if len(faces) == 1:
                cv2.namedWindow('FaceBox', cv2.WINDOW_AUTOSIZE)
                cv2.destroyWindow("FaceBox1")       
                cv2.destroyWindow("FaceBox2")



                eigenBox = frame[y1-addition1:h1+y1+addition1, x1:w1+x1]
                cv2.imshow("FaceBox", eigenBox)

            elif len(faces) == 2:
                cv2.namedWindow('FaceBox1', cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow('FaceBox2', cv2.WINDOW_AUTOSIZE)
                cv2.destroyWindow("FaceBox")

                x2 = faces[1,0]
                y2 = faces[1,1]
                w2 = faces[1,2]
                h2 = faces[1,3]
                addition2 = int(h2//10)


                if y2-addition2 < 0:
                    addition2 = 0

                eigenBox1 = frame[y1-addition1:h1+y1+addition1, x1:w1+x1]
                eigenBox2 = frame[y2-addition2:h2+y2+addition2, x2:w2+x2]

                cv2.imshow("FaceBox1", eigenBox1)
                cv2.imshow("FaceBox2", eigenBox2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
