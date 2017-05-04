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

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('grayframe',gray)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.namedWindow('Haar Cascade Box', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Facebox', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Facebox1', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Facebox2', cv2.WINDOW_AUTOSIZE)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(112,112))
    
    if len(faces) ==  0:
	    faceBoxFrame = frame
    else:
	    #print("The size of the face box is {0} by {1}".format(faces[0,2],faces[0,3]))
        for (x, y, w, h) in faces:

            if len(faces) == 1:
                cv2.destroyWindow("FaceBox1")       
                cv2.destroyWindow("FaceBox2")

                faceBoxFrame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("Haar Cascade Box", faceBoxFrame)
                cv2.imshow("FaceBox", frame[y:y+h, x:x+w])

            elif len(faces) == 2:
                cv2.destroyWindow("FaceBox")
                faceBoxFrame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("Haar Cascade Box", faceBoxFrame)
                
                cv2.imshow("FaceBox1", frame[faces[0,1]:faces[0,1]+faces[0,3], faces[0,0]:faces[0,0]+faces[0,2]])
                cv2.imshow("FaceBox2", frame[faces[1,1]:faces[1,1]+faces[1,3], faces[1,0]:faces[1,0]+faces[1,2]])

            #cv2.imshow("EigenBox", frame[centerH-56:centerH+56, centerW-46:centerW+46])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()