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
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(112,112))
    
    if len(faces) ==  0:
	    faceBoxFrame = frame
    else:
	    #print("The size of the face box is {0} by {1}".format(faces[0,2],faces[0,3]))
        for (x, y, w, h) in faces:
            faceBoxFrame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            cv2.imshow("Haar Cascade Box", faceBoxFrame)

            
            addition = int(h/10)
            box = frame[y-addition:h+y+addition,x:w+x]
            #print(x, x-w, x+w)
            #print(x-w)
            #print(x+w)
            eigenBoxSize = (92, 112)
            cv2.imshow("Facebox", box)

            #centerW, centerH = (x+(w//2), y+(h//2))
            #print(center)
            #print(y, y+h, x, x+w)
            #cv2.imshow("FaceBox", frame[y:y+h, x:x+w])
            #cv2.imshow("EigenBox", frame[centerH-56:centerH+56, centerW-46:centerW+46])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()