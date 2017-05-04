"""
title::
	Facial Detection using Haar Cascades
description::
	A machine learning based approach using a cascade function trained
	from positive and negative images.
attributes::
	video - the video capture object that you are using to capture frames
	verbose - for testing uses to show work done
author::
	Geoffrey Sasaki
copyright::
	Copyright (C) 2017, Rochetser Institute of Technology
"""

import cv2
import numpy as np

def haar_cascades(video,verbose=False):
	
	face_cascade = cv2.CascadeClassifier('./FaceRecognitionModels/haarcascade_frontalface_default.xml')

	while video.isOpened():

		retrived, frame = video.read()		#Read in the video frame
		if retrived == False:
			print("You have reached the end of the video.")
			break
				
		gray = np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

		faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60,60))
		if len(faces) ==  0:
			faceBoxFrame = frame
		else:
			print("The size of the face box is {0} by {1}".format(faces[0,2],faces[0,3]))
			for (x, y, w, h) in faces:
				faceBoxFrame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

		if verbose:
			cv2.imshow("Haar Cascade Box", faceBoxFrame)


		k = cv2.waitKey(framerate)
		if k == 27 or k == (65536 + 27):
			print('Exiting...')
			break

	return faceBoxFrame

if __name__ == '__main__':
	import cv2
	import os
	import time

	currentDir = os.getcwd()

	testVid = currentDir + '/testVideo/MOVE_Video.mp4'

	capture = cv2.VideoCapture(testVid)
	if capture.isOpened() == False:
		msg = "The provided video file was not opened."
		raise ValueError(msg)

	framerate =	np.uint8(capture.get(5))

	verbose = True
	print("Press Esc to quit the video stream")

	faceBoxFrame = haar_cascades(capture, verbose)

	capture.release()
	cv2.destroyAllWindows()