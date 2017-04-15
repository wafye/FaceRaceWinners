"""
title::
	Background Segmentation using an Adaptive Mixture of Gaussians
description::
	Computes the mask used to detect motion in a frame using
	an addaptive mixture of gaussians approach in opencv.
attributes::
	frame - the video frame who's background you want to segment
	btype - the type of background segmentation you wish to do
	verbose - for testing uses to show work done
author::
	Geoffrey Sasaki
copyright::
	Copyright (C) 2017, Rochetser Institute of Technology
"""

import cv2
import numpy as np

def background_segmentation(capture, btype='AMOG', verbose=False):

	width, height, framerate = int(capture.get(3)), int(capture.get(4)), \
										int(capture.get(5))
	longedge = np.max((width,height))
	if longedge > 1024:
		scaleFactor = 1024/longedge
		frameSize = (np.uint8(width*scaleFactor), np.uint8(height*scaleFactor))
	else:
		scaleFactor = 1
		frameSize = (width,height)

	morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

	if btype == 'AMOG':
		fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

	while capture.isOpened():
		retrived, frame = capture.read()		#Read in the video frame
		if retrived == False:
			print("You have reached the end of the video.")
			break
		
		frame = cv2.resize(frame, dsize=(0,0), 
							fx=scaleFactor, fy=scaleFactor)

		if verbose:
			cv2.imshow("Original Frame", frame)
		
		gray = np.float64(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
		threshold = fgbg.apply(gray)
		threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, morphKernel)
	
		if verbose:
			cv2.imshow("Threshold Mask", threshold)

		threshold[threshold>0] = 1
		maskedFrame = np.uint8(threshold*gray)

		if verbose:
			threshold = np.repeat(threshold[:,:,np.newaxis],
										repeats=3, axis=2)
			maskedFrame = np.uint8(threshold*frame)
			cv2.imshow("Masked Frame", maskedFrame)

		k = cv2.waitKey(framerate)
		if k == 27 or k == (65536 + 27):
			print('Exiting...')
			break

	return np.uint8(threshold)

if __name__ == '__main__':
	import cv2
	import os
	import time

	currentDir = os.getcwd()

	testVid = currentDir + '/testVideo/FunnyTestVideo.mp4'

	capture = cv2.VideoCapture(testVid)
	if capture.isOpened() == False:
		msg = "The provided video file was not opened."
		raise ValueError(msg)

	framerate =	np.uint8(capture.get(5))

	verbose = True
	btype = 'AMOG'
	print("Press Esc to quit the video stream")

	threshold = background_segmentation(capture, btype, verbose)

	capture.release()
	cv2.destroyAllWindows()