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

def background_segmentation(frame, btype='AMOG', verbose=False):

	dimensions = frame.shape
	height = dimensions[0]
	width = dimensions[1]
	if len(dimensions) == 3:
		channels = dimensions[2]
	else:
		channels = 1
	dtype = frame.dtype

	morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE(3,3))

	longedge = np.max((width,height))
	if longedge > 500:
		scaleFactor = 500//longedge
		frameSize = (width*scaleFactor, height*scaleFactor)
	else:
		scaleFactor = 1
		frameSize = (width,height)

	frame = cv2.resize(frame, dsize=frameSize, fx=0, fy=0)

	if btype == 'AMOG':
		fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

	elif btype == 'MOG':
		fgbg = cv2.BackgroundSubtractorMOG()

	elif btype == 'GMG':
		fgbg = cv2.createBackgroundSubtractorGMG()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = gray.astype(np.float64)

	threshold = fgbg.apply(gray)
	threshold = cv2.morphologyEX(threshold,
									cv2.MORPH_CLOSE, morphKernel)
	
	if verbose:
		viewMask = fgbg.apply(frame)
		viewMask = cv2.morphologyEX(viewMask,
									cv2.MORPH_CLOSE, morphKernel)
		cv2.imshow("Threshold Mask", viewMask)

	threshold[threshold>0] = 1

	threshold.astype(dtype)

	return threshold

if __name__ == '__main__':
	import cv2
	import os
	import time

	currentDir = os.getcwd()

	testIm = currentDir + '/testImages/lenna.tif'
	testIm = currentDir + '/testImages/lenna_color.tif'

	im = cv2.imread(testIm, cv2.IMREAD_UNCHANGED)

	verbose = True
	btype = 'AMOG'

	startTime = time.process_time()
	threshold = background_segmentation(im, btype, verbose)
	endTime = time.process_time()
	print("It took {0}[s] to complete the background segmentation" \
			+ "operation".format(endTime-startTime))

	if verbose:
		delay = 100
		while True:
			k = cv2.waitKey(delay)

			if k == 27 or k == (65536 + 27):
				action = 'exit'
				print('Exiting...')
				break
			if k == 99 or k == (65536+99) or k == 67 or k == (65536+67):
				action = 'continue'
				print('Continuing...')
				break
		cv2.destroyAllWindows()