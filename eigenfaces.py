"""
title::
	eigenfaces.py **********WITH DATABASE ADDITION
description::
	Face detection and recognition using eigenfaces
attributes::
	database - directory name for the database of training images (contains '
	subdirectioies for each subject)
	image - input image to test face detection and recognition
	f_t - threshold for face detection (maximum face space distance)
	nf_t - threshold for face recognition (maximum weight distance)
author::
	Jackson Knappen
copyright::
	Copyright (C) 2017, Rochetser Institute of Technology
"""

########################
# method definition
########################

def jack_eigenfaces(database, image, f_t, nf_t):

	#############################
	# get faces from the database
	database_face_list = []
	subject_dirs = [x[0] for x in os.walk(database)]
	for subject_dir in subject_dirs:
		faces = next(os.walk(subject_dir))[2]
		if (len(faces)>0):
			for face in faces:
				database_face_list.append(subject_dir + "/" + face)
	# read in the images from the database face list
	database_image_list = imread_collection(database_face_list)
	#############################

	# database and face dimensions
	M = len(database_image_list)
	n,m = np.shape(database_image_list[0])
	# print('database images:', M, ', image dimensions: (',n,m,')\n')

	# compute the average face
	average_face = np.sum(database_image_list, axis=0) / M
	# vectorize for further computations
	vect_average_face = average_face.flatten()
	# cv2.imshow('average_face',average_face.astype(np.uint8))
	# cv2.waitKey(0)

	# compute differences from the average face
	dif_vectors = []
	for i in range(M):
		# vectorize (flatten) the images
		vect_database_face = database_image_list[i].flatten()
		# compute the difference vectors
		dif = vect_database_face.astype(np.float64) - vect_average_face.astype(np.float64)
		dif_vectors.append(dif)
	# print('difference vectors list shape:',np.shape(dif_vectors))

	# decompose the difference vectors, computes eigenvectors, eigenvalues, and variance 
	U,S,V = np.linalg.svd(np.transpose(dif_vectors), full_matrices=False)
	# print('U, S, V shape:',np.shape(U), np.shape(S), np.shape(V))

	# sort,
	indx = np.argsort(-S)
	S = S[indx]
	U = U[:,indx]
	V = V[:,indx]

	# eigenface demonstration
	e = np.dot(np.transpose(dif_vectors), np.transpose(V))
	# print('eigenfaces shape:', np.shape(e))

	# compute corresponding weights for the eigenfaces
	# weights = np.dot(dif_vectors, U)
	weights = np.dot(dif_vectors, e)
	# print('weights shape:', np.shape(weights))

	#############################
	startTime = time.clock()

	# new face
	vect_im = image.flatten()
	im_dif_vect = vect_im - vect_average_face
	# compute set of weights by projecting the new image onto each of the existing eigenfaces
	weight_vect = np.dot(np.transpose(e), im_dif_vect)
	# print('weight vector shape:', np.shape(weight_vect))

	# determine if the new image is a face by measuring closeness to "face space"
	projection = np.dot(weight_vect, np.transpose(e))
	# print('projection shape:',np.shape(projection))

	# face space distance
	fs_distances = np.abs(im_dif_vect - projection)
	min_fs_distance = np.sqrt(np.min(fs_distances))
	# print('face space distances shape:',np.shape(fs_distances))
	print('\nminimum distance to face space:',min_fs_distance)

	# check if its a face based a face threshold value (f_t) 
	if min_fs_distance > f_t:
		print('\n#### NO FACE DETECTED ####\n')
		sys.exit()
	else:
		print('\n#### FACE DETECTED ####\n')

	# determine if the face is a match to an existing face
	weight_distances = np.sum( np.abs(weight_vect - weights), axis=1)
	min_weight_distance = np.sqrt(np.min(weight_distances))
	print('minimum weight distance:', min_weight_distance)

	# sort
	w_indx = np.argsort(weight_distances)
	ordered_weight_distances = weight_distances[w_indx]
	ordered_face_list = np.asarray(database_face_list)[w_indx]
	ordered_image_list = np.asarray(database_image_list)[w_indx]

	# check if the face matches a face in the existing database of faces
	if min_weight_distance > nf_t:
		print('\n#### NEW FACE ####\n')
		home = os.path.expanduser('~')
		#path = 'src/python/modules/ipcv/face_database/'
		dst, new_image = add_to_database('./face_database/', image)
		#dst, new_image = ipcv.add_to_database('src/python/modules/ipcv/face_database/', image)
		cv2.imwrite(dst + new_image +'.jpg', image)
		# elapsedTime = time.clock() - startTime
  #       print('Elapsed time = {0} [s]'.format(elapsedTime),'\n')
		# at this point, one could optionally add the new face to the database
        #add_to_database(database, image)


	else:
		print('\n#### MATCH FOUND ####\n')

		face_indx = np.where(ordered_weight_distances == ordered_weight_distances[0])[0][0]
		subject_id = ordered_face_list[face_indx]
		
		elapsedTime = time.clock() - startTime
		print('Elapsed time = {0} [s]'.format(elapsedTime),'\n')	

		print('closest face id:',face_indx, '(',subject_id,')\n')
		print('ten closest faces:', ordered_face_list[0:10])

		# cv2.imshow('closest face',ordered_image_list[face_indx].astype(np.uint8))
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		#############
		# display nine closest face images
		# row1 = np.hstack((ordered_image_list[face_indx],ordered_image_list[face_indx+1],ordered_image_list[face_indx+2]))
		# row2 = np.hstack((ordered_image_list[face_indx+3],ordered_image_list[face_indx+4],ordered_image_list[face_indx+5]))
		# row3 = np.hstack((ordered_image_list[face_indx+6],ordered_image_list[face_indx+7],ordered_image_list[face_indx+8]))
		# nine_faces_image = np.vstack((row1, row2, row3))
		# cv2.imshow('nine closest faces', nine_faces_image.astype(np.uint8))
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# cv2.imwrite('smiley_results.png', nine_faces_image.astype(np.uint8))
		#############



########################
# test harness
########################

if __name__ == '__main__':

	import os
	import sys
	#import ipcv
	import cv2
	import numpy as np
	from skimage.io import imread_collection
	import re
	import time

	home = os.path.expanduser('~')
	#baseDirectory = 'src/python/modules/ipcv/'
	database = './face_database/'
	# path = baseDirectory + database #+ os.path.sep
	
	#filename_image = home + os.path.sep + 'src/python/modules/ipcv/eigenface_images/obama.png'
	#filename_image = './testImages/lenna_color.tif'
	filename_image = './testImages/obama.png'
	#path = 'src/python/modules/ipcv/face_database/'
	#filename_image = home + os.path.sep + 'src/python/modules/ipcv/eigenface_images/mickey.jpg'


	image = cv2.imread(filename_image, cv2.IMREAD_GRAYSCALE)
	# cv2.imshow('image',image.astype(np.uint8))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


	maxCount = 255
	# face detection threshold
	# f_t = 5000
	f_t = 1000000000
	# face recognition threhsold
	nf_t = 15000
	#nf_t = 10000000000

	eigenface = jack_eigenfaces(database=database, image=image, f_t=f_t, nf_t=nf_t)