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

dict_names = {}
def add_to_database(path, new_image, match_id, max_subject_faces):

	if match_id != None:
		# parse the string to get the name of the matched subject
		id_path = match_id.replace(path,'')[:-6]
		name = id_path.replace('./','')
		print('name:', name)
		dir_name = name

	else:
		 dir_name = input('Enter name:')

	#dict_names = {}
	#if os.path.exists(home + os.path.sep + path + dir_name)== True:
	if os.path.exists( path + dir_name)== True:
		face_image_number = len(os.listdir(path+ dir_name)) + 1
		dict_names[dir_name] = face_image_number
		#new_path = home + os.path.sep + path + dir_name + os.path.sep
		new_path = path + dir_name + os.path.sep

	else:
		# face_image_number = input('Enter image number:')
		face_image_number = 1
		#os.mkdir(home + os.path.sep + path + dir_name)
		os.mkdir(path + dir_name)
		new_path = path + dir_name + os.path.sep
		dict_names[dir_name] = face_image_number
		#new_path = home + os.path.sep + path + dir_name + os.path.sep

	return(new_path, face_image_number)


def read_database(database_path):
	# reads in the database and returns an imread_collection image list
	database_id_list = []
	subject_dirs = [x[0] for x in os.walk(database_path)]
	for subject_dir in subject_dirs[1:]:
		faces = next(os.walk(subject_dir))[2]
		if (len(faces)>0):
			for face in faces:
				database_id_list.append(subject_dir + "/" + face)
	# read in the images from the database face list
	database_images = skimage.io.imread_collection(database_id_list)

	return database_images, database_id_list

def eigenfaces_train(database_images):
	# eigenface database training

	# database and face dimensions
	M = len(database_images)
	#n,m = np.shape(database_images[0])
	# print('database images:', M, ', image dimensions: (',n,m,')\n')

	# compute the average face
	average_face = np.sum(database_images, axis=0) / float(M)
	# vectorize for further computations
	#print(average_face.shape)
	vect_average_face = average_face.flatten()
	#print(vect_average_face.shape)
	# cv2.imshow('average_face',average_face.astype(np.uint8))
	# cv2.waitKey(0)

	# compute differences from the average face
	dif_vectors = []
	for i in range(M):
		# vectorize (flatten) the images
		vect_database_face = database_images[i].flatten()
		# compute the difference vectors
		#dif = vect_database_face - vect_average_face

		#dif_vectors.append(dif)
		dif_vectors.append(vect_database_face - vect_average_face)

	# print('difference vectors list shape:',np.shape(dif_vectors))
	#print(dif_vectors)
	# decompose the difference vectors, computes eigenvectors, eigenvalues, and variance 
	U,S,V = np.linalg.svd(np.transpose(dif_vectors), full_matrices=False)
	#print('U, S, V shape:',np.shape(U), np.shape(S), np.shape(V))
	#print(unS)
	# sort,
	indx = np.argsort(-S)
	#S = unS[indx]
	#S = unS
	#U = unU[:,indx]
	#V = unV[:,indx]
	S, U, V = S[indx], U[:,indx], V[:,indx]

	# eigenface demonstration
	e = np.dot(np.transpose(dif_vectors), np.transpose(V))
	# print('eigenfaces shape:', np.shape(e))

	# compute corresponding weights for the eigenfaces
	# weights = np.dot(dif_vectors, U)
	weights = np.dot(dif_vectors, e)
	# print('weights shape:', np.shape(weights))

	return vect_average_face, weights, e

def eigenfaces_isFace(image, vect_average_face, e, f_t):
	im_dif_vect = image.flatten() - vect_average_face
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
		weight_vect = None
		
	else:
		print('\n#### FACE DETECTED ####\n')

	return weight_vect

def eigenfaces_detect(database_images, database_id_list, max_subject_faces,
						 image, vect_average_face, weights, e,  f_t, nf_t):
	"""
	# database and face dimensions
	#M = len(database_images)
	#n,m = np.shape(database_images[0])

	# new face
	#vect_im = image.flatten()
	#im_dif_vect = vect_im - vect_average_face
	im_dif_vect = image.flatten() - vect_average_face
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
		# sys.exit()
	else:
		print('\n#### FACE DETECTED ####\n')
	"""
	# determine if the face is a match to an existing face
	weight_distances = np.sum( np.abs(weight_vect - weights), axis=1)
	min_weight_distance = np.sqrt(np.min(weight_distances))
	print('minimum weight distance:', min_weight_distance)

	# sort
	w_indx = np.argsort(weight_distances)
	ordered_weight_distances = weight_distances[w_indx]
	ordered_id_list = np.asarray(database_id_list)[w_indx]
	ordered_images = np.asarray(database_images)[w_indx]



	# check if the face matches a face in the existing database of faces
	if min_weight_distance > nf_t:
		print('\n#### NEW FACE ####\n')
		retrain = False

		home = os.path.expanduser('~')
		#path = 'src/python/modules/ipcv/face_database/'
		subject_path, new_image = add_to_database('face_database/', image, match_id=None, max_subject_faces=max_subject_faces)
		if len(os.listdir(subject_path)) < max_subject_faces:
			cv2.imwrite(str(subject_path) + str(new_image) +'.jpg', image)
			# retrain the eigenfaces database if a new face is added
			retrain = True
			
	else:
		print('\n#### MATCH FOUND ####\n')
		retrain = False

		face_indx = np.where(ordered_weight_distances == ordered_weight_distances[0])[0][0]
		subject_id = ordered_id_list[face_indx]

		home = os.path.expanduser('~')
		#path = 'src/python/modules/ipcv/face_database/'
		# print(ordered_face_list[0])

		subject_path, new_image = add_to_database('face_database/', image, match_id=subject_id, max_subject_faces=max_subject_faces)
		if len(os.listdir(subject_path)) < max_subject_faces:
			cv2.imwrite(str(subject_path) + str(new_image) +'.jpg', image)  
			# retrain the eigenfaces database if a new face is added
			retrain = True

		print('closest face id:',face_indx, '(',subject_id,')')
		# print('ten closest faces:', ordered_face_list[0:10])

	return subject_path, retrain



def get_faceFrame(gray, face_cascade, consecutiveFrames):

	#consecutiveThreshold = 60
	#cap = cv2.VideoCapture(0)
	#face_cascade = cv2.CascadeClassifier('FaceRecognitionModels/haarcascade_frontalface_default.xml')
	#consecutive = 0
	"""
	while True:
	# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == False:
			print("You have reached the end of the video.")
			break

		frame = cv2.flip(frame,1)

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		#cv2.imshow('grayframe', gray)

		# Display the resulting frame
		cv2.imshow('frame', frame)
	"""
	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(112,112))
		
	if len(faces) != 0:
		for (x, y, w, h) in faces:
				"""
				if False: #DRAWING THE GREEN RECTANGLE
					cv2.namedWindow('Haar Cascade Box', cv2.WINDOW_AUTOSIZE)
					faceBoxFrame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
					cv2.imshow("Haar Cascade Box", faceBoxFrame)
				"""
				#x1 = faces[0,0]
				#y1 = faces[0,1]
				#w1 = faces[0,2]
				#h1 = faces[0,3]

				x1, y1, w1, h1 = faces[0,0], faces[0,1], faces[0,2], faces[0,3]
				add1 = h1//10
				if y1-add1 < 0:
					add1 = 0
				
				if len(faces) == 1:
					cv2.namedWindow('FaceBox', cv2.WINDOW_AUTOSIZE)
					#cv2.destroyWindow("FaceBox1")       
					#cv2.destroyWindow("FaceBox2")

					eigenBox = gray[y1-add1:h1+y1+add1, x1:w1+x1]
					displayBox = frame[y1-add1:h1+y1+add1, x1:w1+x1]
					cv2.imshow("FaceBox", displayBox)

					#if len(frames) <= numFrames:
					#	frames.append(eigenBox)
					consecutiveFrames.append(eigenBox)
				"""
				elif len(faces) == 2:
					cv2.namedWindow('FaceBox1', cv2.WINDOW_AUTOSIZE)
					cv2.namedWindow('FaceBox2', cv2.WINDOW_AUTOSIZE)
					cv2.destroyWindow("FaceBox")

					x2 = faces[1,0]
					y2 = faces[1,1]
					w2 = faces[1,2]
					h2 = faces[1,3]
					add2 = h2//10

					if y2-add2 < 0:
						add2 = 0

					eigenBox1 = frame[y1-add1:h1+y1+add1, x1:w1+x1]


					# eigenBox2 = frame[y2-add2:h2+y2+add2, x2:w2+x2]

					cv2.imshow("FaceBox1", eigenBox1)
					# cv2.imshow("FaceBox2", eigenBox2)
				"""
		#if cv2.waitKey(1) & 0xFF == ord('q'):
		#	msg = "User Has Stopped the Capture program by pressing 'q'"
		#	raise TypeError(msg)

			
	return consecutiveFrames

if __name__ == '__main__':
	import os
	import cv2
	import numpy as np
	import skimage.io

	database_path = 'face_database/'
	# maximum number of faces to be added to the database per subject
	max_subject_faces = 10
	# path = baseDirectory + database #+ os.path.sep
	cap = cv2.VideoCapture(0)
	if cap.isOpened() == False:
		msg = "The webcam was not able to be opened"
		raise ValueError(msg)

	cascade = cv2.CascadeClassifier(
				'FaceRecognitionModels/haarcascade_frontalface_default.xml')

	# face detection threshold
	f_t = 5000
	# f_t = 1000000000
	# face recognition threhsold
	nf_t = 15000
	#nf_t = 10000000000
	consecutiveThreshold = 60
	consecutiveFrames = []
	stop = False

	# initial read in
	database_images, database_id_list = read_database(database_path)
	# initial eigenfaces training
	vect_average_face, weights, e = eigenfaces_train(database_images)

	while not stop:
		k = cv2.waitKey(30)
		ret, frame = cap.read() #Capture frame-by-frame
		if ret == False:
			print("No Frames Were Able To Be Grabbed."
				" Check if Camera Was Disconnected or "
				"There are no more frames in the video file")
			break
		frame = cv2.flip(frame,1)
		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('frame', frame)
		consecutiveFrames = get_faceFrame(gray, cascade, consecutiveFrames)

		if len(consecutiveFrames) == consecutiveThreshold:
			eigenimage = cv2.resize(consecutiveFrames[0],(92,112))
			#eigenimage = cv2.cvtColor(cv2.resize(frames[0],(92,112)), cv2.COLOR_BGR2GRAY)

			# returns the file path of the matched subject
			weight_vect = eigenfaces_isFace(eigenimage, 
											vect_average_face, e, f_t)
			if weight_vect != None:
				subject_path, retrain = eigenfaces_detect(database_images, 
										database_id_list, max_subject_faces, eigenimage, 
										vect_average_face, weights, e, f_t, nf_t)
				# if len(os.listdir(subject_path)) < max_subject_faces:
				if retrain == True:
					print('#### RETRAINING ####')
					database_images, database_id_list = read_database(database_path)
					vect_average_face, weights, e = eigenfaces_train(database_images)

			consecutiveFrames = []


		if k & 0xFF == ord('q'):
			print("Stopping Video Capture and Eigen Face Detection")
			stop = True


		

	cap.release()
	cv2.destroyAllWindows()



