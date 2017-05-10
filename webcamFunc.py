"""
title::
	Facial Detection and Recognition using Eigenfaces and Haar Cascades

description::
	An implementation of a unique facial detection and recongition algorithm
	that utalises Haar Cascades for initial positioning and detection of
	faces, then implements eigenfaces to match them to a subject. This 
	current version begins with an AT&T Database of eigenfaces and the default
	Haar Frontal Face Detection Cascade Model of detection.

dependencies::
	Python 3.6 or later
	Opencv3 or later
	numpy + OpenCV Dependencies
	scikit-image (aka skimage)
	Tested and Validated on Windows 10

attributes::
	Add To Database 		- Adds a image to the database of faces
		Path 				- Database Path Typically './face_database/'
		match_id 			- Path to Matched Image 'face_database/*Person*/ID#.jpg'
		validate 			- If matched images are close enough no check is required

		new_path			- Directory that image was saved to
		face_image_number	- The current image # in the directory + 1

	Read_Database
		database_path 		- Database Path Typically './face_database/'

		database_id_list 	- List of all of the path name images in the database
		database_images 	- All of the images in the database as arrays

	EigenFaces_Train
		database_images 	- List of all images in database as arrays

		vect_average_face	- Flattened Version of the "Average Face"
		weights 			- The weights of all of the eigen vectors (400, 400)
		e					- Eigen Vectors of all the images (10304x400)

	EigenFaces_IsFace
		image 				-
		Average Face Vector -	
		EigenFace Vector 	- 
		EigenFace Threshold -

	eigenfaces_detect 
		database_id_list
		max_subject_faces 
		image
		weights 
		nf_t
		
	get_FaceFrame
		Frame
		Face Cascade
		Consecutive Faces

author::
	Trevor Canham, Emily Faw, Jackson Knappen, Makayla Roof, Geoffrey Sasaki

copyright::
	Copyright (C) 2017, Rochetser Institute of Technology
"""
import tkinter

class takeInput(object):

	def __init__(self,requestMessage):
		self.root = tkinter.Tk()
		self.string = ''
		self.frame = tkinter.Frame(self.root)
		self.frame.pack()        
		self.acceptInput(requestMessage)

	def acceptInput(self,requestMessage):
		r = self.frame

		k = tkinter.Label(r,text=requestMessage)
		k.pack(side='left')
		self.e = tkinter.Entry(r,text='Name')
		self.e.pack(side='left')
		self.e.focus_set()
		b = tkinter.Button(r,text='okay',command=self.gettext)
		b.pack(side='right')

	def gettext(self):
		self.string = self.e.get()
		self.root.destroy()

	def getString(self):
		return self.string

	def waitForInput(self):
		self.root.mainloop()

def getText(requestMessage):
	msgBox = takeInput(requestMessage)
	#loop until the user makes a decision and the window is destroyed
	msgBox.waitForInput()
	return msgBox.getString()
# var = getText('enter your name')
# print ("Var:", var)

def add_to_database(path, match_id, max_subject_faces, eigenimage, 
												frameSize, validate=True):
	retrain = False

	if match_id is not None:
		# parse the string to get the name of the matched subject
		dir_name = match_id[14:][:-6]

		if validate:
			#validation = getText('Are You {0}? [y/n]'.format(dir_name.title()))
			validation = input('Are You {0}? [y/n] '.format(dir_name.title()))
			if validation.strip() == 'y':
				face_image_number = len(os.listdir(path+ dir_name)) + 1
				if len(os.listdir(path+dir_name)) == max_subject_faces:
					#farthest = replaceFarthestMatch(path+dir_name, eigenimage)
					replaceFarthestMatch(path+dir_name, eigenimage, frameSize)
					retrain = True

			elif validation.strip() == 'n':
				match_id = None
				retrain, dir_name = add_to_database(path, 
										match_id, max_subject_faces,
										eigenimage,frameSize, validate)
			else:
				msg = "User did not enter a 'y' or 'n'."
				raise TypeError(msg)
		else:
			face_image_number = len(os.listdir(path + dir_name)) + 1

		#print('The Closest Face ID in the database is','(',match_id[14:],')',
		#		"The distance away is", min_weight_distance, " units in faceSpace")

	else:

		#var = getText('enter your name')
		# print ("Var:", var)
		dir_name = input('Enter name: ').lower()

		if dir_name == "quit" or dir_name == "Quit" or dir_name == 'q':
			msg = "Getting out of the face race program"
			raise TypeError(msg)
		face_image_number = 0

		if os.path.exists(path + dir_name):
			print("This person already has a directory for them,"
					" Adding Variety")

			if len(os.listdir(path+dir_name)) == max_subject_faces:

				#OVERWRITE IMAGE WHO'S MATCH IS FARTHEST IN THE DIRECTORY
				#farthest = replaceFarthestMatch(path+dir_name, eigenimage)
				replaceFarthestMatch(path+dir_name, eigenimage, frameSize)

				retrain = True
			else:
				face_image_number = len(os.listdir(path + dir_name)) + 1
				cv2.imwrite(path+dir_name+os.path.sep + str(face_image_number) 
													+'.jpg', eigenimage) 
				retrain = True
		else:
			os.mkdir(path + dir_name)
		
	new_path = path + dir_name + os.path.sep
	
	if len(os.listdir(new_path)) < max_subject_faces:
		cv2.imwrite(str(new_path) + str(face_image_number) +'.jpg', eigenimage) 
		retrain = True

	return retrain, dir_name

def replaceFarthestMatch(imageDirectory, eigenimage, frameSize):
	order = 'worst'
	nf_t = np.inf
	font = cv2.FONT_HERSHEY_SIMPLEX

	database_images, database_id_list = read_database(imageDirectory)
	vect_avg_faces, weights, e = eigenfaces_train(database_images)
	weight_vect = eigenfaces_isFace(eigenimage, vect_average_face, e, 100000)
	subject_id, min_weight_distance = eigenfaces_detect(database_id_list, 
											weight_vect, weights, nf_t, order)
	print("The Farthest Face ID is ",subject_id[14:])
	print("It is ", min_weight_distance, "Units away from the current image")
	print("Overwriting the ID with the current image\n")
	badImage = cv2.imread(subject_id)
	replacementImage = np.zeros((112,92*2,3))
	replacementImage[0:112,0:92,:] = badImage[:,:,:].astype(np.uint8)
	replacementImage[0:112,92:92*2,:] = np.repeat(eigenimage[:,:,np.newaxis],3,axis=2)
	replacementImage = cv2.resize(replacementImage, (frameSize))
	print(frameSize)
	cv2.putText(replacementImage, "Old", (140,480), font, 1, (0,0,255), 
		2, cv2.LINE_AA)	
	cv2.putText(replacementImage, "New", (450, 480), font, 1, (0,0,255),
		2, cv2.LINE_AA)
	cv2.imshow("Replacement Window", replacementImage.astype(np.uint8))
	cv2.waitKey(30)
	cv2.imwrite(subject_id, eigenimage)


	return subject_id

def read_database(database_path):
	# reads in the database and returns an imread_collection image list
	database_id_list = []
	subject_dirs = [x[0] for x in os.walk(database_path)]
	for subject_dir in subject_dirs:
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

	# compute the average face
	average_face = np.sum(database_images, axis=0) / float(M)
	# vectorize for further computations

	vect_average_face = average_face.flatten()
	#print(vect_average_face.shape)

	# compute differences from the average face
	dif_vectors = []
	for i in range(M):
		# vectorize (flatten) the images
		vect_database_face = database_images[i].flatten()
		# compute the difference vectors
		dif_vectors.append(vect_database_face - vect_average_face)

	# decompose the difference vectors, computes eigenvectors, 
	# eigenvalues, and variance 
	U,S,V = np.linalg.svd(np.transpose(dif_vectors), full_matrices=False)

	# sort,
	indx = np.argsort(-S)

	S, U, V = S[indx], U[:,indx], V[:,indx]

	# eigenface demonstration
	e = np.dot(np.transpose(dif_vectors), np.transpose(V))

	# compute corresponding weights for the eigenfaces
	weights = np.dot(dif_vectors, e)

	return vect_average_face, weights, e

def eigenfaces_isFace(image, vect_average_face, e, f_t):
	im_dif_vect = image.flatten() - vect_average_face
	# compute set of weights by projecting the new image onto each 
	# of the existing eigenfaces
	weight_vect = np.dot(np.transpose(e), im_dif_vect)

	# determine if the new image is a face by 
	# measuring closeness to "face space"
	projection = np.dot(weight_vect, np.transpose(e))
	# print('projection shape:',np.shape(projection))

	# face space distance
	fs_distances = np.abs(im_dif_vect - projection)
	min_fs_distance = np.sqrt(np.min(fs_distances))
	# print('face space distances shape:',np.shape(fs_distances))
	#print('\nminimum distance to face space:',min_fs_distance)

	# check if its a face based a face threshold value (f_t) 
	if min_fs_distance > f_t:
		print(min_fs_distance)
		print('\n#### THE FACE DETECTED IS NOT CLOSE ENOUGH TO FACESPACE ####\n')
		weight_vect = None

	#else:
	#	print('\n#### FACE DETECTED ####\n')

	return weight_vect

def eigenfaces_detect(database_id_list, weight_vect, weights, nf_t, order):

	# determine if the face is a match to an existing face
	weight_distances = np.sum( np.abs(weight_vect - weights), axis=1)
	min_weight_distance = np.sqrt(np.min(weight_distances))

	# sort
	if order == 'best':
		#print('weight distance from best match:', min_weight_distance)
		w_indx = np.argsort(weight_distances)
	else:
		#print("weight distance from worst match for subject", min_weight_distance)
		w_indx = np.argsort(-weight_distances)
	ordered_weight_distances = weight_distances[w_indx]
	ordered_id_list = np.asarray(database_id_list)[w_indx]
	#print(ordered_id_list)


	# check if the face matches a face in the existing database of faces
	if min_weight_distance > nf_t:
		print('##########################################')
		print('##########################################')
		print('####                                  ####')
		print('####             NEW FACE             ####')
		print('####                                  ####')
		print('##########################################')
		print('##########################################\n')

		match_id = None

	else:
		match_id = ordered_id_list[0]
		if order == 'best':
			print('\n###########################################')
			print('####           MATCH  FOUND            ####')
			print('###########################################\n')
			print("The closest face ID is: ", ordered_id_list[0][14:])
			print("The distance away from the rest of the subject images is", min_weight_distance)
			print('\n')


	return match_id, min_weight_distance

def get_faceFrame(gray, face_cascade, consecutiveFrames):

	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(112,112))
		
	if len(faces) != 0:
		for (x, y, w, h) in faces:
			x1, y1, w1, h1 = faces[0,0], faces[0,1], faces[0,2], faces[0,3]
			add1 = h1//10
			if y1-add1 < 0:
				add1 = 0
				
			if len(faces) == 1:
				cv2.namedWindow('FaceBox', cv2.WINDOW_AUTOSIZE)

				eigenBox = gray[y1-add1:h1+y1+add1, x1:w1+x1]
				displayBox = frame[y1-add1:h1+y1+add1, x1:w1+x1]
				cv2.imshow("FaceBox", displayBox)

				consecutiveFrames.append(eigenBox)
	else:
		consecutiveFrames = []
			
	return consecutiveFrames

def colorPresence(frame):
	presence = False
	lower = np.array([0,10,30], dtype="uint8")
	upper = np.array([20,150,255], dtype="uint8")
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))\

	hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(hsvFrame, lower, upper)
	skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
	cv2.imshow("SkinMask", skinMask)
	#skinMask[skinMask==255] = 1
	if np.sum(skinMask) > 0:
		presence = True

	return presence

if __name__ == '__main__':
	import os
	import cv2
	import numpy as np
	import skimage.io

	database_path = 'face_database/'
	# maximum number of faces to be added to the database per subject
	cap = cv2.VideoCapture(0)
	if cap.isOpened() == False:
		msg = "The webcam was not able to be opened"
		raise ValueError(msg)

	cascade = cv2.CascadeClassifier(
				'FaceRecognitionModels/haarcascade_frontalface_default.xml')
	
	max_subject_faces = 40
	# face detection threshold
	f_t = 10000
	# face recognition threhsold
	nf_t = 16000
	#nf_t = 10000000000
	v_t = 12500 #Validation Threshold
	consecutiveThreshold = 20
	consecutiveFrames = []
	stop = False
	validate = True
	order = 'best'
	font = cv2.FONT_HERSHEY_SIMPLEX
	width, height = int(cap.get(3)), int(cap.get(4))
	found = None
	frameSize = (width,height)

	# initial read in
	database_images, database_id_list = read_database(database_path)
	# initial eigenfaces training
	vect_average_face, weights, e = eigenfaces_train(database_images)

	while not stop:
		k = cv2.waitKey(1)
		ret, frame = cap.read() #Capture frame-by-frame
		if ret == False:
			print("No Frames Were Able To Be Grabbed."
				" Check if Camera Was Disconnected or "
				"There are no more frames in the video file")
			break
		frame = cv2.flip(frame,1)

		presence = colorPresence(frame)

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		if found is not None:

			cv2.putText(frame, found.rstrip('/'), (10,30), font, 1, (0,255,0), 
																2, cv2.LINE_AA)
		cv2.imshow("fame", frame)

		consecutiveFrames = get_faceFrame(gray, cascade, consecutiveFrames)
		
		if presence:
			if len(consecutiveFrames) == consecutiveThreshold:
				eigenimage = cv2.resize(consecutiveFrames[-1],(92,112))

				# returns the file path of the matched subject
				weight_vect = eigenfaces_isFace(eigenimage, 
												vect_average_face, e, f_t)
				if weight_vect is not None:
					match_id, min_weight_distance = eigenfaces_detect(
												database_id_list, weight_vect,
												weights, nf_t, order)

					if min_weight_distance < v_t:
						validate = False

					retrain, dir_name = add_to_database(database_path, match_id, 
							max_subject_faces, eigenimage, frameSize, validate)

					found = dir_name.title()

					if retrain:
						print('\n\n\n#### RETRAINING ####\n\n\n')
						database_images, database_id_list = read_database(database_path)
						vect_average_face, weights, e = eigenfaces_train(database_images)
						cv2.destroyWindow("Replacement Window")


				consecutiveFrames = []

		if k & 0xFF == ord('q'):
			print("Stopping Video Capture and Eigen Face Detection")
			stop = True

	cap.release()
	cv2.destroyAllWindows()



