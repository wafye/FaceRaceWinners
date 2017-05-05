import numpy as np
import cv2

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
def add_to_database(path, new_image):

    dir_name = input('Enter name:')
    #dict_names = {}
    #if os.path.exists(home + os.path.sep + path + dir_name)== True:
    if os.path.exists( path + dir_name)== True:
        face_image_name = len(os.listdir(path+ dir_name)) + 1
        dict_names[dir_name] = face_image_name
        #new_path = home + os.path.sep + path + dir_name + os.path.sep
        new_path = path + dir_name + os.path.sep

    else:
        face_image_name = input('Enter image number:')
        #os.mkdir(home + os.path.sep + path + dir_name)
        os.mkdir(path + dir_name)
        new_path = path + dir_name + os.path.sep
        dict_names[dir_name] = face_image_name
        #new_path = home + os.path.sep + path + dir_name + os.path.sep

    return(new_path, face_image_name)

def eigenfaces(database, image, f_t, nf_t):

    print(image.shape)

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
        dst, new_image = add_to_database('face_database/', image)
        if len(os.listdir(dst)) < 9:
            cv2.imwrite(str(dst) + str(new_image) +'.jpg', image)


    else:
        print('\n#### MATCH FOUND ####\n')

        face_indx = np.where(ordered_weight_distances == ordered_weight_distances[0])[0][0]
        subject_id = ordered_face_list[face_indx]

        home = os.path.expanduser('~')
        #path = 'src/python/modules/ipcv/face_database/'
        # print(ordered_face_list[0])

        dst, new_image = add_to_database('face_database/', image)
        if len(os.listdir(dst)) < 9:
            cv2.imwrite(str(dst) + str(new_image) +'.jpg', image)
        
        elapsedTime = time.clock() - startTime
        # print('Elapsed time = {0} [s]'.format(elapsedTime),'\n')    

        # print('closest face id:',face_indx, '(',subject_id,')\n')
        # print('ten closest faces:', ordered_face_list[0:10])
        print(subject_id[10])
        # print(dict_names[str(subject_id[-4])])

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

# if __name__ == '__main__':

#     import os
#     import sys
#     #import ipcv
#     import cv2
#     import numpy as np
#     from skimage.io import imread_collection
#     import re
#     import time

#     home = os.path.expanduser('~')
#     #baseDirectory = 'src/python/modules/ipcv/'
#     database = './face_database/'
#     # path = baseDirectory + database #+ os.path.sep
    
#     #filename_image = home + os.path.sep + 'src/python/modules/ipcv/eigenface_images/obama.png'
#     #filename_image = './testImages/lenna_color.tif'
#     filename_image = './testImages/obama.png'
#     #path = 'src/python/modules/ipcv/face_database/'
#     #filename_image = home + os.path.sep + 'src/python/modules/ipcv/eigenface_images/mickey.jpg'


#     image = cv2.imread(filename_image, cv2.IMREAD_GRAYSCALE)
#     # cv2.imshow('image',image.astype(np.uint8))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


#     maxCount = 255
#     # face detection threshold
#     # f_t = 5000
#     f_t = 1000000000
#     # face recognition threhsold
#     nf_t = 15000
#     #nf_t = 10000000000

#     eigenface = eigenfaces(database=database, image=image, f_t=f_t, nf_t=nf_t)

def webcam(numFrames):

    frames = []

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./FaceRecognitionModels/haarcascade_frontalface_default.xml')

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
        cv2.imshow('grayframe', gray)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(112,112))
        
        if len(faces) != 0:
            for (x, y, w, h) in faces:

                if False: #DRAWING THE GREEN RECTANGLE
                    cv2.namedWindow('Haar Cascade Box', cv2.WINDOW_AUTOSIZE)
                    faceBoxFrame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.imshow("Haar Cascade Box", faceBoxFrame)

                x1 = faces[0,0]
                y1 = faces[0,1]
                w1 = faces[0,2]
                h1 = faces[0,3]
                add1 = h1//10
                if y1-add1 < 0:
                    add1 = 0
                
                if len(faces) == 1:
                    cv2.namedWindow('FaceBox', cv2.WINDOW_AUTOSIZE)
                    cv2.destroyWindow("FaceBox1")       
                    cv2.destroyWindow("FaceBox2")

                    eigenBox = frame[y1-add1:h1+y1+add1, x1:w1+x1]
                    cv2.imshow("FaceBox", eigenBox)

                    if len(frames) <= numFrames:
                        frames.append(eigenBox)

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(frames) == numFrames:
            return frames


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import os
    import sys
    #import ipcv
    import cv2
    import numpy as np
    from skimage.io import imread_collection
    import re
    import time
    from multiprocessing import Process

    # run database here (separate eigen function)

    # home = os.path.expanduser('~')
    #baseDirectory = 'src/python/modules/ipcv/'
    database = './face_database/'
    # path = baseDirectory + database #+ os.path.sep

    # maxCount = 255
    # face detection threshold
    # f_t = 5000
    f_t = 1000000000
    # face recognition threhsold
    nf_t = 15000
    #nf_t = 10000000000

    while True:
        numFrames = 10
        frames = webcam(numFrames)

        # eigenimage = np.asarray(frames[0])
        # eigenimage = cv2.resize(frames[0],(92,112))
        eigenimage = cv2.cvtColor(cv2.resize(frames[0],(92,112)),cv2.COLOR_BGR2GRAY)

        # return diction key of name to print name
        eigenface = eigenfaces(database,eigenimage,f_t,nf_t)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






