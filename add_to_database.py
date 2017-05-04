import cv2, os

"""
title::

    add to database
description::

    This program takes a path to a database and the new image to be added to the database, makes a new folder
    within the database for the image and puts the image into that folder. The user is prompted to give a 
    folder name and an image name. 

attributes::

    path = path the database
    new_image = new image to be added to database

author::

    Emily Faw

date::
    20170504
"""

def add_to_database(path, new_image):

	dir_name = input('Enter name:')
	if os.path.exists(home + os.path.sep + path + dir_name)== True:
		face_image_name = input('Enter image number:')
		new_path = home + os.path.sep + path + dir_name + os.path.sep

	else:
		face_image_name = input('Enter image number:')
		os.mkdir(home + os.path.sep + path + dir_name)
		new_path = home + os.path.sep + path + dir_name + os.path.sep

	return(new_path, face_image_name)

if __name__ == '__main__':

	import cv2
	import fnmatch
	import numpy
	import os
	import os.path
	import time
	home = os.path.expanduser('~')

	new = home + os.path.sep + 'src/python/modules/ipcv/eigenface_images/obama.png'
	new_face = cv2.imread(new, cv2.IMREAD_UNCHANGED)


	dst, image = add_to_database('src/python/modules/ipcv/face_database/', new_face)
	cv2.imwrite(dst + image +'.jpg', new_face)

