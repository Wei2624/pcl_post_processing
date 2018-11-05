import os
import cv2
import numpy as np
import sys
import scipy.io


for i in range(0,1):
	print i
	rgbd_path = '/home/weizhang/DA-RNN/data/LabScene/data/'  + '{:04d}/'.format(i)
	for j in xrange(0,100):
		print j
		filename_label = os.path.join(rgbd_path,'{:04d}_label.png'.format(j))

		im = cv2.imread(filename_label,-1)


		kernel = np.ones((19,19),np.uint8)
		im_ero = cv2.erode(im,kernel,iterations=1)

		cv2.imshow('Frame',im)
		cv2.waitKey(0)
 
		# Press Q on keyboard to  exit
		# if cv2.waitKey(25) & 0xFF == ord('q'):
		# 	break
        







