import os
import cv2
import numpy as np
import sys
import scipy.io
import curvox
import pcl
import image_geometry
import random


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import tf
import scipy.io
import math
import pickle


for i in range(3,4):
	print i
	base_path = '/home/weizhang/DA-RNN/data/LabScene/data/'  + '{:04d}/'.format(i)
	for j in xrange(0,3):
		depth_path = os.path.join(base_path,'{:04d}_depth.png'.format(j))
		im_depth = cv2.imread(depth_path)

		# cv2.imwrite('test.png',im_depth[:,:,0])

		# cv2.imshow('name',im_depth)

		# cmap = plt.get_cmap('jet')

		# im_depth = cmap(im_depth)

		# cv2.imshow('image',im_depth)
		# cv2.waitKey(0)

		# plt.imshow(im_depth[:,:,0])
		plt.imsave(depth_path,im_depth[:,:,0])
		# plt.show()