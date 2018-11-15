import os
import cv2
import numpy as np
import sys
import scipy.io
import curvox
import pcl


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

for i in range(16,22):
	print i
	rgbd_path = '/home/weizhang/DA-RNN/data/LabScene/data/'  + '{:04d}/'.format(i)
	for j in xrange(0,50):
		print j
		filename_label = os.path.join(rgbd_path,'{:04d}_label.png'.format(j))

		im = cv2.imread(filename_label)
		print im.shape

		im = im[...,[2,1,0]]

		cv2.imwrite(filename_label,im)







