import os
import cv2
import numpy as np
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from subprocess import call
from subprocess import Popen, PIPE




for i in range(16,17):
	# print i
	base_path = '/home/weizhang/DA-RNN/data/LabScene/data/'  + '{:04d}/'.format(i)
	for j in xrange(0,50):
		print i,j
		path_full_pcd = os.path.join(base_path,'{:04d}_pcl.pcd'.format(j))
		path_plane_pcd = os.path.join(base_path,'{:04d}/'.format(j))

		if not os.path.exists(path_plane_pcd): os.makedirs(path_plane_pcd)
		path_plane_pcd = path_plane_pcd + '{:04d}'.format(j)

		process = Popen(['/home/weizhang/Documents/pcl_post_processing/table_finder/build/extract_indices', path_full_pcd, path_plane_pcd], stdout=PIPE, stderr=PIPE)
		stdout, stderr = process.communicate()
		print stdout
		print stderr


