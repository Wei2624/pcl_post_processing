import os
import cv2
import numpy as np
import sys
import scipy.io
import curvox
import pcl
import image_geometry
import random


# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import tf
import scipy.io
import math
import pickle

import transformer
import depth_pcl


def show_img_with_mask(img,mask):
	img[:,:,0] = np.multiply(img[:,:,0], mask)
	img[:,:,1] = np.multiply(img[:,:,1], mask)
	img[:,:,2] = np.multiply(img[:,:,2], mask)

	plt.imshow(img)
	plt.show()

def show_3D_plot(pcd):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(pcd[:,:,0].flatten(), pcd[:,:,1].flatten(), pcd[:,:,2].flatten())

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

def transformer_test_trim(pcd_i,):
	pcd = pcl_trim(pcd_i)

	# idx_test = np.argwhere(np.isnan(pcd[:,2]))
	idx_test = np.where(pcd[:,2] < 0)
	print "before transform:",idx_test[0].shape

	# Start to transform
	filename_meta = os.path.join(base_path,'{:04d}_meta.mat'.format(j))
	transformed = transformer.transform_view(pcd,filename_meta)

	# save_pcl(transformed, "after.pcd")

	# idx_test = np.argwhere(np.isnan(transformed[:,2]))
	# print "before trim: ",idx_test.shape

	idx_test = np.where(transformed[:,2] < 0)
	print "before trim: ",idx_test[0].shape

	trim_transformed = depth_pcl.pcl_trim(transformed)

	# idx_test = np.argwhere(np.isnan(trim_pcd[:,2]))
	# print np.argwhere(np.isnan(trim_transformed[:,2])).shape
	idx_test = np.where(trim_transformed[:,2] <0)
	print "after trim:",idx_test[0].shape

	# save_pcl(trim_transformed,"trimed.pcd")


def pcd_max_min_dist_viewer(pcd):
	print "min x, max x: ", np.min(pcd[:,0]), np.max(pcd[:,0])
	print "min y, max y: ", np.min(pcd[:,1]), np.max(pcd[:,1])
	print "min z, max z: ", np.min(pcd[:,2]), np.max(pcd[:,2])
	print "min distance, max distance: ", np.min(np.square(pcd[:,0]) + np.square(pcd[:,1]) + np.square(pcd[:,2])),\
										  np.max(np.square(pcd[:,0]) + np.square(pcd[:,1]) + np.square(pcd[:,2]))