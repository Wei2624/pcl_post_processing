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

import math
import pickle

def save_pcl(pcd, name):
	pc = pcl.PointCloud()
	pc.from_array(pcd.astype(np.float32))
	pc.to_file(name)


def table_plane_pcl(p, path):
	table_pcd = []
	min_distance = 10000000000000
	# p = pcl.PointCloud()
	for f in os.listdir(path):
		p.from_file(os.path.join(path, f))
		pcd = p.to_array()
		dist = np.min(np.square(pcd[:,0]) + np.square(pcd[:,1]) + np.square(pcd[:,2]))
		if dist < min_distance:
			min_distance = dist
			table_pcd = pcd

	return table_pcd

def find_normal_vector(pcd):
	rows = pcd.shape[0]
	normal_vector = 0
	max_itr = 20
	for itr in xrange(0,max_itr):
		idx = random.sample(range(0,rows),3)
		point1 = pcd[idx[0],:]
		point2 = pcd[idx[1],:]
		point3 = pcd[idx[2],:]

		vector1 = point2 - point1
		vector2 = point3 - point1


		test_vector = -point1


		normal_vector += np.cross(vector1,vector2)

	if np.dot(normal_vector,test_vector) < 0 : return -normal_vector/max_itr
	else: return normal_vector/max_itr

# def check_ele_list(elem,lst):
# 	try:
# 		return lst.index(elem)
# 	except ValueError:
# 		return -1



def pcl_above_plane(plane_pcl,all_pcl):
	anchor_point = np.mean(plane_pcl,axis=0)

	roi_pcl = np.zeros(all_pcl.shape)
	index = 0

	normal_vector = find_normal_vector(plane_pcl)

	for i in xrange(all_pcl.shape[0]):
		if np.dot(normal_vector,all_pcl[i,:] - anchor_point) > 0 :
			roi_pcl[index,:] = all_pcl[i,:]
			index += 1
	return roi_pcl[:index,:]