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
import math
import pickle

from pcl_pixel_processing import pcl_processing
from pcl_pixel_processing import pcl_pixel_transform
from clustering import clustering_2D
from lib.util import point_inline_check
from lib.util import draw_contours
from lib.cfg_importer import cfg



def clean_tabletop_pcd(cam_model, table_mask,tabletop_pcd):
	tabletop_pcd_clean = np.zeros(tabletop_pcd.shape)
	# with open(path, 'rb') as input:
	# 	camera_info = pickle.load(input)

	# img_geo = image_geometry.PinholeCameraModel()
	# img_geo.fromCameraInfo(camera_info)
	index = 0
	for i in xrange(tabletop_pcd.shape[0]):
		coord_2d = cam_model.project3Dto2D(tabletop_pcd[i,:])
		coord_2d = list(coord_2d)
		coord_2d[0] = int(round(coord_2d[0]))
		coord_2d[1] = int(round(coord_2d[1]))
		if table_mask[coord_2d[1],coord_2d[0]] == 0:
			tabletop_pcd_clean[index,:] = tabletop_pcd[i,:]
			index += 1

	return tabletop_pcd_clean[:index,:]

def density_clustering(table_top_pcd_clean):
	return DBSCAN(eps=0.06, min_samples=100,algorithm='auto').fit(table_top_pcd_clean)


def seg_pcl_from_labels(cluster_labels,table_top_pcd_clean):
	obj_3d = []
	for i in range(np.unique(cluster_labels).shape[0]-1):
		idx = np.where(cluster_labels==i)
		obj_i = table_top_pcd_clean[idx,:]
		obj_i = np.squeeze(obj_i,axis=0)
		obj_3d.append(obj_i)

	return obj_3d

def clustering(cam_model, table_mask, table_top_pcd):
	table_top_pcd_clean = clean_tabletop_pcd(cam_model,table_mask,table_top_pcd)

	db = density_clustering(table_top_pcd_clean)
	cluster_labels = db.labels_
	obj_pcl = seg_pcl_from_labels(cluster_labels,table_top_pcd_clean)
	print np.unique(cluster_labels)
	# import ipdb
	# ipdb.set_trace()
	filtered_mask = table_mask.copy()
	mask_idx = 2
	for i in range(len(obj_pcl)):
		for j in range(obj_pcl[i].shape[0]):
			coord_2d = cam_model.project3Dto2D(obj_pcl[i][j,:])
			coord_2d = list(coord_2d)
			coord_2d[0] = int(round(coord_2d[0]))
			coord_2d[1] = int(round(coord_2d[1]))
			filtered_mask[coord_2d[1],coord_2d[0]] = mask_idx
		mask_idx += 1

	return filtered_mask,mask_idx
