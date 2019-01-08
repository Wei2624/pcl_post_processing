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
import time

from lib.cfg_importer import cfg
from pcl_pixel_processing import pcl_processing
from pcl_pixel_processing import pcl_pixel_transform
from clustering import clustering_2D
from clustering import clustering_3D
from lib.util import point_inline_check
from lib.util import draw_contours
from lib.util import label_pcd
from lib.py_wrapper import find_table_plane
from vote import voting 
from pcl_pixel_processing import plane_finder
from lib.cfg_importer import cfg



def post_proc(rgb_image,pcl_array,label_pred, camera_obj):
	print 'Starting post processing------------'
	start_time = time.time()
	full_pcd = pcl_array
	table_pcd = find_table_plane(full_pcd)

	print "--- %s seconds ---" % (time.time() - start_time)

	table_top_pcd = pcl_processing.pcl_above_plane(table_pcd, full_pcd)

	print "--- %s seconds ---" % (time.time() - start_time)



	cam_model = pcl_pixel_transform.Transfomer(int(camera_obj.K[0]),int(camera_obj.K[4]),\
												int(camera_obj.K[2]),int(camera_obj.K[5]))

	table_mask = cam_model.pcl_to_2dcoord(table_pcd)

	print "--- %s seconds ---" % (time.time() - start_time)


	if cfg.CLUSTERING_DIM == '2D':
		filtered_mask, mask_idx = clustering_2D.clustering(cam_model, table_mask, table_top_pcd)
	if cfg.CLUSTERING_DIM == '3D':
		filtered_mask, mask_idx = clustering_3D.clustering(cam_model, table_mask, table_top_pcd)

	print "--- %s seconds ---" % (time.time() - start_time)

	# im_label = label_pred[...,[2,1,0]]



	im_label = voting.post_proc_label(filtered_mask,label_pred, mask_idx)

	print "--- %s seconds ---" % (time.time() - start_time)

	print 'Ending post processing----------------------'

	# plt.imshow(im_label)
	# plt.show()
	# print filtered_mask[148,376]
	# print filtered_mask[228,387]
	# print im_label[211,181,:]
	# plt.imshow(filtered_mask)
	# plt.show()
	# im_label = im_label[...,[2,1,0]]

	# lbl_pcd = label_pcd(cam_model, full_pcd, im_label,rgb_image)
	# lbl_pcd = lbl_pcd.flatten()


	return im_label, 0

