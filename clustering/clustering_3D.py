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



def clean_tabletop_pcd(path, table_mask,tabletop_pcd):
	tabletop_pcd_clean = np.zeros(tabletop_pcd.shape)
	with open(path, 'rb') as input:
		camera_info = pickle.load(input)

	img_geo = image_geometry.PinholeCameraModel()
	img_geo.fromCameraInfo(camera_info)
	index = 0
	for i in xrange(tabletop_pcd.shape[0]):
		coord_2d = img_geo.project3dToPixel(tabletop_pcd[i,:])
		coord_2d = list(coord_2d)
		coord_2d[0] = int(round(coord_2d[0]))
		coord_2d[1] = int(round(coord_2d[1]))
		if table_mask[coord_2d[1],coord_2d[0]] == 0:
			tabletop_pcd_clean[index,:] = tabletop_pcd[i,:]
			index += 1

	return tabletop_pcd_clean[:index,:]

