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
import math
import pickle



from lib.cfg_importer import cfg



def pcl_to_2dcoord(path, pcd):
	with open(path, 'rb') as input:
		camera_info = pickle.load(input)

	img_geo = image_geometry.PinholeCameraModel()
	img_geo.fromCameraInfo(camera_info)
	mask = np.zeros((480,640))
	for i in xrange(pcd.shape[0]):
		coord_2d = img_geo.project3dToPixel(pcd[i,:])
		coord_2d = list(coord_2d)
		coord_2d[0] = int(round(coord_2d[0]))
		coord_2d[1] = int(round(coord_2d[1]))
		mask[coord_2d[1],coord_2d[0]] = 1

	return mask



def pixel_to_pcl(path,uv_pixel):
	with open(path, 'rb') as input:
		camera_info = pickle.load(input)

	img_geo = image_geometry.PinholeCameraModel()
	img_geo.fromCameraInfo(camera_info)

	unit_pcl = np.zeros((len(uv_pixel),3))

	for key, pixel in enumerate(uv_pixel):
		point = img_geo.projectPixelTo3dRay((pixel[0],pixel[1]))
		unit_pcl[key,:] = point

	return unit_pcl