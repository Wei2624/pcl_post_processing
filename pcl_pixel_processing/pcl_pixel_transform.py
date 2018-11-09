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


class Transfomer():
	def __init__(self, path):
		with open(path, 'rb') as input:
			camera_info = pickle.load(input)
		self.cam_model = image_geometry.PinholeCameraModel()
		self.cam_model.fromCameraInfo(camera_info)

	def pcl_to_2dcoord(self, pcd):
		mask = np.zeros((cfg.IMG_HEIGHT,cfg.IMG_WIDTH))
		for i in xrange(pcd.shape[0]):
			coord_2d = self.cam_model.project3dToPixel(pcd[i,:])
			coord_2d = list(coord_2d)
			coord_2d[0] = int(round(coord_2d[0]))
			coord_2d[1] = int(round(coord_2d[1]))
			mask[coord_2d[1],coord_2d[0]] = 1

		return mask

	def pixel_to_pcl(self,uv_pixel):
		unit_pcl = np.zeros((len(uv_pixel),3))

		for key, pixel in enumerate(uv_pixel):
			point = self.cam_model.projectPixelTo3dRay((pixel[0],pixel[1]))
			unit_pcl[key,:] = point

		return unit_pcl