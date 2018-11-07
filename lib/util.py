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

from cfg_importer import cfg

def point_inline_check(x,y):
	return (x>0 and x<cfg.IMG_HEIGHT-1 and y>0 and y<cfg.IMG_WIDTH-1)


def draw_contours(img, cont):
	# import ipdb
	# ipdb.set_trace()
	img_c = img.copy()
	for c in cont:
		for i in range(c.shape[0]):
			# print c[i,0,1],c[i,0,0]
			img_c[c[i,0,1],c[i,0,0]] = 2

			print img_c[c[i,0,1],c[i,0,0]] == 2

	# print img_c.shape
	# print np.where(img_c==2)

	return img_c

def check_ele_list(elem,lst):
	try:
		return lst.index(elem)
	except ValueError:
		return -1