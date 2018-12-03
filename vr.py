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
import time

from lib.cfg_importer import cfg


# from os.path import dirname, join, abspath
# print abspath(dirname(__file__))
# sys.path.insert(0, abspath(dirname(__file__)))
# print sys.path

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


alpha = 0.6

if __name__ == "__main__":
	p = pcl.PointCloud()
	for i in range(15,16):
		print i
		base_path = '/home/weizhang/DA-RNN/data/LabScene/data/'  + '{:04d}/'.format(i)
		if not os.path.exists(base_path): os.mkdir(base_path)
		for j in xrange(0,50):
			print j

			filename_rgba = os.path.join(base_path,'{:04d}_rgba.png'.format(j))

			im_rgba = cv2.imread(filename_rgba)

			im_rgba = im_rgba[...,[2,1,0]]




			cam_model = pcl_pixel_transform.Transfomer(cfg.fx,cfg.fy,cfg.cx,cfg.cy)

			filename_full_pcd = os.path.join(base_path,'{:04d}_pcl.pcd'.format(j))
			p.from_file(filename_full_pcd)
			full_pcd = p.to_array()

			filename_label = os.path.join(base_path,'{:04d}_label_filter_noMarker.png'.format(j))
			im_label = cv2.imread(filename_label)
			im_label = im_label[...,[2,1,0]]

			# res = alpha*im_rgba + (1-alpha)*im_label

			# res = res.astype(np.uint8)

			# plt.imshow(res)
			# plt.show()


			lbl_pcd = label_pcd(cam_model, full_pcd, im_label,im_rgba)
			lbl_pcd = lbl_pcd.flatten()
			# print lbl_pcd[0:12]
			# sys.exit()
			csv_file_path = os.path.join(base_path,"lbl_pcd_bg_{:04d}.csv".format(j))
			np.savetxt(csv_file_path, lbl_pcd, delimiter=",")


