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


pcd_filepath = '/home/weizhang/Documents/domain-adaptation/examples/0_pcl.pcd'
p = pcl.PointCloud()
p.from_file(pcd_filepath)
full_pcd = p.to_array()

# full_pcd = full_pcd[~np.isnan(full_pcd[:,2]),:]

start_time = time.time()

full_pcd = full_pcd[~np.isnan(full_pcd[:,2]),:]

print "--- %s seconds ---" % (time.time() - start_time)

table_pcd = find_table_plane(full_pcd)

print "--- %s seconds ---" % (time.time() - start_time)