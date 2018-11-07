import os
import cv2
import numpy as np
import sys
import scipy.io
import curvox
import pcl
import image_geometry
import random
import scipy.io



def point_cloud(depth,cx,cy,fx,fy):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    print z.shape
    x = np.where(valid, z * (c - cx) / fx, 0)
    print x.shape
    y = np.where(valid, z * (r - cy) / fy, 0)
    print y.shape


    return np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))


def depth_from_pcl(point_cloud,cx,cy,fx,fy):
	im_depth = np.zeros((480,640))

	count = 0

	for i in xrange(0,point_cloud.shape[0]):
		if math.isnan(point_cloud[i,2]): continue
		z = point_cloud[i,2] * 256.0
		x = int(round(point_cloud[i,0] * fx / point_cloud[i,2] + cx))
		y = int(round(point_cloud[i,1] * fy / point_cloud[i,2] + cy))

		im_depth[y,x] = z

	return im_depth


def pcl_trim(point_cloud):
	# idx = np.where()
    thres = np.nanpercentile(point_cloud[:,2],25)
    print "threshold value is: ",thres
    idx = np.where(point_cloud[:,2] > 1.2)
    point_cloud[idx[0], 2] = np.nan

    return point_cloud
