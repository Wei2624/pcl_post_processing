import os
import cv2
import numpy as np
import sys
import scipy.io
import curvox
import pcl
import image_geometry

import tf



def transform_view(pcd, meta_file):
	meta = scipy.io.loadmat(meta_file)
	translation_rotation = meta['first_marker']
	rot = translation_rotation[0][1][0] 
	trans = translation_rotation[0][0][0]


	R = tf.transformations.quaternion_matrix(list(rot))
	R[0,3] = trans[0]
	R[1,3] = trans[1]
	R[2,3] = trans[2]

	# R[3,0] = trans[0]
	# R[3,1] = trans[1]
	# R[3,2] = trans[2]
	# print list(rot)
	# import ipdb
	# ipdb.set_trace()
	# import IPython
	# IPython.embed()
	# trans = np.expand_dims(trans,axis=1)
	# trans = np.tile(trans,(1,pcd.shape[0]))

	pcd = np.transpose(pcd)

	pcd = np.append(pcd,np.ones((1, pcd.shape[1])),axis=0)


	transformed = np.matmul(R, pcd)
	transformed = np.delete(transformed,-1,0)

	# transformed = transformed + trans

	transformed = np.transpose(transformed)

	return transformed