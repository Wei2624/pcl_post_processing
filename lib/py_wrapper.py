import sys
import os
import ctypes
import pcl
import numpy.ctypeslib as npct
import numpy as np


def find_table_plane(pcd):
	lib = npct.load_library("table_finder/testlib", ".")
	array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
	out = np.zeros((500000))
	pcd_arr = np.ones((pcd.shape[0]*3))
	pcd_arr[0:pcd.shape[0]] = pcd[:,0]
	pcd_arr[pcd.shape[0]:pcd.shape[0]*2] = pcd[:,1]
	pcd_arr[2*pcd.shape[0]:3*pcd.shape[0]] = pcd[:,2]



	lib.add_one.restype = ctypes.c_int
	lib.add_one.argtypes = [array_1d_double,ctypes.c_int,ctypes.c_int, array_1d_double]
	num_idx = lib.add_one(pcd_arr,pcd.shape[0],3, out)

	res_out = np.zeros((num_idx,3))
	res_out[:,0] = out[0:num_idx]
	res_out[:,1] = out[num_idx:2*num_idx]
	res_out[:,2] = out[2*num_idx:3*num_idx]

	return res_out