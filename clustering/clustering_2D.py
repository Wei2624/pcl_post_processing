import os
import cv2
import numpy as np
import sys
import scipy.io
import pcl
import image_geometry
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import pickle

from pcl_pixel_processing import pcl_processing
from pcl_pixel_processing import pcl_pixel_transform
from lib.util import point_inline_check
from lib.util import draw_contours
from lib.cfg_importer import cfg


def ScanLineStack_Exchange(mask, target_idx, seed_coord):
	seed_idx = 0
	mask_cp = np.copy(mask)
	x1 = 0
	spanAbove = 0
	spanBelow = 0
	stack = [seed_coord]
	previous = np.where(mask_cp==1)[0].shape[0]
	while len(stack) > 0:
		coord = stack.pop()
		x = coord[0]
		y = coord[1]
		x1 = x

		while point_inline_check(x1,y) and mask_cp[x1,y] == seed_idx: x1 -= 1
		x1 += 1

		spanAbove = spanBelow = False
		while point_inline_check(x1,y) and mask_cp[x1,y] == seed_idx:
			mask_cp[x1,y] = target_idx
			if not spanAbove and point_inline_check(x1,y) and mask_cp[x1,y-1] == seed_idx:
				# print "append"
				stack.append((x1,y-1))
				spanAbove = True
			elif spanAbove and point_inline_check(x1,y) and mask_cp[x1,y-1] != seed_idx:
				# print "no append"
				spanAbove = False

			if not spanBelow and point_inline_check(x1,y) and mask_cp[x1,y + 1] == seed_idx:
				# print "append"
				stack.append((x1,y+1))
				spanBelow = True
			elif spanBelow and point_inline_check(x1,y) and mask_cp[x1,y+1] != seed_idx:
				# print "no append"
				spanBelow = False
			x1 += 1

	return mask_cp

def search_around_point(idx_list, mask):
	mask_updated = np.copy(mask)
	fill_idx = 3
	for idx in idx_list:
		if mask_updated[idx[0],idx[1]] == 0: 
			mask_updated = ScanLineStack_Exchange(mask_updated,fill_idx,idx)
			# if fill_idx == 7: print idx[0],idx[1]
			fill_idx += 1
	# print np.where(mask_updated==2)[0].shape
			# plt.imshow(mask_updated)
	# print mask_updated[192,128]
	# print mask_updated[179,314]
	# print mask_updated[135,442]
			# plt.show()
	# sys.exit()
	return mask_updated, fill_idx