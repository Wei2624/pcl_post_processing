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

def validate_bg_points(idx_0_table_mask,table_mask,table_full_mask):
	idx_0_table_mask_ls = []
	for i in xrange(idx_0_table_mask[0].shape[0]):
		x = idx_0_table_mask[0][i]
		y = idx_0_table_mask[1][i]
		if (1 in table_mask[0:x,y] or 2 in table_mask[0:x,y]) and \
			(1 in table_mask[x:cfg.IMG_HEIGHT,y] or 2 in table_mask[x:cfg.IMG_HEIGHT,y]) \
			and (1 in table_mask[x,0:y] or 1 in table_mask[x,0:y])\
			 and (1 in table_mask[x,y:cfg.IMG_WIDTH] or 2 in table_mask[x,y:cfg.IMG_WIDTH]): 


			if not np.any(table_mask[x:x+5,y]) and not np.any(table_mask[x:x-5,y]) \
				and not np.any(table_mask[x,y:y+5]) and not np.any(table_mask[x,y-5]):
				if table_full_mask[x,y] == 1:
					idx_0_table_mask_ls.append((x,y))


	return idx_0_table_mask_ls



def clustering(cam_model,table_mask, table_top_pcd):
	table_top_mask = cam_model.pcl_to_2dcoord(table_top_pcd)
	table_full_mask = np.logical_or(table_top_mask,table_mask)
	_, contours, _ = cv2.findContours(table_full_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(table_mask, contours,-1, (2), 3)

	idx_0_table_mask = np.where(table_mask == 0)
	idx_0_table_mask_ls = validate_bg_points(idx_0_table_mask,table_mask,table_full_mask)

	return search_around_point(idx_0_table_mask_ls, table_mask)