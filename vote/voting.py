import os
import cv2
import numpy as np
import sys
import scipy.io
import pcl
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import pickle

from lib.cfg_importer import cfg


def post_proc_label(full_mask, im_label, total_index):
	proc_label = np.zeros((cfg.IMG_HEIGHT,cfg.IMG_WIDTH,3))
	idx = np.where(full_mask == 1)
	proc_label[idx[0],idx[1],:] = (255,0,0)
	# im_label[idx[0],idx[1],:] = (255,0,0)

	# return im_label

	for idx in xrange(3,total_index):
		pos = np.where(full_mask == idx)
		print pos[0].shape
		votes = np.zeros((len(cfg.LABEL_COLOR)))
		for i in xrange(pos[0].shape[0]):
			if point_inline_check(pos[0][i]+40,pos[1][i]-10):
				if not tuple(im_label[pos[0][i]+40,pos[1][i]-10,:]) in outlier_color: 
					vote = label_color.index(tuple(im_label[pos[0][i]+40,pos[1][i]-10,:]))
					votes[vote] += 1
		print votes
		# mask = np.zeros((HEIGHT,WIDTH,3))
		# mask[pos[0],pos[1],:] =(1,1,1) 
		# plt.imshow(np.multiply(im_label,mask))
		# plt.show()
		# sys.exit()
		print cfg.LABEL_COLOR[np.argmax(votes)]
		proc_label[pos[0],pos[1],:] = cfg.LABEL_COLOR[np.argmax(votes)]
		# if idx == 6:
		# 	proc_label[pos[0],pos[1],:] = label_color[0]
		# if idx == 4:
		# 	mask = np.zeros((HEIGHT,WIDTH,3))
		# 	mask[pos[0],pos[1],:] =(1,1,1) 
		# 	# print np.multiply(proc_label,mask)[136,424,:]
		# 	plt.imshow(np.multiply(im_label,mask))
		# 	plt.show()
		# 	sys.exit()

	return proc_label