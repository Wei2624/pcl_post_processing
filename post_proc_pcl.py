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
from vote import voting 
from lib.cfg_importer import cfg


WIDTH = 640
HEIGHT = 480

label_color = [(0,255,0), (0,0,255), (0,255,255), (255,255,0),(255,0,255),(0,0,188)]
outlier_color = [(0,0,0), (255,0,0)]


# def pcl_to_2dcoord(path, pcd):
# 	with open(path, 'rb') as input:
# 		camera_info = pickle.load(input)

# 	img_geo = image_geometry.PinholeCameraModel()
# 	img_geo.fromCameraInfo(camera_info)
# 	mask = np.zeros((480,640))
# 	for i in xrange(pcd.shape[0]):
# 		coord_2d = img_geo.project3dToPixel(pcd[i,:])
# 		coord_2d = list(coord_2d)
# 		coord_2d[0] = int(round(coord_2d[0]))
# 		coord_2d[1] = int(round(coord_2d[1]))
# 		mask[coord_2d[1],coord_2d[0]] = 1

# 	return mask

# def pixel_to_pcl(path,uv_pixel):
# 	with open(path, 'rb') as input:
# 		camera_info = pickle.load(input)

# 	img_geo = image_geometry.PinholeCameraModel()
# 	img_geo.fromCameraInfo(camera_info)

# 	unit_pcl = np.zeros((len(uv_pixel),3))

# 	for key, pixel in enumerate(uv_pixel):
# 		point = img_geo.projectPixelTo3dRay((pixel[0],pixel[1]))
# 		unit_pcl[key,:] = point

# 	return unit_pcl

# def table_plane_pcl(path):
# 	table_pcd = []
# 	min_distance = 10000000000000
# 	p = pcl.PointCloud()
# 	for f in os.listdir(path):
# 		p.from_file(os.path.join(path, f))
# 		pcd = p.to_array()
# 		dist = np.min(np.square(pcd[:,0]) + np.square(pcd[:,1]) + np.square(pcd[:,2]))
# 		if dist < min_distance:
# 			min_distance = dist
# 			table_pcd = pcd

# 	return table_pcd

# def point_inline_check(x,y):
# 	return (x>0 and x<HEIGHT-1 and y>0 and y<WIDTH-1)

# def ScanLineStack_Exchange(mask, target_idx, seed_coord):
# 	seed_idx = 0
# 	mask_cp = np.copy(mask)
# 	x1 = 0
# 	spanAbove = 0
# 	spanBelow = 0
# 	stack = [seed_coord]
# 	previous = np.where(mask_cp==1)[0].shape[0]
# 	while len(stack) > 0:
# 		coord = stack.pop()
# 		x = coord[0]
# 		y = coord[1]
# 		x1 = x

# 		while point_inline_check(x1,y) and mask_cp[x1,y] == seed_idx: x1 -= 1
# 		x1 += 1

# 		spanAbove = spanBelow = False
# 		while point_inline_check(x1,y) and mask_cp[x1,y] == seed_idx:
# 			mask_cp[x1,y] = target_idx
# 			if not spanAbove and point_inline_check(x1,y) and mask_cp[x1,y-1] == seed_idx:
# 				# print "append"
# 				stack.append((x1,y-1))
# 				spanAbove = True
# 			elif spanAbove and point_inline_check(x1,y) and mask_cp[x1,y-1] != seed_idx:
# 				# print "no append"
# 				spanAbove = False

# 			if not spanBelow and point_inline_check(x1,y) and mask_cp[x1,y + 1] == seed_idx:
# 				# print "append"
# 				stack.append((x1,y+1))
# 				spanBelow = True
# 			elif spanBelow and point_inline_check(x1,y) and mask_cp[x1,y+1] != seed_idx:
# 				# print "no append"
# 				spanBelow = False
# 			x1 += 1

# 	return mask_cp

# def search_around_point(idx_list, mask):
# 	mask_updated = np.copy(mask)
# 	fill_idx = 3
# 	for idx in idx_list:
# 		if mask_updated[idx[0],idx[1]] == 0: 
# 			mask_updated = ScanLineStack_Exchange(mask_updated,fill_idx,idx)
# 			# if fill_idx == 7: print idx[0],idx[1]
# 			fill_idx += 1
# 	# print np.where(mask_updated==2)[0].shape
# 			# plt.imshow(mask_updated)
# 	# print mask_updated[192,128]
# 	# print mask_updated[179,314]
# 	# print mask_updated[135,442]
# 			# plt.show()
# 	# sys.exit()
# 	return mask_updated, fill_idx

# def post_proc_label(full_mask, im_label, total_index):
# 	proc_label = np.zeros((HEIGHT,WIDTH,3))
# 	idx = np.where(full_mask == 1)
# 	proc_label[idx[0],idx[1],:] = (255,0,0)
# 	# im_label[idx[0],idx[1],:] = (255,0,0)

# 	# return im_label

# 	for idx in xrange(3,total_index):
# 		pos = np.where(full_mask == idx)
# 		print pos[0].shape
# 		votes = np.zeros((len(label_color)))
# 		for i in xrange(pos[0].shape[0]):
# 			if point_inline_check(pos[0][i]+40,pos[1][i]-10):
# 				if not tuple(im_label[pos[0][i]+40,pos[1][i]-10,:]) in outlier_color: 
# 					vote = label_color.index(tuple(im_label[pos[0][i]+40,pos[1][i]-10,:]))
# 					votes[vote] += 1
# 		print votes
# 		# mask = np.zeros((HEIGHT,WIDTH,3))
# 		# mask[pos[0],pos[1],:] =(1,1,1) 
# 		# plt.imshow(np.multiply(im_label,mask))
# 		# plt.show()
# 		# sys.exit()
# 		print label_color[np.argmax(votes)]
# 		proc_label[pos[0],pos[1],:] = label_color[np.argmax(votes)]
# 		# if idx == 6:
# 		# 	proc_label[pos[0],pos[1],:] = label_color[0]
# 		# if idx == 4:
# 		# 	mask = np.zeros((HEIGHT,WIDTH,3))
# 		# 	mask[pos[0],pos[1],:] =(1,1,1) 
# 		# 	# print np.multiply(proc_label,mask)[136,424,:]
# 		# 	plt.imshow(np.multiply(im_label,mask))
# 		# 	plt.show()
# 		# 	sys.exit()

# 	return proc_label

# def find_normal_vector(pcd):
# 	rows = pcd.shape[0]
# 	normal_vector = 0
# 	max_itr = 20
# 	for itr in xrange(0,max_itr):
# 		idx = random.sample(range(0,rows),3)
# 		point1 = pcd[idx[0],:]
# 		point2 = pcd[idx[1],:]
# 		point3 = pcd[idx[2],:]

# 		vector1 = point2 - point1
# 		vector2 = point3 - point1


# 		test_vector = -point1


# 		normal_vector += np.cross(vector1,vector2)

# 	if np.dot(normal_vector,test_vector) < 0 : return -normal_vector/max_itr
# 	else: return normal_vector/max_itr

# # def check_ele_list(elem,lst):
# # 	try:
# # 		return lst.index(elem)
# # 	except ValueError:
# # 		return -1



# def pcl_above_plane(plane_pcl,all_pcl):
# 	anchor_point = np.mean(plane_pcl,axis=0)

# 	roi_pcl = np.zeros(all_pcl.shape)
# 	index = 0

# 	normal_vector = find_normal_vector(plane_pcl)

# 	for i in xrange(all_pcl.shape[0]):
# 		if np.dot(normal_vector,all_pcl[i,:] - anchor_point) > 0 :
# 			roi_pcl[index,:] = all_pcl[i,:]
# 			index += 1
# 	return roi_pcl[:index,:]


# def draw_contours(img, cont):
# 	# import ipdb
# 	# ipdb.set_trace()
# 	img_c = img.copy()
# 	for c in cont:
# 		for i in range(c.shape[0]):
# 			# print c[i,0,1],c[i,0,0]
# 			img_c[c[i,0,1],c[i,0,0]] = 2

# 			print img_c[c[i,0,1],c[i,0,0]] == 2

# 	# print img_c.shape
# 	# print np.where(img_c==2)

# 	return img_c


# def clean_tabletop_pcd(path, table_mask,tabletop_pcd):
# 	tabletop_pcd_clean = np.zeros(tabletop_pcd.shape)
# 	with open(path, 'rb') as input:
# 		camera_info = pickle.load(input)

# 	img_geo = image_geometry.PinholeCameraModel()
# 	img_geo.fromCameraInfo(camera_info)
# 	index = 0
# 	for i in xrange(tabletop_pcd.shape[0]):
# 		coord_2d = img_geo.project3dToPixel(tabletop_pcd[i,:])
# 		coord_2d = list(coord_2d)
# 		coord_2d[0] = int(round(coord_2d[0]))
# 		coord_2d[1] = int(round(coord_2d[1]))
# 		if table_mask[coord_2d[1],coord_2d[0]] == 0:
# 			tabletop_pcd_clean[index,:] = tabletop_pcd[i,:]
# 			index += 1

# 	return tabletop_pcd_clean[:index,:]


if __name__ == "__main__":
	for i in range(16,17):
		print i
		base_path = '/home/weizhang/DA-RNN/data/LabScene/data/'  + '{:04d}/'.format(i)
		if not os.path.exists(base_path): os.mkdir(base_path)
		for j in xrange(29,30):
			print j			
			path_plane_pcd = os.path.join(base_path,'{:04d}/'.format(j))
			table_pcd = pcl_processing.table_plane_pcl(path_plane_pcd)


			filename_pcd = os.path.join(base_path,'{:04d}_pcl.pcd'.format(j))
			p = pcl.PointCloud()
			p.from_file(filename_pcd)
			full_pcd = p.to_array()

			table_top_pcd = pcl_processing.pcl_above_plane(table_pcd, full_pcd)

			# table_top_pcd = clustering_3D.clean_tabletop_pcd(table_pcd,table_top_pcd)


			filename_camera = os.path.join(base_path,'{:04d}_pkl.pkl'.format(j))
			# full_mask = pcl_pixel_transform.pcl_to_2dcoord(filename_camera,full_pcd)
			table_top_mask = pcl_pixel_transform.pcl_to_2dcoord(filename_camera,table_top_pcd)

			# full_mask = pcl_to_2dcoord(filename_camera,full_pcd)
			table_mask = pcl_pixel_transform.pcl_to_2dcoord(filename_camera,table_pcd)
			# print type(table_mask[0,0])

			table_top_pcd_clean = clustering_3D.clean_tabletop_pcd(filename_camera,table_mask,table_top_pcd)
			db = DBSCAN(eps=0.1, min_samples=1000).fit(table_top_pcd_clean)

			cluster_labels = db.labels_

			obj_3d = []
			for i in range(np.unique(cluster_labels).shape[0]-1):
				idx = np.where(cluster_labels==i)
				obj_i = table_top_pcd_clean[idx,:]
				obj_i = np.squeeze(obj_i,axis=0)
				obj_3d.append(obj_i)

			# import ipdb
			# ipdb.set_trace()

			norm_sum_ls=[]
			for i in range(len(obj_3d)):
				# norm_sum = 0
				# for j in range(obj_3d[i].shape[0]):
				# 	min_norm = 100000
				# 	anchor = obj_3d[i][j,:]
				# 	for k in range(j+1,obj_3d[i].shape[0]):
				# 		dis = np.linalg.norm(anchor-obj_3d[i][k,:])
				# 		if dis <min_norm: min_norm = dis
				# 	print min_norm
				# 	norm_sum += min_norm
				# norm_sum_ls.append(norm_sum)
				# print float(norm_sum/obj_3d[i].shape[0])
				pcl_processing.save_pcl(obj_3d[i],'test_{:d}.pcd'.format(i))



			import ipdb
			ipdb.set_trace()

			table_full_mask = np.logical_or(table_top_mask,table_mask)

			# plt.imshow(table_mask)
			# plt.show()
			pcl_processing.save_pcl(table_top_pcd_clean,'test.pcd')
			sys.exit()


			_, contours, _ = cv2.findContours(table_full_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# print type(contours)

			# table_full_mask = np.expand_dims(table_full_mask,axis=2)
			# table_full_mask = np.tile(table_full_mask,(1,1,3))
			# print contours
			# cv2.drawContours(table_full_mask.astype(np.uint8), contours, -1, (255,0,0), 3)


			# kernel = np.ones((3,3),np.uint8)
			# table_mask_full = cv2.erode(table_mask_full,kernel,iterations=1)


			# plt.imshow(table_mask_full)
			# plt.show()


			# kernel = np.ones((3,3),np.uint8)
			# table_mask = cv2.erode(table_mask,kernel,iterations=1)

			# table_mask = np.expand_dims(table_mask,axis=2)
			# table_mask = np.tile(table_mask,(1,1,3))
			# ret, thresh = cv2.threshold(table_mask, 127, 255, 0)
			# print thresh
			# table_mask, contours, hierarchy = cv2.findContours(table_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# table_mask = np.expand_dims(table_mask,axis=2)
			# table_mask = np.tile(table_mask,(1,1,3))
			cv2.drawContours(table_mask, contours,-1, (2), 3)

			# plt.imshow(table_mask)
			# plt.show()
			# print 1 in table_mask[0:164,589] and 1 in table_mask[164:HEIGHT,589] and 1 in table_mask[164,0:589] and 1 in table_mask[164,589:WIDTH]
			# sys.exit()

			idx_0_table_mask = np.where(table_mask == 0)
			idx_0_table_mask_ls = []
			for i in xrange(idx_0_table_mask[0].shape[0]):
				x = idx_0_table_mask[0][i]
				y = idx_0_table_mask[1][i]
				if (1 in table_mask[0:x,y] or 2 in table_mask[0:x,y]) and \
					(1 in table_mask[x:HEIGHT,y] or 2 in table_mask[x:HEIGHT,y]) \
					and (1 in table_mask[x,0:y] or 1 in table_mask[x,0:y])\
					 and (1 in table_mask[x,y:WIDTH] or 2 in table_mask[x,y:WIDTH]): 


					if not np.any(table_mask[x:x+5,y]) and not np.any(table_mask[x:x-5,y]) \
						and not np.any(table_mask[x,y:y+5]) and not np.any(table_mask[x,y-5]):
						if table_full_mask[x,y] == 1:
							idx_0_table_mask_ls.append((x,y))

			

			# points = pcl_pixel_transform.pixel_to_pcl(filename_camera,idx_0_table_mask_ls)
			# pcl_processing.save_pcl(points,'test.pcd')
			# print points.shape
			filtered_mask, total_index = clustering_2D.search_around_point(idx_0_table_mask_ls, table_mask)

			# print filtered_mask[10,10]

			# kernel = np.ones((3,3),np.uint8)
			# filtered_mask = cv2.erode(filtered_mask,kernel,iterations=1)

			# print np.where(filtered_mask==2)[0].shape
			# sys.exit()

			filename_label = os.path.join(base_path,'{:04d}_label.png'.format(j))
			im_label = cv2.imread(filename_label)
			im_label = im_label[...,[2,1,0]]

			im_label = voting.post_proc_label(filtered_mask,im_label, total_index)

			im_label = im_label[...,[2,1,0]]

			plt.imshow(im_label)
			plt.show()

			filename_label = os.path.join(base_path,'{:04d}_label_filter_noMarker.png'.format(j))
			cv2.imwrite(filename_label, im_label)

