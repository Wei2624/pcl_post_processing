import os
import cv2
import numpy as np
import sys
import scipy.io
import curvox
import pcl


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

HEIGHT = 480
WIDTH = 640

for i in range(12,13):
	print i
	base_path = '/home/weizhang/DA-RNN/data/LabScene/data/'  + '{:04d}/'.format(i)


	# video_path = os.path.join(base_path,'demo.avi')
	# video = cv2.VideoWriter(video_path, -1, 1, (WIDTH*2,HEIGHT), True)
	# video.open(video_path, -1, 1, (WIDTH*2,HEIGHT), True)
	for j in xrange(0,50):
		print j
		label_path = os.path.join(base_path,'{:04d}_label_filter_noMarker.png'.format(j))
		rgba_path = os.path.join(base_path,'{:04d}_rgba.png'.format(j))

		label_im = cv2.imread(label_path)
		rgba_im = cv2.imread(rgba_path)

		# label_im = label_im[...,[2,1,0]]
		# rgba_im = rgba_im[...,[2,1,0]]


		combined_im = np.zeros((HEIGHT,WIDTH*2,3))

		combined_im[:,0:WIDTH,:] = rgba_im
		combined_im[:,WIDTH:WIDTH*2,:] = label_im
		test_path = os.path.join(base_path,'{:04d}_combined.png'.format(j))
		cv2.imwrite(test_path,combined_im)

		# plt.imshow(combined_im)
		# plt.show()

		# video.write(combined_im)


	# cv2.destroyAllWindows()
	# video.release()



		# im = cv2.imread(filename_label)
		# print im.shape

		# im = im[...,[2,1,0]]

		# cv2.imwrite(filename_label,im)