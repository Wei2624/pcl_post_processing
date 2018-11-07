import os
import os.path as osp
import numpy as np
import math
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.IMG_WIDTH = 640
__C.IMG_HEIGHT = 480
__C.LABEL_COLOR = [(0,255,0), (0,0,255), (0,255,255), (255,255,0),(255,0,255),(0,0,188)]


def _merge_a_into_b(a, b):
	"""Merge config dictionary a into config dictionary b, clobbering the
	options in b whenever they are also specified in a.
	"""
	if type(a) is not edict:
		return

	for k, v in a.iteritems():
	# a must specify keys that are in b
		if not b.has_key(k):
			raise KeyError('{} is not a valid config key'.format(k))

		# the types must match, too
		if type(b[k]) is not type(v):
			raise ValueError(('Type mismatch ({} vs. {}) '
								'for config key: {}').format(type(b[k]),type(v), k))

		# recursively merge dicts
		if type(v) is edict:
			try:
				_merge_a_into_b(a[k], b[k])
			except:
				print('Error under config key: {}'.format(k))
				raise
		else:
			b[k] = v


def cfg_from_file(filename):
	"""Load a config file and merge it into the default options."""
	import yaml
	with open(filename, 'r') as f:
		yaml_cfg = edict(yaml.load(f))

	_merge_a_into_b(yaml_cfg, __C)



# cfg_from_file('/home/weizhang/Documents/pcl_post_processing/cfgs/cfg_2d_clustering.yaml')
# print cfg