from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#normalize mean std (ImageNet)
__C.rgb_mean = (0.4914, 0.4822, 0.4465)
__C.rgb_std = (0.2023, 0.1994, 0.2010)

#Number of classes (include background)
__C.NUM_CLASSES = 2
__C.IMAGE_WIDTH = 640
__C.IMAGE_HEIGHT = 512

#
# Training options
#
__C.TRAIN = edict()

#thermal image train
__C.TRAIN.THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'

#RGB image train
__C.TRAIN.ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'

#list train image name CSV
__C.TRAIN.IMGS_CSV = 'mydata/imgs_train.csv'

#list rois data train on KAIST with ACF thres -70 csv
__C.TRAIN.ROIS_CSV = 'mydata/rois_trainKaist_thr70_MSDN.csv'

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 1e-4

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
# __C.TRAIN.WEIGHT_DECAY = 0.0005

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 1

#shuffle dataset
__C.TRAIN.SHUFFLE = True

#number CPU used
__C.TRAIN.NUM_WORKERS = 24

#maximum epoch
__C.TRAIN.MAX_EPOCH = 5

#maximum rois in ground-truth
__C.TRAIN.MAX_GTS = 9

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.ROI_PER_IMAGE = 128

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test_tm/'
__C.TEST.ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test'
__C.TEST.IMGS_CSV = 'mydata/imgs_test.csv'
__C.TEST.ROIS_CSV = 'mydata/rois_trainKaist_thr70_MSDN.csv'

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

#maximum rois in ground-truth test
__C.TEST.MAX_GTS = 14

#thres
__C.TEST.THRESS = 0.05

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3
