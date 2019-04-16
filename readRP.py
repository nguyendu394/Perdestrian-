import pandas as pd
import os, cv2
import numpy as np
from image_processing import showBbs

THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
IMGS_CSV = 'mydata/imgs_train.csv'
ROIS_CSV = 'mydata/rois_train_thr70.csv'

imgs = pd.read_csv(IMGS_CSV)
bbs = pd.read_csv(ROIS_CSV)
bbs.set_index('id', inplace=True)
bb = bbs.loc[37918].iloc[:].values
img_path = imgs.loc[37918].values
print(print(bb[0]))
img = cv2.imread(os.path.join(ROOT_DIR,img_path[0]))
cv2.imshow('raw',img)
img_bb = showBbs(img,bb)
cv2.imshow('bbs',img_bb)
cv2.waitKey(0)
cv2.destroyAllWindows()




# tworois = []
# aa= np.concatenate((aa,aa),axis=0)
# print(aa)


# min = -1
# idx = -1
# for i in range(50184):
#     if i not in tworois:
#         aa = bbs.loc[i].reset_index().as_matrix()
#         if aa.shape[0] >= min:
#             min = aa.shape[0]
#             idx = i
# print(min,idx)
# print(aa.shape[0])
# aa[:, 0] = aa[:, 0] - aa[0, 0]
