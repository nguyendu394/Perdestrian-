import pandas as pd
import os, cv2
import numpy as np
from image_processing import showBbs
import vgg
import torch

def main():
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_test.csv'
    ROIS_CSV = 'mydata/rois_test_thr70.csv'
    ROIS_KAIST_CSV = 'mydata/rois_testKaist_thr110_MSDN.csv'

    imgs = pd.read_csv(IMGS_CSV)
    bbs = pd.read_csv(ROIS_KAIST_CSV)
    bbs.set_index('id', inplace=True)
    print(bbs)
    # for idx in range(45139):
    #     print('IDX: ', idx)
    #     bb = bbs.loc[idx].iloc[:15].reset_index().as_matrix()
    #     try:
    #         bb = bb.astype('float')
    #     except:
    #         ans.append(idx)
    # print(ans)
    # bb = bbs.loc[37874].iloc[:15].reset_index().values
    # print(bb)
    # bb = bbs.loc[37875].iloc[:].values

    # img_path = imgs.loc[37918].values
    # #
    # min = -1
    # idx = -1
    # for i in range(50184):
    #     aa = bbs.loc[i].reset_index().values
    #     if aa.shape[0] > min:
    #         min = aa.shape[0]
    #         idx = i
    # print(min,idx)
    # print(aa.shape[0])
    # aa[:, 0] = aa[:, 0] - aa[0, 0]
def splitTestPredicted(path_txt):
    pd.read_csv(imgs_csv)

if __name__ == '__main__':
    # main()
    splitTest('mydata/imgs_test.csv')
