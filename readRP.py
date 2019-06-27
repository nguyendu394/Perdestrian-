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
    ROIS_KAIST_CSV = 'mydata/rois_trainCaltech_thr70_MSDN.csv'

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

def testConcat():
    a = np.array([[1,1,1,1,1],
                  [2,2,2,2,2],
                  [3,3,3,3,3],
                  [4,4,4,4,4]])
    b = np.array([[11,11,11,11,11],
                  [22,22,22,22,22],
                  [33,33,33,33,33],
                  [44,44,44,44,44]])
    c = np.array([[111,111,111,111,111],
                  [222,222,222,222,222],
                  [333,333,333,333,333],
                  [444,444,444,444,444]])
    print(np.concatenate((a,b,c),axis=1))

def test(aa,bb,cc):
    print(aa)
    return aa + bb + cc
if __name__ == '__main__':
    main()
    # img = cv2.imread('mydata/I00000.png')
    # a = torch.from_numpy(img)
    # a = torch.flip(a,(1,2))
    # cv2.imshow('winname', img[:,::-1,:])
    # cv2.imshow('wiame', a.numpy().astype(np.uint8))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
