import pandas as pd
import os, cv2,glob
import numpy as np
from image_processing import showBbs
import vgg
import torch
from config import cfg

def main():
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_test.csv'
    ROIS_CSV = 'mydata/rois_test_thr70.csv'
    ROIS_KAIST_CSV = 'mydata/rois_testKaist_thr70_MSDN.csv'

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
    #
    # min = -1
    # idx = -1
    # for i in range(50184):
    #     aa = bbs.loc[i].reset_index().values
    #     if aa.shape[0] > min:
    #         min = aa.shape[0]
    #         idx = i
    # print(min,idx)

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



def test():
    PATH = cfg.TEST.ROOT_DIR
    data = glob.glob(PATH + '/*.jpg')
    data.sort()
    with open('mydata/imgs_test_caltech.txt','a') as f:
        f.write('img_name\n')
        for i in data:
            print(i)
            f.write(i.split('/')[-1])
            f.write('\n')
    print('done!')
if __name__ == '__main__':
    # test()
    main()
