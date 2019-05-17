from __future__ import print_function, division
import os, sys, cv2
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from my_transforms import *
from torchvision import transforms
# import torchvision.transforms.functional as TF
# import matplotlib.patches as patches
# from PIL import Image
# import random
# from model.roi_layers import ROIPool
from image_processing import convertTensor2Img, showBbs,visualizeRP, resizeThermal
import torch.nn as nn
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

class MyDataset(Dataset):
    """docstring for MyDataset."""
    def __init__(self, imgs_csv,rois_csv, ther_path,root_dir, transform=None):
        super(MyDataset, self).__init__()
        self.imgs = pd.read_csv(imgs_csv)
        self.bb = pd.read_csv(rois_csv)
        self.bb.set_index('id', inplace=True)
        self.root_dir = root_dir
        self.transform = transform
        self.ther_path = ther_path
        self.gt = None
        self.NUM_BBS = 128

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs.iloc[idx,0])
        gt_name = img_name.replace('images_train', 'annotations_train')
        gt_name = gt_name.replace('jpg', 'txt')

        image = io.imread(img_name)

        bbs = self.bb.loc[idx].iloc[:self.NUM_BBS].reset_index().as_matrix()
        bbs = bbs.astype('float')

        tm = io.imread(os.path.join(self.ther_path,self.imgs.iloc[idx,0].replace('visible','lwir')))
        sample = {'image': image, 'bb': bbs, 'tm':tm, 'gt':gt_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


def testResizeThermal(sample,NUM_BBS):
    sam = sample['image']
    bbb = sample['bb']
    tm = sample['tm']
    gt = sample['gt']
    bbb=bbb.view(-1, 5)
    idx = -1
    for j,v in enumerate(bbb[:,0]):
        if not j%NUM_BBS:
            idx = idx + 1
        bbb[j,0] = idx

    labels_output = resizeThermal(tm, bbb)
    # print(labels_output.size())
    # labels_output = labels_output.type('torch.ByteTensor')
    out = labels_output.cpu()
    out = out.detach().numpy()
    ther = convertTensor2Img(tm,0)
    imgg = visualizeRP(ther, bbb)

    cv2.imshow('winname', imgg)
    for ind,labels in enumerate(out):
        cv2.imshow('bbs{}'.format(ind), labels.transpose((1, 2, 0)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testDataset(sample):
    sam = sample['image']
    bbb = sample['bb']
    tm = sample['tm']
    gt = sample['gt']
    print(bbb.size())

    # print(sam.shape)
    # print(bbb.shape)
    # print(tm.shape)
    # print(len(gt))


    # for ind,labels in enumerate(labels_output):
    #     print(labels.shape)
    #     cv2.imshow('bbs{}'.format(ind), labels)
    raw = convertTensor2Img(sam,1)
    ther = convertTensor2Img(tm,0)
    bbs = visualizeRP(raw, bbb)

    cv2.imshow('raw',raw)
    cv2.imshow('thermal', ther)
    cv2.imshow('bbs', bbs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    # roi_pool = ROIPool((50, 50), 1)
    # tm, bbb = tm.to(device),bbb.to(device)
    # labels_output = roi_pool(tm,bbb)
    # print(labels_output.shape)

if __name__ == '__main__':
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_train.csv'
    ROIS_CSV = 'mydata/rois_trainKaist_thr70_0.csv'
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),])
                                  # Normalize(rgb_mean,rgb_std)])

    device = torch.device("cuda:0")
    params = {'batch_size':1,
              'shuffle':True,
              'num_workers':24}


    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    dataiter = iter(dataloader)
    sample = dataiter.next()
    # sample = my_dataset[789]
    NUM_BBS = my_dataset.NUM_BBS

    testDataset(sample)

    # bbb = bbb.view(-1,5)
    # idx = -1
    # for i,v in enumerate(bbb[:,0]):
    #     if not i%NUM_BBS:
    #         idx = idx + 1
    #     bbb[i,0] = idx
    # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]

    # ther = convertTensor2Img(tm,0)
    # labels_output = resizeThermal(tm, bbb)
    # print(labels_output.shape)
        # for ind,labels in enumerate(labels_output):
        #     print(labels.shape)
        #     cv2.imshow('bbs{}'.format(ind), labels)
    # for i,p in enumerate(labels_output):
    #     print(p.shape)
    #     cv2.imshow('bba{}'.format(i), p)
    # tmm = cv2.resize(labels_output, (50,50))
    # cv2.imshow('bbbb', tmm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # exit()
    # bbb=bbb.view(-1, 5)
    # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]
    # #reset id
    # tmp = set()

    # idx = -1
    # print(bbb)
    # # # #
    # criterion = nn.MSELoss()
    #
    # out = roi_pool(tm,bbb)
    # loss = criterion(out, out)
    # print(loss.item())
    # showTensor(out)



    # image = io.imread(img_id)
    # a = torch.from_numpy(image)
    # print(a.view(-1,3).shape)

    # bb = data_frame.iloc[0, 1:-1].as_matrix()
    # bb = bb.astype('float')
    # print(bb)
# print('load batch size')

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #           sample_batched['bb'])
    #
    #     roi_pool = ROIPool((7, 7), 1.0)
    #     pooled_output = roi_pool(sample_batched['image'])
    #     input()
    #     exit()


# print(len(my_dataset))
# for i in range(len(my_dataset)):
#     print(i,my_dataset[i]['image'].shape)
# sample = my_dataset[0]

# fig = plt.figure()

# for i in range(len(my_dataset)):
#     sample = my_dataset[i]
#
#     print(i, sample['image'].shape, sample['bb'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_img_bb(**sample)
#
#     if i == 3:
#         plt.show()
#         break
# plt.waitforbuttonpress()
