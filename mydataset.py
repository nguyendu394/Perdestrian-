from __future__ import print_function, division
import os, sys, cv2
import torch
import pandas as pd
# from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import matplotlib.patches as patches
# from PIL import Image
from model.roi_layers import ROIPool
from image_processing import equalizeHist, showTensor
# sys.path.append('~/pytorch/simple-faster-rcnn-pytorch/')
#
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

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
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs.iloc[idx,0])
        image = cv2.imread(img_name)

        bbs = self.bb.loc[idx].iloc[:20].reset_index().as_matrix()
        bbs = bbs.astype('float')

        tm = cv2.imread(os.path.join(self.ther_path,'set00_V000_lwir_{}'.format(self.imgs.iloc[idx,0])),0)
        sample = {'image': image, 'bb': bbs, 'tm':tm}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbs, tm = sample['image'], sample['bb'], sample['tm']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        tm = equalizeHist(tm)
        tm = np.expand_dims(tm, axis=0)
        return {'image': torch.from_numpy(image).type('torch.FloatTensor'),
                'bb': torch.from_numpy(bbs).type('torch.FloatTensor'),
                'tm': torch.from_numpy(tm).type('torch.FloatTensor')}

if __name__ == '__main__':
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/images/set00/V000/visible'
    IMGS_CSV = 'mydata/imgs_name_set00_v000.csv'
    ROIS_CSV = 'mydata/rois_set00_v000.csv'
    my_transform = ToTensor()
    device = torch.device("cuda:0")
    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = my_transform)

    dataloader = DataLoader(my_dataset, batch_size=5,
    shuffle=True, num_workers=1)

    # data_frame = pd.read_csv('data/1256.csv')
    # img_id = data_frame.iloc[0,0]

    dataiter = iter(dataloader)
    sample = dataiter.next()
    sam = sample['image']
    bbb = sample['bb']
    tm = sample['tm']
    print(sam.shape)
    print(bbb.shape)
    print(tm.shape)
    # roi_pool = ROIPool((50, 50), 1)
    # sam, bbb = sam.to(device),bbb.to(device)
    # # # #
    # out = roi_pool(sam,bbb)
    showTensor(tm)



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
