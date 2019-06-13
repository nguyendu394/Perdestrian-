import torch, sys
import torchvision
import torchvision.transforms as transforms
from my_transforms import *
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from model.roi_layers import ROIPool
import torchvision.ops.roi_pool as ROIPool
# from skimage import io

import vgg, cv2, time
from mydataset import MyDataset
from image_processing import resizeThermal
from config import cfg

# from image_processing import showTensor



raw_vgg16 = vgg.vgg16(pretrained=True)

class MyRRN(nn.Module):
    def __init__(self):
        super(MyRRN, self).__init__()
        self.features = raw_vgg16.features
        self.deconv1 = nn.ConvTranspose2d(in_channels=512,out_channels=64,kernel_size=4,stride=8, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x, rois):
        x = self.features(x)
        x = ROIPool(x, rois,(7, 7), 1/16)
        x = F.relu(self.deconv1(x))
        x = self.conv1(x)
        return x

def train():
    params = {'batch_size': cfg.TRAIN.BATCH_SIZE,
          'shuffle': cfg.TRAIN.SHUFFLE,
          'num_workers': cfg.TRAIN.NUM_WORKERS}
    print(params)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    print('max_epoch',max_epoch)
    LR = cfg.TRAIN.LEARNING_RATE #learning rate
    print('learning_rate',LR)
    MT = cfg.TRAIN.MOMENTUM

    # cudnn.benchmark = True
    # transform = ToTensor()
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),
                                       Normalize(cfg.BGR_MEAN,cfg.BGR_STD)])

    my_dataset = MyDataset(imgs_csv=cfg.TRAIN.IMGS_CSV,rois_csv=cfg.TRAIN.ROIS_CSV,
    root_dir=cfg.TRAIN.ROOT_DIR, ther_path=cfg.TRAIN.THERMAL_PATH,transform = full_transform)

    dataloader = DataLoader(my_dataset, **params)

    RRN_net = MyRRN()
    RRN_net.to(cfg.DEVICE)

    RRN_net.load_state_dict(torch.load('models/RRN/model24/model24_lr_1e-6_bz_6_NBS_128_norm_epoch_9.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,RRN_net.parameters()), lr=LR, momentum=MT)

    f = open('models/RRN/model27/log27.txt','a')
    for epoch in range(max_epoch):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
        running_loss = 0.0
        st = time.time()
        for i, data in enumerate(dataloader):
            # Lấy dữ liệu
            sample = data
            sam = sample['image']
            bbb = sample['bb']
            num = bbb.size(1)

            bbb=bbb.view(-1, 5)

            #reset id
            # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]
            ind = torch.arange(params['batch_size'],requires_grad=False).view(-1,1)
            ind = ind.repeat(1,num).view(-1,1)
            bbb[:,0] = ind[:,0]

            tm = sample['tm']

            sam,bbb,tm = sam.to(cfg.DEVICE), bbb.to(cfg.DEVICE), tm.to(cfg.DEVICE)

            # labels_output = roi_pool(tm,bbb)
            labels_output = resizeThermal(tm, bbb)
            labels_output = labels_output.to(cfg.DEVICE)

            # Xoá giá trị đạo hàm
            optimizer.zero_grad()

            # Tính giá trị tiên đoán, đạo hàm, và dùng bộ tối ưu hoá để cập nhật trọng số.
            out_RRN = RRN_net(sam,bbb)

            loss = criterion(out_RRN, labels_output)
            # print('loss at {}: '.format(i),loss.item())
            loss.backward()
            optimizer.step()

            # In ra số liệu trong quá trình huấn luyện
            running_loss += loss.item()
            if i % 10 == 9:    # In mỗi 2000 mini-batches.
                text = '[{}, {}] loss: {:.3f}  time: {:.3f}'.format(epoch + 1, i + 1, running_loss / 10,time.time()-st)
                print(text)

                f.write(text + '\n')
                running_loss = 0.0
                st = time.time()
        torch.save(RRN_net.state_dict(), 'models/RRN/model27/model27_lr_1e-6_bz_6_NBS_128_norm_epoch_{}.pth'.format(epoch))
    f.close()
    print('Huấn luyện xong')



def test():
    print('TESTING RRN ...')
    TEST_THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test_tm/'
    TEST_ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test'
    IMGS_CSV = 'mydata/imgs_test.csv'
    ROIS_CSV = 'mydata/rois_testKaist_thr70_0.csv'

    params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 24}

    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),])
                                  # Normalize(rgb_mean,rgb_std)])

    my_dataset = MyDataset(imgs_csv=cfg.TRAIN.IMGS_CSV,rois_csv=cfg.TRAIN.ROIS_CSV,
    root_dir=cfg.TRAIN.TEST_ROOT_DIR, ther_path=cfg.TRAIN.TEST_THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)

    RRN_net = MyRRN()
    RRN_net.to(cfg.DEVICE)
    RRN_net.load_state_dict(torch.load('models/model21/model21_lr_1e-7_bz_6_NBS_128_data_True_epoch_7.ptx'))

    st = time.time()
    running_loss = 0.0
    total_loss = []
    criterion = nn.MSELoss()

    for i, sample in enumerate(dataloader):
        # Lấy dữ liệu
        sam = sample['image']
        bbb = sample['bb']
        print(bbb.size())
        exit()
        num=bbb.size(1)
        bbb=bbb.view(-1, 5)
        #reset id
        # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]

        ind = torch.arange(params['batch_size'],requires_grad=False).view(-1,1)
        ind = ind.repeat(1,num).view(-1,1)
        bbb[:,0] = ind[:,0]


        tm = sample['tm']
        # print(bbb.shape)
        # print(tm.shape)
        sam,bbb,tm = sam.to(cfg.DEVICE), bbb.to(cfg.DEVICE), tm.to(cfg.DEVICE)

        # roi_pool = ROIPool((50, 50), 1/1)

        # labels_output = roi_pool(tm,bbb)
        labels_output = resizeThermal(tm, bbb)
        labels_output = labels_output.to(cfg.DEVICE)
        # print('label shape',labels_output.shape)

        # Tính giá trị tiên đoán, đạo hàm, và dùng bộ tối ưu hoá để cập nhật trọng số.
        out_RRN = RRN_net(sam,bbb)

        loss = criterion(out_RRN, labels_output)

        # In ra số liệu trong quá trình huấn luyện
        running_loss += loss.item()
        if i % 10 == 9:    # In mỗi 9 mini-batches.
            text = '[{}, {}] loss: {:.3f}  time: {:.3f}'.format(0 + 1, i + 1, running_loss / 10,time.time()-st)
            print(text)
            with open('test2_model21_epoch7.txt','a') as f:
                f.write(text + '\n')
            total_loss.append(running_loss)
            running_loss = 0.0
            st = time.time()
    print("TOTAL: ", sum(total_loss)/len(total_loss))


if __name__ == '__main__':
    train()
