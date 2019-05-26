import torch, sys
import torchvision
import torchvision.transforms as transforms
from my_transforms import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.roi_layers import ROIPool
from skimage import io

import vgg, cv2, time
from mydataset import MyDataset, testResizeThermal
from image_processing import resizeThermal,visualizeRP,convertTensor2Img
# from image_processing import showTensor

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

raw_vgg16 = vgg.vgg16(pretrained=True)

class MyRRN(nn.Module):
    def __init__(self):
        super(MyRRN, self).__init__()
        self.features = raw_vgg16.features
        self.roi_pool = ROIPool((7, 7), 1/16)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512,out_channels=64,kernel_size=4,stride=8, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear( 512* 7 * 7, 4096)
        # self.deconv2 = nn.ConvTranspose2d(in_channels=3,out_channels=64,kernel_size=4,stride=8, padding=1)


    def forward(self, x, rois):
        x = self.features(x)
        x = self.roi_pool(x, rois)
        x = F.relu(self.deconv1(x))
        x = self.conv1(x)
        return x

def train():
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_train.csv'
    ROIS_CSV = 'mydata/rois_trainKaist_thr70_1.csv'

    params = {'batch_size': 6,
          'shuffle': True,
          'num_workers': 24}
    max_epoch = 30
    LR = 1e-7 #learning rate
    MT = 0.9 #momentum

    device = torch.device("cuda:0")
    # cudnn.benchmark = True
    # transform = ToTensor()
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),
                                       my_normalize()])
                                  # Normalize(rgb_mean,rgb_std)])

    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)

    dataloader = DataLoader(my_dataset, **params)

    RRN_net = MyRRN()
    RRN_net.to(device)
    NUM_BBS = my_dataset.NUM_BBS
    print('NUM_BBS',NUM_BBS)
    # RRN_net.load_state_dict(torch.load('models/model21/model21_lr_1e-7_bz_6_NBS_128_data_True_epoch_29.ptx'))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,RRN_net.parameters()), lr=LR, momentum=MT)

    for epoch in range(max_epoch):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
        running_loss = 0.0
        st = time.time()
        for i, data in enumerate(dataloader):
            # Lấy dữ liệu
            sample = data
            sam = sample['image']
            bbb = sample['bb']
            bbb=bbb.view(-1, 5)

            #reset id
            # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]
            ind = torch.arange(params['batch_size']).view(-1,1)
            ind = ind.repeat(1,NUM_BBS).view(-1,1)
            bbb[:,0] = ind[:,0]


            # idx = -1
            # for j,v in enumerate(bbb[:,0]):
            #     if not j%NUM_BBS:
            #         idx = idx + 1
            #     bbb[j,0] = idx


            tm = sample['tm']
            # print(bbb.shape)
            # print(tm.shape)
            sam,bbb,tm = sam.to(device), bbb.to(device), tm.to(device)

            # roi_pool = ROIPool((50, 50), 1/1)

            # labels_output = roi_pool(tm,bbb)
            labels_output = resizeThermal(tm, bbb)
            labels_output = labels_output.to(device)
            # print('label shape',labels_output.shape)

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
                with open('models/model23/log23.txt','a') as f:
                    f.write(text + '\n')
                running_loss = 0.0
                st = time.time()
        torch.save(RRN_net.state_dict(), 'models/model23/_model23_lr_1e-7_bz_6_NBS_128_norm_epoch_{}.ptx'.format(epoch))
    print('Huấn luyện xong')



def test():
    TEST_THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test_tm/'
    TEST_ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test'
    IMGS_CSV = 'mydata/imgs_test.csv'
    ROIS_CSV = 'mydata/rois_testKaist_thr70_0.csv'

    params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 24}

    device = torch.device("cuda:0")
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),])
                                  # Normalize(rgb_mean,rgb_std)])

    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=TEST_ROOT_DIR, ther_path=TEST_THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)

    RRN_net = MyRRN()
    RRN_net.to(device)
    RRN_net.load_state_dict(torch.load('models/model21/model21_lr_1e-7_bz_6_NBS_128_data_True_epoch_7.ptx'))
    NUM_BBS = my_dataset.NUM_BBS
    st = time.time()
    running_loss = 0.0
    total_loss = []
    criterion = nn.MSELoss()

    for i, sample in enumerate(dataloader):
        # Lấy dữ liệu
        sam = sample['image']
        bbb = sample['bb']
        bbb=bbb.view(-1, 5)
        #reset id
        # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]

        idx = -1
        for j,v in enumerate(bbb[:,0]):
            if not j%NUM_BBS:
                idx = idx + 1
            bbb[j,0] = idx


        tm = sample['tm']
        # print(bbb.shape)
        # print(tm.shape)
        sam,bbb,tm = sam.to(device), bbb.to(device), tm.to(device)

        # roi_pool = ROIPool((50, 50), 1/1)

        # labels_output = roi_pool(tm,bbb)
        labels_output = resizeThermal(tm, bbb)
        labels_output = labels_output.to(device)
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

def testRRN_Pretrain(sample,pre):
    RRN_net = MyRRN()
    RRN_net.to(device)
    RRN_net.load_state_dict(torch.load(pre))
    NUM_BBS = my_dataset.NUM_BBS

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
    # labels_output = labels_output.to(device)

    # print(labels_output.size())
    # labels_output = labels_output.type('torch.ByteTensor')
    out = labels_output.cpu()
    out = out.detach().numpy()
    ther = convertTensor2Img(tm,0)
    imgg = visualizeRP(ther, bbb)

    cv2.imshow('aa',imgg)
    # plt.imshow(imgg)
    # plt.show()
    # sam,bbb,tm = sam.to(device), bbb.to(device), tm.to(device)

    # out_RRN = RRN_net(sam,bbb)
    # out_RRN = out_RRN.cpu().detach().numpy()

    # criterion = nn.MSELoss()
    # loss = criterion(out_RRN,labels_output)
    # print(loss)

    # print(out_RRN.size())
    # print(abs(out_RRN-labels_output).sum()/(50*50))

    for ind,labels in enumerate(out_RRN):
        # print('output')
        # print(labels)
        print(np.min(labels))
        print(labels.transpose((1, 2, 0))[0])
        cv2.imshow('rrn{}'.format(ind), labels.transpose((1, 2, 0)))
        # plt.show()
        # plt.imshow(labels.transpose((1, 2, 0)))

    for ind,labels in enumerate(out):
        cv2.imshow('bbs{}'.format(ind), labels.transpose((1, 2, 0)))
        # plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    train()

    # THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    # ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    # IMGS_CSV = 'mydata/imgs_train.csv'
    # ROIS_CSV = 'mydata/rois_trainKaist_thr70_0.csv'
    # full_transform=transforms.Compose([RandomHorizontalFlip(p=5),
    #                                    ToTensor(),])
    #                               # Normalize(rgb_mean,rgb_std)])
    #
    # device = torch.device("cuda:0")
    # params = {'batch_size':1,
    #           'shuffle':True,
    #           'num_workers':24}
    #
    #
    # my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    # root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)
    # print(my_dataset.__len__())
    # dataloader = DataLoader(my_dataset, **params)
    # dataiter = iter(dataloader)
    # sample = dataiter.next()
    # # sample = my_dataset[789]
    # NUM_BBS = my_dataset.NUM_BBS
    # testRRN_Pretrain(sample, 'models/model21/model21_lr_1e-7_bz_6_NBS_128_data_True_epoch_7.ptx')
