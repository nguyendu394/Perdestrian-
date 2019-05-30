import torch, sys
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from model.roi_layers import ROIPool
import torchvision.ops.roi_pool as ROIPool
from RRN import MyRRN
import vgg, cv2, time
from mydataset import MyDataset
from torchvision.ops import box_iou
from my_transforms import *

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

raw_vgg16 = vgg.vgg16(pretrained=True)

raw_RRN = MyRRN()
raw_RRN.load_state_dict(torch.load('models/model12/model12_lr_1e-7_bz_7_data_True_epoch_9.ptx'))
#freeze the feature layers of RRN
for param in raw_RRN.features.parameters():
    param.requires_grad = False

class MyMSDN(nn.Module):
    def __init__(self):
        super(MyMSDN, self).__init__()
        self.front_subnetA = raw_vgg16.features
        self.last_subnetA = raw_vgg16.features[-7:]
        self.front_subnetB = raw_RRN.features[:-7]
        self.last_subnetB = raw_RRN.features[-7:]
        # self.front_roi_pool = ROIPool((7, 7), 1/8)
        # self.last_roi_pool = ROIPool((7, 7), 1/16)
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        # self.fc1 = nn.Linear(1024*7*7, 4096)
        # self.avgpool = raw_vgg16.avgpool
        self.FC = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )

        self.score_fc = nn.Linear(4096, 2)
        self.bbox_fc = nn.Linear(4096,8)
        # self.deconv1 = nn.ConvTranspose2d(in_channels=512,out_channels=64,kernel_size=4,stride=8, padding=1)
        # self.fc1 = nn.Linear( 512* 7 * 7, 4096)
        # self.deconv2 = nn.ConvTranspose2d(in_channels=3,out_channels=64,kernel_size=4,stride=8, padding=1)


    def forward(self, x, rois):
        a1 = self.front_subnetA(x)
        a2 = self.last_subnetA(a1)

        b1 = self.front_subnetB(x)
        b2 = self.last_subnetB(b1)

        #front_roi_pool
        x1 = ROIPool(a1,rois,(7, 7), 1/8)
        #last_roi_pool
        x2 = ROIPool(a2,rois,(7, 7), 1/16)
        #front_roi_pool
        x3 = ROIPool(b1,rois,(7, 7), 1/8)
        #last_roi_pool
        x4 = ROIPool(b2,rois,(7, 7), 1/16)

        x = torch.cat((x1,x2,x3,x4),1)
        x = self.conv1(x)
        # x = self.fc1(x)
        # x = F.relu(self.deconv1(x))
        # x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        return cls_prob, bbox_pred

def train():
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_train.csv'
    ROIS_CSV = 'mydata/rois_trainKaist_thr70_0.csv'

    params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 24}
    print(params)
    max_epoch = 10
    LR = 1e-9 #learning rate
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
    # print(list(aa.front_subnetB.parameters())[2])
    device = torch.device("cuda:0")

    MSDN_net = MyMSDN()
    MSDN_net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,MSDN_net.parameters()), lr=LR, momentum=MT)

    for epoch in range(max_epoch):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
        running_loss = 0.0
        st = time.time()
        for i, data in enumerate(dataloader):
            # Lấy dữ liệu
            sample = data
            sam = sample['image']
            bbb = sample['bb']
            num=bbb.size(1)
            bbb=bbb.view(-1, 5)

            ind = torch.arange(params['batch_size'],requires_grad=False).view(-1,1)
            ind = ind.repeat(1,num).view(-1,1)
            bbb[:,0] = ind[:,0]

            tm = sample['tm']
            # print(bbb.shape)
            # print(tm.shape)
            sam,bbb,tm = sam.to(device), bbb.to(device), tm.to(device)

            # roi_pool = ROIPool((50, 50), 1/1)

            # labels_output = roi_pool(tm,bbb)
            # labels_output = resizeThermal(tm, bbb)
            # labels_output = labels_output.to(device)
            # print('label shape',labels_output.shape)

            # Xoá giá trị đạo hàm
            optimizer.zero_grad()

            # Tính giá trị tiên đoán, đạo hàm, và dùng bộ tối ưu hoá để cập nhật trọng số.
            print(sam.shape)
            out_MSDN = MSDN_net(sam,bbb)
            # out_MSDN = raw_vgg16(sam)
            print(len(out_MSDN))
            print(out_MSDN[1].shape)
            torch.save((out_MSDN[0].cpu(),out_MSDN[1].cpu()),'out_RRN.pth')
            exit()
            # loss = criterion(out_RRN, labels_output)
            # print('loss at {}: '.format(i),loss.item())
            # loss.backward()
            # optimizer.step()
if __name__ == '__main__':
    # train()
    cls, pros = torch.load('out_RRN.pth')
    print(cls.size())
    print(pros)
