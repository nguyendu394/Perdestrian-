import torch, sys
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.roi_layers import ROIPool

import vgg, cv2
from mydataset import MyDataset, ToTensor
from image_processing import showTensor

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



# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=1,
#                                          shuffle=False, num_workers=2)

def main():
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/images/set00/V000/visible'
    IMGS_CSV = 'mydata/imgs_name_set00_v000.csv'
    ROIS_CSV = 'mydata/rois_set00_v000.csv'

    params = {'batch_size': 5,
          'shuffle': False,
          'num_workers': 24}
    max_epoch = 10
    LR = 0.000000001 #learning rate
    MT = 0.9 #momentum

    device = torch.device("cuda:0")
    # cudnn.benchmark = True
    transform = ToTensor()

    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = transform)

    dataloader = DataLoader(my_dataset, **params)

    RRN_net = MyRRN()
    print(RRN_net)
    input()
    RRN_net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(RRN_net.parameters(), lr=LR, momentum=MT)

    for epoch in range(max_epoch):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # Lấy dữ liệu
            sample = data
            sam = sample['image']
            bbb = sample['bb']
            bbb=bbb.view(-1, 5)
            #reset id
            bbb[:, 0] = bbb[:, 0] - bbb[0, 0]
            print(bbb)
            input("AAAA")
            tm = sample['tm']
            # print(sam.shape)
            # print(bbb.shape)
            # print(tm.shape)
            sam,bbb,tm = sam.to(device), bbb.to(device), tm.to(device)

            roi_pool = ROIPool((50, 50), 1/1)

            labels_output = roi_pool(tm,bbb)
            # print('label shape',labels_output.shape)

            # Xoá giá trị đạo hàm
            optimizer.zero_grad()

            # Tính giá trị tiên đoán, đạo hàm, và dùng bộ tối ưu hoá để cập nhật trọng số.
            out_RRN = RRN_net(sam,bbb)
            # print('output RRN',out_RRN.shape)
            loss = criterion(out_RRN, labels_output)
            print('loss at {}: '.format(i),loss.item())
            loss.backward()
            optimizer.step()

            # In ra số liệu trong quá trình huấn luyện
            running_loss += loss.item()
            if i % 10 == 0:    # In mỗi 2000 mini-batches.
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    torch.save(net.state_dict(), 'firstmodel.ptx')
    print('Huấn luyện xong')

if __name__ == '__main__':
    main()
# paras = list(net.parameters())
# print(paras[0])
