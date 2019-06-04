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
from proposal_processing import compute_targets_pytorch, smooth_l1_loss, get_bbox_regression_labels_pytorch
from config import cfg

raw_vgg16 = vgg.vgg16(pretrained=True)

raw_RRN = MyRRN()
raw_RRN.load_state_dict(torch.load('models/model24/model24_lr_1e-6_bz_6_NBS_128_norm_epoch_9.pth'))
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

        self.score_fc = nn.Linear(4096, cfg.NUM_CLASSES)
        self.bbox_fc = nn.Linear(4096,4)
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

def createTarget(label,bbb,gt_rois):

    bbox_target_data = compute_targets_pytorch(bbb[:,:, 1:5], gt_rois[:,:,:4])
    bbox_targets, bbox_inside_weights = get_bbox_regression_labels_pytorch(bbox_target_data, label, num_classes=cfg.NUM_CLASSES)
    bbox_outside_weights = (bbox_inside_weights > 0).float()
    # print(bbox_outside_weights.size())
    label = label.view(-1).long()
    bbox_targets = bbox_targets.view(-1, bbox_targets.size(2))
    bbox_inside_weights = bbox_inside_weights.view(-1, bbox_inside_weights.size(2))
    bbox_outside_weights = bbox_outside_weights.view(-1, bbox_outside_weights.size(2))

    return label,bbox_targets,bbox_inside_weights,bbox_outside_weights

def train():
    print('TRAINING MSDN...')
    # THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    # ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    # IMGS_CSV = 'mydata/imgs_train.csv'
    # ROIS_CSV = 'mydata/rois_trainKaist_thr70_MSDN.csv'
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),
                                       my_normalize()])
                                  # Normalize(rgb_mean,rgb_std)])

    device = torch.device("cuda:0")
    params = {'batch_size': cfg.TRAIN.BATCH_SIZE,
              'shuffle': cfg.TRAIN.SHUFFLE,
              'num_workers': cfg.TRAIN.NUM_WORKERS}
    print(params)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    print('max_epoch',max_epoch)
    LR = cfg.TRAIN.LEARNING_RATE #learning rate
    print('learning_rate',LR)
    MT = cfg.TRAIN.MOMENTUM #momentum

    my_dataset = MyDataset(imgs_csv=cfg.TRAIN.IMGS_CSV,rois_csv=cfg.TRAIN.ROIS_CSV,
    root_dir=cfg.TRAIN.ROOT_DIR, ther_path=cfg.TRAIN.THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    # print(list(aa.front_subnetB.parameters())[2])
    device = torch.device("cuda:0")

    MSDN_net = MyMSDN()
    MSDN_net.to(device)
    MSDN_net.load_state_dict(torch.load('models/MSDN/model1/model1_lr_1e-3_bz_6_NBS_128_norm_epoch_4.pth'))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,MSDN_net.parameters()), lr=LR, momentum=MT)

    for epoch in range(max_epoch):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
        running_loss = 0.0
        st = time.time()
        for i, sample in enumerate(dataloader):

            label = sample['label']
            bbb = sample['bb']
            gt_rois = sample['gt_roi']

            # label,bbb,gt_rois = label.to(device),bbb.to(device),gt_rois.to(device)

            bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = createTarget(label,bbb,gt_rois)

            bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = bbox_label.to(device),bbox_targets.to(device),bbox_inside_weights.to(device),bbox_outside_weights.to(device)

            sam = sample['image']

            # bbb = sample['bb']

            num=bbb.size(1)
            bbb=bbb.view(-1, 5)

            ind = torch.arange(params['batch_size'],requires_grad=False).view(-1,1)
            ind = ind.repeat(1,num).view(-1,1)
            bbb[:,0] = ind[:,0]


            # print(bbb.shape)
            # print(tm.shape)
            sam = sam.to(device)
            bbb = bbb.to(device)

            cls_score, bbox_pred = MSDN_net(sam,bbb)

            RCNN_loss_cls = F.cross_entropy(cls_score, bbox_label)
            # print(RCNN_loss_cls.mean())
            RCNN_loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
            # print(RCNN_loss_bbox.mean())

            loss = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            # print('loss at {}: '.format(i),loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # In mỗi 2000 mini-batches.
                text = '[{}, {}] loss: {:.3f}  time: {:.3f}'.format(epoch + 1, i + 1, running_loss / 10,time.time()-st)
                print(text)
                with open('models/MSDN/model1/log1.txt','a') as f:
                    f.write(text + '\n')
                running_loss = 0.0
                st = time.time()
        torch.save(MSDN_net.state_dict(), 'models/MSDN/model1/model1_lr_1e-4_bz_2_NBS_128_norm_epoch_{}.pth'.format(epoch))
    print('Huấn luyện xong')

def test():
    print('TESTING MSDN...')

    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test'
    IMGS_CSV = 'mydata/imgs_test.csv'
    ROIS_CSV = 'mydata/rois_trainKaist_thr70_MSDN.csv'
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),
                                       my_normalize()])
                                  # Normalize(rgb_mean,rgb_std)])

    device = torch.device("cuda:0")
    params = {'batch_size':2,
              'shuffle':True,
              'num_workers':24}
    print(params)
    # max_epoch = 5
    # print('max_epoch',max_epoch)
    # LR = 1e-4 #learning rate
    # print('learning_rate',LR)
    # MT = 0.9 #momentum

    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    # print(list(aa.front_subnetB.parameters())[2])
    device = torch.device("cuda:0")

    MSDN_net = MyMSDN()
    MSDN_net.to(device)
    MSDN_net.load_state_dict(torch.load('mymodel/MSDN/model1_lr_1e-4_bz_2_NBS_128_norm_epoch_4.pth'))

    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad,MSDN_net.parameters()), lr=LR, momentum=MT)

    # for epoch in range(max_epoch):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
    running_loss = 0.0
    st = time.time()
    for i, sample in enumerate(dataloader):

        label = sample['label']
        bbb = sample['bb']
        gt_rois = sample['gt_roi']

        # label,bbb,gt_rois = label.to(device),bbb.to(device),gt_rois.to(device)

        bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = createTarget(label,bbb,gt_rois)

        bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = bbox_label.to(device),bbox_targets.to(device),bbox_inside_weights.to(device),bbox_outside_weights.to(device)

        sam = sample['image']

        # bbb = sample['bb']

        num=bbb.size(1)
        bbb=bbb.view(-1, 5)

        ind = torch.arange(params['batch_size'],requires_grad=False).view(-1,1)
        ind = ind.repeat(1,num).view(-1,1)
        bbb[:,0] = ind[:,0]
        # print(bbb.shape)
        # print(tm.shape)
        sam = sam.to(device)
        bbb = bbb.to(device)

        cls_score, bbox_pred = MSDN_net(sam,bbb)

        # RCNN_loss_cls = F.cross_entropy(cls_score, bbox_label)
        # print(RCNN_loss_cls.mean())
        # RCNN_loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        # print(RCNN_loss_bbox.mean())

        # loss = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        # # print('loss at {}: '.format(i),loss.item())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # In mỗi 2000 mini-batches.
            text = '[{}, {}] loss: {:.3f}  time: {:.3f}'.format(epoch + 1, i + 1, running_loss / 10,time.time()-st)
            print(text)
            with open('models/MSDN/model1/log1.txt','a') as f:
                f.write(text + '\n')
            running_loss = 0.0
            st = time.time()
        # torch.save(MSDN_net.state_dict(), 'models/MSDN/model1/model1_lr_1e-4_bz_2_NBS_128_norm_epoch_{}.pth'.format(epoch))
    # print('Huấn luyện xong')


if __name__ == '__main__':
    train()
    # cls, pros = torch.load('out_MSDN_test.pth')
    # print(cls.size())
    # print(pros)
