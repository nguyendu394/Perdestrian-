import torch, sys
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.ops.roi_pool as ROIPool
from RRN import MyRRN
import vgg, cv2, time
from mydataset import MyDataset
from torchvision.ops import box_iou, nms
from my_transforms import *
from proposal_processing import compute_targets_pytorch, smooth_l1_loss, get_bbox_regression_labels_pytorch, bbox_transform_inv, clip_boxes
from config import cfg
from image_processing import showBbs
import argparse
raw_vgg16 = vgg.vgg16(pretrained=True)

raw_RRN = MyRRN()
raw_RRN.load_state_dict(torch.load('mymodel/RRN/model27_lr_1e-6_bz_6_NBS_128_norm_epoch_9.pth'))
if cfg.TRAIN.FREEZE_RRN:
    print('Freeze RRN parameters')
    #freeze the feature layers of RRN
    for param in raw_RRN.features.parameters():
        param.requires_grad = False
else:
    print('Unfreeze RRN parameters')

class MyMSDN(nn.Module):
    def __init__(self):
        super(MyMSDN, self).__init__()
        self.front_subnetA = raw_vgg16.features[:-7]
        self.last_subnetA = raw_vgg16.features[-7:]
        self.front_subnetB = raw_RRN.features[:-7]
        self.last_subnetB = raw_RRN.features[-7:]

        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)

        self.FC = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.score_fc = nn.Linear(4096, cfg.NUM_CLASSES)
        self.bbox_fc = nn.Linear(4096,4)

    def forward(self, x, rois):
        a1 = self.front_subnetA(x)

        a2 = self.last_subnetA(a1)

        b1 = self.front_subnetB(x)
        b2 = self.last_subnetB(b1)

        #front_roi_pool subsetA
        x1 = ROIPool(a1,rois,(7, 7), 1/8)
        #last_roi_pool subsetA
        x2 = ROIPool(a2,rois,(7, 7), 1/16)

        #front_roi_pool subsetB
        x3 = ROIPool(b1,rois,(7, 7), 1/8)
        #last_roi_pool subsetB
        x4 = ROIPool(b2,rois,(7, 7), 1/16)

        x = torch.cat((x1,x2,x3,x4),1)
        x = self.conv1(x)

        x = x.view(x.size(0), -1)
        x = self.FC(x)

        cls_score = self.score_fc(x)
        # cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        return cls_score, bbox_pred

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

    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),
                                       Normalize(cfg.BGR_MEAN,cfg.BGR_STD)])

    params = {'batch_size': cfg.TRAIN.BATCH_SIZE,
              'shuffle': cfg.TRAIN.SHUFFLE,
              'num_workers': cfg.TRAIN.NUM_WORKERS}
    print(params)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    print('max_epoch',max_epoch)
    LR = cfg.TRAIN.LEARNING_RATE #learning rate
    print('learning_rate',LR)
    MT = cfg.TRAIN.MOMENTUM #momentum
    W_DECAY = cfg.TRAIN.WEIGHT_DECAY
    print('weight decay',W_DECAY)

    my_dataset = MyDataset(imgs_csv=cfg.TRAIN.IMGS_CSV,rois_csv=cfg.TRAIN.ROIS_CSV,
    root_dir=cfg.TRAIN.ROOT_DIR, ther_path=cfg.TRAIN.THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)

    MSDN_net = MyMSDN()
    MSDN_net.to(cfg.DEVICE)

    #load pretrain model
    pretrain = 'models/MSDN/model10/model10_lr_1e-3_bz_2_decay_epoch_4.pth'
    print('pretrain: ' + pretrain)
    MSDN_net.load_state_dict(torch.load(pretrain))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,MSDN_net.parameters()), lr=LR, momentum=MT,weight_decay=W_DECAY)

    f = open('models/MSDN/model10/log10.txt','a')
    for epoch in range(max_epoch):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
        running_loss = 0.0
        st = time.time()
        for i, sample in enumerate(dataloader):

            label = sample['label']
            bbb = sample['bb']
            gt_rois = sample['gt_roi']
            # label,bbb,gt_rois = label.to(cfg.DEVICE),bbb.to(cfg.DEVICE),gt_rois.to(cfg.DEVICE)

            bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = createTarget(label,bbb,gt_rois)

            bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = bbox_label.to(cfg.DEVICE),bbox_targets.to(cfg.DEVICE),bbox_inside_weights.to(cfg.DEVICE),bbox_outside_weights.to(cfg.DEVICE)

            sam = sample['image']

            num=bbb.size(1)
            bbb=bbb.view(-1, 5)

            ind = torch.arange(params['batch_size'],requires_grad=False).view(-1,1)
            ind = ind.repeat(1,num).view(-1,1)
            bbb[:,0] = ind[:,0]

            sam = sam.to(cfg.DEVICE)
            bbb = bbb.to(cfg.DEVICE)

            cls_score, bbox_pred = MSDN_net(sam,bbb)

            RCNN_loss_cls = F.cross_entropy(cls_score, bbox_label)

            RCNN_loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
            # print(RCNN_loss_bbox.mean())

            loss = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            # print('loss at {}: '.format(i),loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # In mỗi 10 mini-batches.
                text = '[{}, {}] loss: {:.3f}  time: {:.3f}'.format(epoch + 1, i + 1, running_loss / 10,time.time()-st)
                print(text)
                #write txt file
                f.write(text + '\n')
                running_loss = 0.0
                st = time.time()
        name_model = 'models/MSDN/model10/model10_lr_1e-4_bz_2_decay_epoch_{}.pth'.format(epoch)
        torch.save(MSDN_net.state_dict(), name_model)
    f.close()
    print('model saved: ' + name_model)
    print('Huấn luyện xong')

def test():
    print('TESTING MSDN...')
    full_transform=transforms.Compose([ToTensor(),
                                       Normalize(cfg.BGR_MEAN,cfg.BGR_STD)])

    params = {'batch_size': 1,
              'shuffle':False,
              'num_workers':cfg.TRAIN.NUM_WORKERS}
    print(params)
    print('score thress: {}'.format(cfg.TEST.THRESS ))

    my_dataset = MyDataset(imgs_csv=cfg.TEST.IMGS_CSV,rois_csv=cfg.TEST.ROIS_CSV,
    root_dir=cfg.TEST.ROOT_DIR, ther_path=cfg.TEST.THERMAL_PATH,transform = full_transform,train=False)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)

    MSDN_net = MyMSDN()
    MSDN_net.to(cfg.DEVICE)
    pretrain = 'models/MSDN/model10/model10_lr_1e-4_bz_2_decay_epoch_3.pth'
    print('pretrain: ' + pretrain)
    MSDN_net.load_state_dict(torch.load(pretrain))

    running_loss = 0.0
    st = time.time()

    f = open('mymodel/MSDN/test/test_model10_lr_1e-4_bz_2_decay_epoch_3.txt','a')
    for i, sample in enumerate(dataloader):
        print(sample['img_info'])
        label = sample['label']
        bbb = sample['bb']
        gt_rois = sample['gt_roi']

        # label,bbb,gt_rois = label.to(cfg.DEVICE),bbb.to(cfg.DEVICE),gt_rois.to(cfg.DEVICE)

        bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = createTarget(label,bbb,gt_rois)

        bbox_label,bbox_targets,bbox_inside_weights,bbox_outside_weights = bbox_label.to(cfg.DEVICE),bbox_targets.to(cfg.DEVICE),bbox_inside_weights.to(cfg.DEVICE),bbox_outside_weights.to(cfg.DEVICE)


        sam = sample['image']
        boxes = bbb[:, :, 1:5].to(cfg.DEVICE)
        # print('boxes of size', boxes)
        # bbb = sample['bb']

        num=bbb.size(1)
        bbb=bbb.view(-1, 5)

        ind = torch.arange(params['batch_size'],requires_grad=False).view(-1,1)
        ind = ind.repeat(1,num).view(-1,1)
        bbb[:,0] = ind[:,0]
        sam = sam.to(cfg.DEVICE)
        bbb = bbb.to(cfg.DEVICE)

        cls_score, bbox_pred = MSDN_net(sam,bbb)

        scores = F.softmax(cls_score).detach()
        # print('score of size', scores.shape)

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.detach()
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(cfg.DEVICE) \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(cfg.DEVICE)
                box_deltas = box_deltas.view(1, -1, 4)

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            # print(pred_boxes)
            pred_boxes = clip_boxes(pred_boxes, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        #clear all dimension 1
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        # print(scores.shape)
        # print(pred_boxes.shape)

        inds = torch.nonzero(scores[:,1]>cfg.TEST.THRESS).view(-1)
        # print(inds)
        if inds.numel() > 0:
            # print(scores)
            cls_scores = scores[:,1][inds]
            # print(cls_score.shape)
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds, :]
            # print(cls_boxes.shape)
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            # print(cls_dets)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            # print(cls_dets)
        else:
            cls_dets = torch.Tensor([[],[],[],[],[]]).permute(1,0)

        print('writing image {}'.format(i+1))
        if cls_dets.numel() > 0:
            bbs = cls_dets.cpu().detach().numpy()
            for bb in bbs:
                l,t,r,b,s = bb
                w = r-l
                h = b-t
                # if w > 0 and h >= cfg.TEST.MIN_HEIGHT and h <= cfg.TEST.MAX_HEIGHT:
                f.write('{id},{l:9.3f},{t:9.3f},{w:9.3f},{h:9.3f},{s:9.3f}\n'.format(id=i+1,l=l,t=t,w=w,h=h,s=s*100))
    f.close()

def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a MSDN network')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()

    if args.test:
        test()
    else:
        train()

    # cls, pros = torch.load('out_MSDN_test.pth')
    # print(cls.size())
    # print(pros)
