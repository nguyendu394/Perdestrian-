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


# sys.path.append('/mnt/01D3F51B1A6C3EA0/UIT/Duyld/paper/DACN/home/dungnm/pytorch/faster-rcnn.pytorch/lib')
# from utils.blob import im_list_to_blob

raw_vgg13 = vgg.vgg16(pretrained=True)

class MyRRN(nn.Module):
    def __init__(self):
        super(MyRRN, self).__init__()
        self.features = raw_vgg13.features
        self.roi_pool = ROIPool((7, 7), 1/16)
        # self.fc1 = nn.Linear( 512* 7 * 7, 4096)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512,out_channels=64,kernel_size=4,stride=8, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        # self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)


    def forward(self, x, rois):
        x = self.features(x)
        # x = self.maxunpool1(x)
        print(rois.size())
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



if __name__ == '__main__':
    transform = ToTensor()
    device = torch.device("cuda:0")
    my_dataset = MyDataset(imgs_csv='mydata/imgs.csv',rois_csv='mydata/rois.csv',
    root_dir='mydata/imgs', transform = transform)

    dataloader = DataLoader(my_dataset, batch_size=1,
    shuffle=True, num_workers=1)

    # dataiter = iter(dataloader)
    # sample = dataiter.next()
    # print(sample['bb'])

    # roi_pool = ROIPool((7, 7), 1/16)
    # # roi_pool = RoIPooling2D((7,7),1/16)
    # sam = sample['image'].type('torch.DoubleTensor')
    # bbb = sample['bb'].type('torch.DoubleTensor')
    # sam, bbb = sam.to(device),bbb.to(device)
    #
    # pooled_output = roi_pool(sam,bbb)
    # print(raw_vgg13)


    # img = cv2.imread('1256.jpg')
    # print(img.shape)
# img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('aaa',img)

    # img = np.expand_dims(img.T, axis=0)
    # print(img.shape)
    # inp = torch.from_numpy(img)
    # inp = inp.type(torch.FloatTensor)

#
# def imshow(img):
#     img = img / 2 + 0.5     # Ánh xạ giá trị lại khoảng [0, 1].
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#+++++++++++++
dataiter = iter(dataloader)
sample = dataiter.next()
sam = sample['image']
bbb = sample['bb']
print(bbb)
bbb=bbb.view(-1, 5)
sam, bbb = sam.to(device),bbb.to(device)
# # imshow(torchvision.utils.make_grid(images))
# print(type(images))
# print(type(inp))
# # print(labels)
net = MyRRN()
print(net)
net.to(device)
# #
# #
out = net(sam,bbb)
print(out.shape)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.000000001, momentum=0.9)

res = out.cpu()
res = res.detach().numpy()

# img = res[0]
# img = img.transpose((1, 2, 0))
# # print(img.shape)
# to_pil = torchvision.transforms.ToPILImage()
# img = to_pil(img)
# img.show()
#=========
# for i in range(1):
#     img = res[i].transpose((1, 2, 0))
#     cv2.imshow('aa {}.jpg'.format(i), img)
# #
# # plt.plot(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# paras = list(net.parameters())
# print(paras[0])
