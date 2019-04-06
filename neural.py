import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vgg

# Thủ tục này giúp hiển thị ảnh
def imshow(img):
    img = img / 2 + 0.5     # Ánh xạ giá trị lại khoảng [0, 1].
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('Máy Bay', 'Ôtô', 'Chim', 'Mèo',
           'Hươu', 'Chó', 'Ếch', 'Ngựa', 'Thuyền', 'Xe Tải')

# Lấy vài tấm ảnh huấn luyện ngẫu nhiên
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # Hiển thị ảnh
# imshow(torchvision.utils.make_grid(images))
# # In nhãn
# print('   /   '.join('%5s' % classes[labels[j]] for j in range(4)))




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0")
# net = vgg.vgg13(pretrained=True)
print('aaa')
# net.load_state_dict(torch.load('modelvgg'))
net = Net()
# print(net)
# params = list(net.parameters())
# print(params[0])

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # Lặp qua bộ dữ liệu huấn luyện nhiều lần
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Lấy dữ liệu
        inputs, labels = data
        # print(inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        # Xoá giá trị đạo hàm
        optimizer.zero_grad()

        # Tính giá trị tiên đoán, đạo hàm, và dùng bộ tối ưu hoá để cập nhật trọng số.
        outputs = net(inputs)
        print(outputs.shape)
        input()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # In ra số liệu trong quá trình huấn luyện
        running_loss += loss.item()
        if i % 2000 == 1999:    # In mỗi 2000 mini-batches.
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
torch.save(net.state_dict(), 'modelvgg')
print('Huấn luyện xong')

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(images)
#         # print(outputs)
#         # input()
#         # outputs.to(device)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Độ chính xác của mạng trên 10000 ảnh trong tập kiểm tra: %d %%' % (
#     100 * correct / total))

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(images)
#         outputs.to(device)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(10):
#     print('Độ chính xác của loại %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# _, predicted = torch.max(outputs, 1)
