# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:45:06 2019

@author: Junfeng Hu
"""

import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CUDA_VISIBLE_DEVICES = (2)
# 定义是否使用GPU
BATCH_SIZE = 256
LR = 0.01
EPOCH = 3


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):  # 构造函数
        super(VGG, self).__init__()
        # 网络结构（仅包含卷积层和池化层，不包含分类器）
        self.features = features
        self.classifier = nn.Sequential(  # 分类器结构
            # fc6
            nn.Linear(256 * 1 * 1, 1024),
            nn.ReLU(),
            nn.Dropout(),

            # fc7
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),

            # fc8
            nn.Linear(512, num_classes))
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

cfg = {
    'A': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 生成网络每层的信息
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 设定卷积层的输出数量
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # 返回一个包含了网络结构的时序容器


def vgg16(**kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    # model.load_state_dict(torch.load(model_path))
    return model


def getData():  # 定义数据预处理
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307],
                             std=[0.3081])])
    trainset = tv.datasets.CIFAR10(
        root='CIFAR10/',  # dataset存储路径
        train=True,  # True表示是train训练集，Flase表示测试集
        transform=transform,  # tv.transforms.ToTensor(),  #将原数据规范化到（0,1）之间
        download=True,
    )
    testset = tv.datasets.CIFAR10(
        root='CIFAR10/',  # dataset存储路径
        train=False,  # True表示是train训练集，False表示test测试集
        transform=transform,  # tv.transforms.ToTensor(),
        download=True,
    )
    # trainset = tv.datasets.CIFAR10(root='./root/', train=True, transform=transform, download=True)
    # testset = tv.datasets.CIFAR10(root='./root/', train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train():
    trainset_loader, testset_loader = getData()
    net = vgg16().cuda()
    net.train()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    # Train the model
    losses = []
    acces = []
    test_losses = []
    test_acces = []
    for epoch in range(EPOCH):
        train_loss = 0
        acc_tmp = []
        for step, (inputs, labels) in enumerate(trainset_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            output = net(inputs)
            loss = criterion(output, labels)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == labels).sum().item()
            acc_tmp.append(correct / total)

            train_loss += loss.item()
            if step % 400 == 0:
                print('Epoch:', epoch, '|Step:', step,
                      '|train loss:%.4f' % loss.data.item())
                acc = test(net, testset_loader)
                # print('Epoch', epoch, '|step ', step, 'loss: %.4f' %loss.item(), 'test accuracy:%.4f' %acc)
        losses.append(train_loss / len(trainset_loader))
        acces.append(sum(acc_tmp) / len(acc_tmp))
        print('Finished Training')
        acc, los = test(net, testset_loader)
        print('Epoch', epoch, '|step ', step, 'loss: %.4f' % loss.item(), 'test accuracy:%.4f' % acc)
        test_losses.append(los)
        test_acces.append(acc)
    plt.plot(losses)
    plt.plot(test_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('256_vgg_loss.png')
    plt.plot(acces)
    plt.plot(test_acces)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('256_vgg_acce.png')
    return net


def test(net, testdata):
    criterion = nn.CrossEntropyLoss()
    correct, total = .0, .0
    net.eval()
    los = []
    for inputs, labels in testdata:
        inputs, labels = inputs.cuda(), labels.cuda()
        # inputs = inputs.cuda()
        # labels = labels.cuda()
        outputs = net(inputs)
        los.append(criterion(outputs, labels).item())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    net.train()
    return float(correct) / total, sum(los) / len(los)


if __name__ == '__main__':
    net = train()