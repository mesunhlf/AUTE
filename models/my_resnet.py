# https://github.com/NVlabs/AdaBatch/blob/master/models/cifar/resnet.py
import torch.nn as nn
import math
import torch
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, leaky_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, before_relu=False, intermediate=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if intermediate:
            return out if before_relu else self.relu(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out if before_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, leaky_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, before_relu=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out if before_relu else self.relu(out)


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10, leaky_relu=False, mask=0):
        super(ResNet, self).__init__()
        self.leaky_relu = leaky_relu
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        self.n = n
        self.depth = depth

        self.mask = mask
        # self.dropout = torch.nn.Dropout(id * 0.02)
        self.dropout = torch.nn.Dropout(self.mask)

        self.mixup_feature = None
        self.drop_rate = 1.0
        self.dropout_layer = 0
        ma = np.random.beta(self.drop_rate, self.drop_rate, [64, 1])
        ma_img = ma[:, :, None, None]
        self.mixup_alpha = torch.from_numpy(ma_img).cuda().float()

        self.mixup_alpha1 = torch.from_numpy(
            np.random.beta(self.drop_rate, self.drop_rate, [64, 1])[:, :, None, None]).cuda().float()
        self.mixup_alpha2 = torch.from_numpy(
            np.random.beta(self.drop_rate, self.drop_rate, [64, 1])[:, :, None, None]).cuda().float()
        self.mixup_alpha3 = torch.from_numpy(
            np.random.beta(self.drop_rate, self.drop_rate, [64, 1])[:, :, None, None]).cuda().float()
        self.mixup_alpha4 = torch.from_numpy(
            np.random.beta(self.drop_rate, self.drop_rate, [64, 1])[:, :, None, None]).cuda().float()
        self.mf1 = []
        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)  # original 8
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # self.sm = nn.Softmax(dim=1)
        self.bernoulli = torch.distributions.Bernoulli(torch.tensor([0.9]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, leaky_relu=self.leaky_relu))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, leaky_relu=self.leaky_relu))

        return nn.Sequential(*layers)

    def norm(self):
        arr = torch.cat([self.mixup_alpha1, self.mixup_alpha2, self.mixup_alpha3, self.mixup_alpha4], 0)
        mean = torch.mean(arr, 0)
        std = torch.std(arr, 0)
        return (self.mixup_alpha1 - mean) / std, (self.mixup_alpha2 - mean) / std, (self.mixup_alpha3 - mean) / std, (self.mixup_alpha4 - mean) / std

    def shuffle(self, x):
        c = x.size(1)
        rnd = torch.randint(0, c-1, (1,1))
        idx = rnd[0][0]
        tmp = x[:,0:idx,:,:]
        x[:,0:c-idx,:,:] = x[:,idx:c,:,:]
        x[:,c-idx:c,:,:] = tmp
        return x

    def forward(self, x):
        if self.dropout_layer > 0:
            self.mixup_feature.requires_grad = False
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 16x32x32
        # mask = torch.randint(0, 2, x.size()).cuda()
        # mask = torch.bernoulli(torch.tensor(x.size()).uniform_(0, 0.6))
        x = self.layer1(x)  # 16x32x32
        if(self.mask > 0):
            x = self.dropout(x)
        # x = self.shuffle(x)
        # x = x[:, torch.randperm(x.size(1)), :, :]
        x = self.layer2(x)  # 32x16x16
        # x = self.dropout(x)
        # mask = torch.squeeze(self.bernoulli.sample(x.size()), dim=-1).cuda()
        # x1 = x * mask
        # x2 = x * (1-mask)
        # x = self.layer3(x1) + self.layer3(x2)
        x = self.layer3(x)  # 64x8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x, layer, before_relu=False):
        layers_per_block = 2 * self.n

        x = self.conv1(x)
        x = self.bn1(x)

        if layer == 1:
            return x

        x = self.relu(x)

        if layer > 1 and layer <= 1 + layers_per_block:
            relative_layer = layer - 1
            x = self.layer_block_forward(x, self.layer1, relative_layer, before_relu=before_relu)
            return x

        x = self.layer1(x)
        if layer > 1 + layers_per_block and layer <= 1 + 2 * layers_per_block:
            relative_layer = layer - (1 + layers_per_block)
            x = self.layer_block_forward(x, self.layer2, relative_layer, before_relu=before_relu)
            return x

        x = self.layer2(x)
        if layer > 1 + 2 * layers_per_block and layer <= 1 + 3 * layers_per_block:
            relative_layer = layer - (1 + 2 * layers_per_block)
            x = self.layer_block_forward(x, self.layer3, relative_layer, before_relu=before_relu)
            return x

        x = self.layer3(x)
        if layer == self.depth:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
        else:
            raise ValueError('layer {:d} is out of index!'.format(layer))

    def get_features_finals(self, x, layer, before_relu=False):
        layers_per_block = 2 * self.n

        features = x
        finals = x

        x = self.conv1(x)
        x = self.bn1(x)

        if layer == 1:
            features = x

        x = self.relu(x)

        if layer > 1 and layer <= 1 + layers_per_block:
            relative_layer = layer - 1
            features = self.layer_block_forward(x, self.layer1, relative_layer, before_relu=before_relu)

        x = self.layer1(x)
        if layer > 1 + layers_per_block and layer <= 1 + 2 * layers_per_block:
            relative_layer = layer - (1 + layers_per_block)
            features = self.layer_block_forward(x, self.layer2, relative_layer, before_relu=before_relu)

        x = self.layer2(x)
        if layer > 1 + 2 * layers_per_block and layer <= 1 + 3 * layers_per_block:
            relative_layer = layer - (1 + 2 * layers_per_block)
            features = self.layer_block_forward(x, self.layer3, relative_layer, before_relu=before_relu)

        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        finals = x
        if layer == self.depth:
            features = x
        return features, finals

    def layer_block_forward(self, x, layer_block, relative_layer, before_relu=False):
        out = x
        if relative_layer == 1:
            return layer_block[0](out, before_relu, intermediate=True)

        if relative_layer == 2:
            return layer_block[0](out, before_relu, intermediate=False)

        out = layer_block[0](out)
        if relative_layer == 3:
            return layer_block[1](out, before_relu, intermediate=True)

        if relative_layer == 4:
            return layer_block[1](out, before_relu, intermediate=False)

        out = layer_block[1](out)
        if relative_layer == 5:
            return layer_block[2](out, before_relu, intermediate=True)

        if relative_layer == 6:
            return layer_block[2](out, before_relu, intermediate=False)

        raise ValueError('relative_layer is invalid')

    def infer(self, x):
        return self.forward(x).max(1, keepdim=True)[1]


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
