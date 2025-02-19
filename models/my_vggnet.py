'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import math
import numpy as np


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, leaky_relu=False, layers=2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.layers = layers
        if self.layers == 4:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(planes)
            self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(planes)

    def forward(self, x, before_relu=False, intermediate=0):
        out = self.conv1(x)
        out = self.bn1(out)

        if intermediate == 1:
            return out if before_relu else self.relu(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.layers == 2:
            if self.downsample is not None:
                out = self.downsample(out)
            return out if before_relu else self.relu(out)

        if intermediate == 2:
            return out if before_relu else self.relu(out)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if intermediate == 3:
            return out if before_relu else self.relu(out)

        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            out = self.downsample(out)
        return out if before_relu else self.relu(out)


class VggNet(nn.Module):
    def __init__(self, depth=19, leaky_relu=False, num_classes=10):
        super(VggNet, self).__init__()
        self.leaky_relu = leaky_relu
        self.num_classes = num_classes
        self.depth = depth
        self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, True)
        block = BasicBlock
        self.layers1, self.layers2, self.layers3, self.layers4, self.layers5 = self._make_layers(cfg['VGG19'], block)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)

        self.mixup_feature = None
        self.drop_rate = 1.0
        self.dropout_layer = 0
        ma = np.random.beta(self.drop_rate, self.drop_rate, [128, 1])
        ma_img = ma[:, :, None, None]
        self.mixup_alpha = torch.from_numpy(ma_img).cuda().float()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = x

        if self.dropout_layer > 0:
            self.mixup_feature.requires_grad = False
        if 1 <= self.dropout_layer <= 2:
            out = self.mixup_alpha * out + (1 - self.mixup_alpha) * self.normalizer(self.mixup_feature)

        out = self.layers1(out)

        if 2 < self.dropout_layer <= 4:
            out = self.mixup_alpha * out + (1 - self.mixup_alpha) * self.normalizer(self.mixup_feature)

        out = self.layers2(out)

        if 4 < self.dropout_layer <= 8:
            out = self.mixup_alpha * out + (1 - self.mixup_alpha) * self.normalizer(self.mixup_feature)

        out = self.layers3(out)

        if 8 < self.dropout_layer <= 12:
            out = self.mixup_alpha * out + (1 - self.mixup_alpha) * self.normalizer(self.mixup_feature)

        out = self.layers4(out)

        if 12 < self.dropout_layer <= 16:
            out = self.mixup_alpha * out + (1 - self.mixup_alpha) * self.normalizer(self.mixup_feature)

        out = self.layers5(out)

        if 16 < self.dropout_layer <= 19:
            out = self.mixup_alpha * out + (1 - self.mixup_alpha) * self.normalizer(self.mixup_feature)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, block):
        layers = [[], [], [], [], []]
        in_channels = 3
        out_channels = 3
        layer_index = 0
        layers_num = 0
        for x in cfg:
            # if x == 'M':
            #     layers[layer_index] += [nn.MaxPool2d(kernel_size=2, stride=2)]
            #     layer_index += 1
            # else:
            #     layers[layer_index] += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
            #                nn.BatchNorm2d(x),
            #                self.relu]
            #     in_channels = x

            if x == 'M':
                layers[layer_index].append(block(in_channels, out_channels, downsample=None,
                                                 leaky_relu=self.leaky_relu, layers=layers_num))
                layers[layer_index].append(nn.MaxPool2d(kernel_size=2, stride=2))
                layer_index += 1
                in_channels = out_channels
                layers_num = 0
            else:
                layers_num += 1
                out_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        features = []
        for i in range(layer_index):
            features.append(nn.Sequential(*layers[i]))
        return features

    def get_features(self, x, layer, before_relu=False):
        if 1 <= layer <= 2:
            relative_layer = layer
            x = self.layer_block_forward(x, self.layers1, relative_layer, before_relu=before_relu)
            return x
        x = self.layers1(x)

        if 2 < layer <= 4:
            relative_layer = layer - 2
            x = self.layer_block_forward(x, self.layers2, relative_layer, before_relu=before_relu)
            return x
        x = self.layers2(x)

        if 4 < layer <= 8:
            relative_layer = layer - 4
            x = self.layer_block_forward(x, self.layers3, relative_layer, before_relu=before_relu)
            return x
        x = self.layers3(x)

        if 8 < layer <= 12:
            relative_layer = layer - 8
            x = self.layer_block_forward(x, self.layers4, relative_layer, before_relu=before_relu)
            return x
        x = self.layers4(x)

        if 12 < layer <= 16:
            relative_layer = layer - 12
            x = self.layer_block_forward(x, self.layers5, relative_layer, before_relu=before_relu)
            return x
        x = self.layers5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if 17 <= layer <= self.depth:
            # x = self.fc3(x)
            # return x if before_relu else self.relu(x)
            return self.classifier(x)
        else:
            raise ValueError('layer {:d} is out of index!'.format(layer))

    def layer_block_forward(self, x, layer_block, relative_layer, before_relu=False):
        out = x
        return layer_block[0](out, before_relu, intermediate=relative_layer)
        # if relative_layer == 1:
        #     return layer_block[0](out, before_relu, intermediate=True)

        # out = layer_block[](out)
        # if relative_layer == 2:
        #     return layer_block[1](out, before_relu)
        #
        # out = layer_block[1](out)
        # if relative_layer == 3:
        #     return layer_block[2](out, before_relu)
        #
        # out = layer_block[2](out)
        # if relative_layer == 4:
        #     return layer_block0[3](out, before_relu)

        # out = layer_block[3](out)
        # if relative_layer == 5:
        #     return layer_block[4](out, before_relu, intermediate=True)
        #
        # out = layer_block[4](out)
        # if relative_layer == 6:
        #     return layer_block[5](out, before_relu, intermediate=True)
        #
        # out = layer_block[5](out)
        # if relative_layer == 7:
        #     return layer_block[6](out, before_relu, intermediate=True)
        #
        # out = layer_block[6](out)
        # if relative_layer == 8:
        #     return layer_block[7](out, before_relu, intermediate=True)
        #
        # out = layer_block[7](out)
        # if relative_layer == 9:
        #     return layer_block[8](out, before_relu, intermediate=True)
        #
        # out = layer_block[8](out)
        # if relative_layer == 10:
        #     return layer_block[9](out, before_relu, intermediate=True)
        #
        # out = layer_block[9](out)
        # if relative_layer == 11:
        #     return layer_block[10](out, before_relu, intermediate=True)
        #
        # out = layer_block[10](out)
        # if relative_layer == 12:
        #     return layer_block[11](out, before_relu, intermediate=True)
        #
        # out = layer_block[11](out)
        # if relative_layer == 13:
        #     return layer_block[12](out, before_relu, intermediate=True)
        #
        # out = layer_block[12](out)
        # if relative_layer == 14:
        #     return layer_block[13](out, before_relu, intermediate=True)
        #
        # out = layer_block[13](out)
        # if relative_layer == 15:
        #     return layer_block[14](out, before_relu, intermediate=True)
        #
        # out = layer_block[14](out)
        # if relative_layer == 16:
        #     return layer_block[15](out, before_relu, intermediate=False)

        # raise ValueError('relative_layer is invalid')


# def test():
#     net = VggNet('VGG11')
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())
