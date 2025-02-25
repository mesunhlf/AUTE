import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from advertorch.utils import NormalizeByChannelMeanStd

from models.my_resnet import ResNet
from models.my_vggnet import VggNet
from models.ensemble import Ensemble
from models.preact_resnet import PreActResNet18
from models.resnet18 import ResNet18
from models.wideresnet import WideResNet

###################################
# Models                          #
###################################
def get_models(args, train=True, as_ensemble=False, model_file=None, leaky_relu=False, is_paral=False):
    models = []
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    if model_file:
        state_dict = torch.load(model_file)
        # state_dict = torch.load(model_file, map_location='cuda:0')
        if train:
            print('Loading pre-trained models...')
    
    iter_m = state_dict.keys() if model_file else range(args.model_num)
    for i in iter_m:
        print(args)
        if args.arch.lower() == 'resnet20':
            model = ResNet(depth=20, leaky_relu=leaky_relu, num_classes=args.num_class)
        elif args.arch.lower() == 'vggnet':
            model = VggNet(depth=args.depth, leaky_relu=leaky_relu, num_classes=args.num_class)
        elif args.arch.lower() == 'preact_resnet18':
            model = PreActResNet18(num_classes=args.num_class)
        elif args.arch.lower() == 'resnet18':
            model = ResNet18(num_classes=args.num_class)
        elif args.arch.lower() == 'wideresnet34':
            model = WideResNet(num_classes=args.num_class, depth=34, widen_factor=10)
        else:
            raise ValueError('[{:s}] architecture is not supported yet...')
        # we include input normalization as a part of the model
        model = ModelWrapper(model, normalizer)

        if is_paral:
            model = nn.DataParallel(model)
        if model_file:
            model.load_state_dict(state_dict[i])
        if train:
            model.train()
        else:
            model.eval()
        model = model.cuda()
        models.append(model)

    if as_ensemble:
        assert not train, 'Must be in eval mode when getting models to form an ensemble'
        ensemble = Ensemble(models)
        ensemble.eval()
        return ensemble
    else:
        return models

def get_single_model(args, train=True, as_ensemble=False, model_file=None, leaky_relu=False, is_paral=False):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    if model_file:
        state_dict = torch.load(model_file)
        # state_dict = torch.load(model_file, map_location='cuda:0')
        if train:
            print('Loading pre-trained models...')

    iter_m = state_dict.keys() if model_file else range(args.model_num)
    for i in iter_m:
        print(args)
        if args.arch.lower() == 'resnet':
            # if(args.mask != None):
            #     model = ResNet(depth=args.depth, leaky_relu=leaky_relu, num_classes=args.num_class, mask=args.mask)
            # else:
            model = ResNet(depth=args.depth, leaky_relu=leaky_relu, num_classes=args.num_class)
        elif args.arch.lower() == 'lenet':
            model = LeNet(leaky_relu=leaky_relu, num_classes=args.num_class)
        elif args.arch.lower() == 'vggnet':
            model = VggNet(depth=args.depth, leaky_relu=leaky_relu, num_classes=args.num_class)
        elif args.arch.lower() == 'preact_resnet':
            model = PreActResNet18(num_classes=args.num_class)
        else:
            raise ValueError('[{:s}] architecture is not supported yet...')
        # we include input normalization as a part of the model
        model = ModelWrapper(model, normalizer)
        if is_paral:
            model = nn.DataParallel(model)
        if model_file:
            model.load_state_dict(state_dict[i])
        if train:
            model.train()
        else:
            model.eval()
        model = model.cuda()
        return model

def get_ensemble(args, train=False, model_file=None, leaky_relu=False):
    return get_models(args, train, as_ensemble=True, model_file=model_file, leaky_relu=leaky_relu)


class ModelWrapper(nn.Module):
    def __init__(self, model, normalizer):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x):
        x = self.normalizer(x)
        return self.model(x)
    
    def get_features(self, x, layer, before_relu=True):
        x = self.normalizer(x)
        return self.model.get_features(x, layer, before_relu)

    def get_features_finals(self, x, layer, before_relu=True):
        x = self.normalizer(x)
        features, finals = self.model.get_features_finals(x, layer, before_relu)
        return features, finals


###################################
# data loader
###################################
def get_loaders(args, add_gaussian=False):
    kwargs = {'num_workers': 4,
              'batch_size': args.batch_size,
              'shuffle': True,     # default True
              'pin_memory': True}
    if not add_gaussian:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.045) #https://arxiv.org/pdf/1901.09981.pdf
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    cifar_func = datasets.CIFAR10 if args.num_class == 10 else datasets.CIFAR100
    trainset = cifar_func(root=args.data_dir, train=True,
                                transform=transform_train,
                                download=True)
    testset = cifar_func(root=args.data_dir, train=False,
                                transform=transform_test,
                                download=True)
    # trainset = datasets.CIFAR10(root=args.data_dir, train=True,
    #                             transform=transform_train,
    #                             download=True)
    # testset = datasets.CIFAR10(root=args.data_dir, train=False,
    #                             transform=transform_test,
    #                             download=True)
    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)   # 4, 100, F, T
    # testloader = DataLoader(testset, num_workers=4, batch_size=100, shuffle=True, pin_memory=True)
    return trainloader, testloader


def get_testloader(args, train=False, batch_size=100, shuffle=False, subset_idx=None):
    kwargs = {'num_workers': 4,
              'batch_size': batch_size,
              'shuffle': shuffle,
              'pin_memory': True}
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    cifar_func = datasets.CIFAR10 if args.num_class == 10 else datasets.CIFAR100
    if subset_idx is not None:
        testset = Subset(cifar_func(root=args.data_dir, train=train,
                                transform=transform_test,
                                download=False), subset_idx)
    else:
        testset = cifar_func(root=args.data_dir, train=train,
                                transform=transform_test,
                                download=False)

    # if subset_idx is not None:
    #     testset = Subset(datasets.CIFAR10(root=args.data_dir, train=train,
    #                             transform=transform_test,
    #                             download=False), subset_idx)
    # else:
    #     testset = datasets.CIFAR10(root=args.data_dir, train=train,
    #                             transform=transform_test,
    #                             download=False)
    testloader = DataLoader(testset, **kwargs)
    return testloader


class DistillationLoader:
    def __init__(self, seed, target):
        self.seed = iter(seed)
        self.target = iter(target)

    def __len__(self):
        return len(self.seed)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            si, sl = next(self.seed)
            ti, tl = next(self.target)
            return si, sl, ti, tl
        except StopIteration as e:
            raise StopIteration


class MyLoader:
    def __init__(self, seed):
        self.seed = iter(seed)

    def __len__(self):
        return len(self.seed)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            si, sl = next(self.seed)
            return si, sl
        except StopIteration as e:
            raise StopIteration


###################################
# optimizer and scheduler         #
###################################
def get_optimizers(args, models):
    optimizers = []
    lr = args.lr
    weight_decay = 5e-4
    # weight_decay = 5 * 1e-4
    momentum = 0.9
    for model in models:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
        optimizers.append(optimizer)
    return optimizers


def get_single_optimizers(args, models):
    optimizers = []
    lr = args.lr
    weight_decay = 1e-4
    # weight_decay = 5 * 1e-4
    momentum = 0.9
    optimizer = optim.SGD(models.parameters(), lr=lr, momentum=momentum,
                    weight_decay=weight_decay)
    return optimizer


def get_schedulers(args, optimizers):
    schedulers = []
    gamma = args.lr_gamma
    intervals = args.sch_intervals
    for optimizer in optimizers:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=intervals, gamma=gamma)
        schedulers.append(scheduler)
    return schedulers

def get_single_schedulers(args, optimizer):
    schedulers = []
    gamma = args.lr_gamma
    intervals = args.sch_intervals
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=intervals, gamma=gamma)
    return scheduler

# This is used for training of GAL
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., prob=.5):
        self.std = std
        self.mean = mean
        self.prob = prob
        
    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor
        else:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomLossFunction:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def softlabel_ce(self, x, t):
        b, c = x.shape
        x_log_softmax = torch.log_softmax(x, dim=1)
        if self.reduction == 'mean':
            loss = -torch.sum(t * x_log_softmax) / b
        elif self.reduction == 'sum':
            loss = -torch.sum(t * x_log_softmax)
        elif self.reduction == 'none':
            loss = -torch.sum(t * x_log_softmax, keepdims=True)
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def label_smoothing(onehot, n_classes, factor):
    return onehot * factor + (onehot - 1) * ((factor - 1) / (n_classes - 1))

import torch.nn.functional as F
class CELossWithLabelSmoothing(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth, class_num):
        self.label_smooth = label_smooth
        self.class_num = class_num
        super().__init__()

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # ×ª»»³Éone-hot

            # label smoothing
            # ÊµÏÖ 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # ÊµÏÖ 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()


class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing_alpha, num_classes):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.alpha = smoothing_alpha
        self.num_classes = num_classes

    def forward(self, input, target1, target2, model_idx=0):
        confidence = 1. - self.alpha ### * model_idx
        logprobs = nn.functional.log_softmax(input, dim=-1)
        logprobs_2 = logprobs.clone().detach()
        # rnd_target = torch.randint(self.num_classes, target1.shape).cuda()
        # if epoch > 100:
        #     _, max_idx = logprobs.max(-1)
        #     logprobs_2[:, max_idx] = 0.
        #     _, rnd_target = logprobs_2.max(-1)
        # else:
        # for i in range(model_idx):
        bs = logprobs.size(0)
        row = torch.linspace(0, bs-1, bs).long()
        # if self.num_classes == 10:
        #     # target = confidence * nn.functional.one_hot(target1, self.num_classes)
        #     # for i in range(3):
        #     #     _, max_idx = logprobs_2.max(-1)
        #     #     logprobs_2[row, max_idx] = 0.
        #     #     _, rnd_target = logprobs_2.max(-1)
        #     #     target += self.alpha / 3 * nn.functional.one_hot(rnd_target, self.num_classes)
        #     _, max_idx = logprobs_2.max(-1)
        #     logprobs_2[:, max_idx] = 0.
        #     _, max_idx = logprobs_2.max(-1)
        #     target = confidence * nn.functional.one_hot(target1, self.num_classes) + \
        #              self.alpha * nn.functional.one_hot(max_idx, self.num_classes)
        # else:
        #     rate = self.alpha / 10.
        #     target = confidence * nn.functional.one_hot(target1, self.num_classes)
        #     # logprobs_2[row, target1] = 0.
        #     # logprobs_2[row, target2] = 0.
        #     for i in range(10):
        #         _, max_idx = logprobs_2.max(-1)
        #         # target += rate * nn.functional.one_hot(max_idx, self.num_classes)
        #         logprobs_2[row, max_idx] = 0.
        #         _, rnd_target = logprobs_2.max(-1)
        #         target += rate * nn.functional.one_hot(rnd_target, self.num_classes)
        onehot = nn.functional.one_hot(target1, self.num_classes)
        target = torch.clamp(onehot.float(), min=self.alpha * model_idx / (self.num_classes - 1), max=confidence)
        ### target = torch.clamp(onehot.float(), min=self.alpha / (self.num_classes - 1), max=confidence)
        # logprobs[row, target2] = 0.
        # target = confidence * nn.functional.one_hot(target1, self.num_classes) + \
        #          self.alpha * nn.functional.one_hot(rnd_target, self.num_classes)
        loss = -1 * torch.sum(target * logprobs, 1)
        return loss.mean()
