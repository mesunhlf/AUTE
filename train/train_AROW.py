import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import arguments, utils
from models.ensemble import Ensemble
from gen_adv import *
from advertorch.attacks import LinfPGDAttack
from attacks.pgd import PGD_Linf, GA_PGD

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Adversarial_Trainer():
    def __init__(self, models, optimizers, schedulers,
                 trainloader, testloader,
                 writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.trainloader = trainloader
        self.testloader = testloader
        self.gamma = kwargs['gamma']
        self.writer = writer
        self.save_root = save_root

        self.criterion = nn.CrossEntropyLoss()
        self.best_epoch = 0
        self.best_robust = 0.0
        # PGD configs
        self.attack_cfg = {'eps': kwargs['eps'],
                           'alpha': kwargs['alpha'],
                           'steps': kwargs['steps'],
                           'is_targeted': False,
                           'rand_start': True
                           }

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs + 1)), total=self.epochs, desc='Epoch',
                        leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            if(epoch % 10 == 0):
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = [0 for i in range(len(self.models))]
        sup_losses = [0 for i in range(len(self.models))]
        reg_losses = [0 for i in range(len(self.models))]

        batch_iter = self.get_batch_iterator()
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            inputs, targets = inputs.cuda(), targets.cuda()

            ensemble = Ensemble(self.models)
            adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            for i, m in enumerate(self.models):
                loss = 0

                LS_loss = LabelSmoothingCrossEntropy(0.2)
                outputs = m(inputs)
                adv_outputs = m(adv_inputs)
                adv_probs = F.softmax(adv_outputs, dim=1)
                nat_probs = F.softmax(outputs, dim=1)
                true_probs = torch.gather(adv_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
                sup_loss = LS_loss(outputs, targets)
                rob_loss = (F.kl_div((adv_probs + 1e-12).log(), nat_probs, reduction='none').sum(dim=1) * (
                            1. - true_probs)).mean()
                reg_loss = self.gamma * rob_loss
                loss = sup_loss + reg_loss
                losses[i] += loss.item()
                reg_losses[i] += reg_loss.item()
                sup_losses[i] += sup_loss.item()

                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model loss {i:d}: {loss:.4f}  '.format(
                i=i + 1, loss=losses[i] / (batch_idx + 1))

        for i in range(len(self.models)):
            print_message += 'Model reg_loss {i:d}: {reg_loss:.4f}  '.format(
                i=i + 1, reg_loss=reg_losses[i] / (batch_idx + 1))

        for i in range(len(self.models)):
            print_message += 'Model sup_loss {i:d}: {sup_loss:.4f}  '.format(
                i=i + 1, sup_loss=sup_losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        for i in range(len(self.models)):
            self.schedulers[i].step()

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i] / len(self.trainloader)
        self.writer.add_scalars('train/adv_loss', loss_dict, epoch)


    def test(self, epoch):
        for m in self.models:
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = ensemble(inputs)
                loss += torch.mean(self.criterion(outputs, targets)).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/ensemble_loss', loss / len(self.testloader), epoch)
        self.writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

        print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
            loss=loss / len(self.testloader), acc=correct / total)
        tqdm.write(print_message)


    def save(self, epoch):
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d' % i] = m.state_dict()
        torch.save(state_dict, os.path.join(self.save_root, 'epoch_%d.pth' % epoch))
        print(self.save_root)


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Adversarial Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.adv_train_args(parser)
    arguments.arow_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    base_dir = 'AROW' if args.num_class == 10 else 'AROW_cifar100'
    save_root = os.path.join('checkpoints', base_dir, 'seed_{:d}'.format(args.seed),
                             '{:d}_{:s}{:d}_eps_{:.3f}'.format(args.model_num, args.arch, args.depth, args.eps))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # To speed up training
    torch.backends.cudnn.benchmark = True

    # initialize models
    models = utils.get_models(args, train=True, as_ensemble=False, model_file=None)

    # get data loaders
    trainloader, testloader = utils.get_loaders(args)

    # get optimizers and schedulers
    optimizers = utils.get_optimizers(args, models)
    schedulers = utils.get_schedulers(args, optimizers)

    # train the ensemble
    trainer = Adversarial_Trainer(models, optimizers, schedulers,
                                  trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
