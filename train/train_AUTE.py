import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import arguments, utils
from models.ensemble import Ensemble
from gen_adv import *
from advertorch.attacks import LinfPGDAttack
import numpy as np

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
        self.num_class = kwargs['num_class']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        self.margin = kwargs['margin']
        self.writer = writer
        self.save_root = save_root

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # PGD configs
        self.attack_cfg = {'eps': kwargs['eps'],
                           'alpha': kwargs['alpha'],
                           'steps': kwargs['steps'],
                           'is_targeted': False,
                           'rand_start': True
                           }
        self.best_epoch = 0
        self.best_robust = 0.0
        self.best_self_robust = [0.0 for i in range(len(self.models))]

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
            if (epoch % 10 == 0):
                self.save(epoch)



    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = [0 for i in range(len(self.models))]

        xcent_losses = [0 for i in range(len(self.models))]
        kl_losses = [0 for i in range(len(self.models))]

        self_correct = [0 for i in range(len(self.models))]
        other_correct = [0 for i in range(len(self.models))]

        def linear_function(x, start, end, max_beta):
            m = max_beta / (end - start)
            y = (x - start) * m
            return min(y, max_beta)

        start = 40
        end = 80
        if (epoch > start):
            beta = linear_function(epoch, start, end, self.beta)
        else:
            beta = 0
        print('beta: ', beta)

        batch_iter = self.get_batch_iterator()
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            inputs, targets = inputs.cuda(), targets.cuda()

            ensemble = Ensemble(self.models)
            adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)
            basic_weights = torch.ones((len(inputs)))
            basic_weights = basic_weights.cuda()

            for i, m in enumerate(self.models):
                outputs = m(adv_inputs)
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum()
                self_correct[i] += correct / len(inputs)
                correct_idx = predicted.eq(targets)

                probs = torch.softmax(outputs, dim=1)
                true_probs = probs[torch.arange(len(inputs)), targets]
                top2_probs, _ = torch.topk(probs, k=2, dim=1)
                top2_probs = top2_probs[:, 1]
                probs_diff = (true_probs - top2_probs) * correct_idx
                unlearn_idx = probs_diff > self.margin
                unlearning_weights = -self.gamma * i * (probs_diff) * unlearn_idx
                self_weights = basic_weights * ~unlearn_idx * (1 - probs_diff)

                weights = (self_weights + unlearning_weights)
                xcent = self.criterion(outputs, targets)

                tmp1 = torch.argsort(probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
                # margin_loss = F.nll_loss(torch.log(1.0001 - probs + 1e-12), new_y, reduction='none')
                # unlearning BCE loss
                xcent_loss = torch.mean(weights * (self.criterion(outputs, targets) +
                                                   F.nll_loss(torch.log(1.0001 - probs + 1e-12), new_y, reduction='none')))

                if (epoch > start and beta > 0.0):
                    others = []
                    for j in range(len(self.models)):
                        if (i == j):
                            continue
                        others.append(self.models[j])
                    other_ensemble = Ensemble(others)
                    t_outputs = other_ensemble(adv_inputs)
                    _, t_predicted = t_outputs.max(1)
                    other_correct[i] += t_predicted.eq(targets).sum() / len(inputs)
                    criterion_kl = nn.KLDivLoss(reduction='none')
                    kl = (1.0 / len(inputs)) * criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(t_outputs.detach(), dim=1))
                    kl = torch.sum(kl, dim=1)
                    kl_loss = beta * torch.sum((1 - true_probs) * kl)
                    loss = xcent_loss + kl_loss
                    kl_losses[i] += kl_loss.item()
                else:
                    loss = xcent_loss

                xcent_losses[i] += xcent_loss.item()
                losses[i] += loss.item()

                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

        for i in range(len(self.models)):
            self.schedulers[i].step()

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i] / len(self.trainloader)
        self.writer.add_scalars('train/loss', loss_dict, epoch)

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(
                i=i + 1, loss=losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        print_message = 'Epoch [%3d] Xcent | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {xcent_loss:.4f}  '.format(
                i=i + 1, xcent_loss=xcent_losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        print_message = 'Epoch [%3d] KL | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {kl_loss:.4f}  '.format(
                i=i + 1, kl_loss=kl_losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        print_message = 'Epoch [%3d] Self_correct | ' % epoch
        for i in range(len(self.models)):
            cur_correct = self_correct[i] / (batch_idx + 1)
            print_message += 'Model{i:d}: {self_correct:.4f}  '.format(
                i=i + 1, self_correct=cur_correct)
        tqdm.write(print_message)

        print_message = 'Epoch [%3d] Other_correct | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {other_correct:.4f}  '.format(
                i=i + 1, other_correct=other_correct[i] / (batch_idx + 1))
        tqdm.write(print_message)

    def test(self, epoch):
        with torch.no_grad():
            for m in self.models:
                m.eval()

            ensemble = Ensemble(self.models)

            loss = 0
            correct = 0
            total = 0
            adv_correct = 0
            adv_max = 1000
            adv_total = 0
            eps = 0.031
            steps = 10
            with torch.no_grad():
                for _, (inputs, targets) in enumerate(self.testloader):
                    inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = ensemble(inputs)
                    loss += torch.mean(self.criterion(outputs, targets)).item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += inputs.size(0)
                    if (adv_total < adv_max):
                        with torch.enable_grad():
                            adversary = LinfPGDAttack(
                                ensemble, loss_fn=nn.CrossEntropyLoss(), eps=eps,
                                nb_iter=steps, eps_iter=eps / 5, rand_init=True, clip_min=0., clip_max=1.,
                                targeted=False)
                            adv_inputs = adversary.perturb(inputs, targets).detach()
                            outputs = ensemble(adv_inputs)
                            _, adv_predicted = outputs.max(1)
                            adv_correct += adv_predicted.eq(targets).sum().item()
                            adv_total += inputs.size(0)

                self.writer.add_scalar('test/ensemble_loss', loss / len(self.testloader), epoch)
                self.writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

                print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
                    loss=loss / len(self.testloader), acc=correct / total)
                tqdm.write(print_message)

                adv_acc = adv_correct / adv_total
                print_message = 'Evaluation  | Ensemble Adv {eps:.3f} Acc {acc:.2%}'.format(
                    eps=eps, acc=adv_correct / adv_total)
                tqdm.write(print_message)
                if (adv_acc > self.best_robust):
                    self.best_robust = adv_acc
                    self.best_epoch = epoch

                print(self.attack_cfg)
                print('best robust acc: ', self.best_robust, '  best epoch: ', self.best_epoch)

    def save(self, epoch):
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d' % i] = m.state_dict()
        torch.save(state_dict, os.path.join(self.save_root, 'epoch_%d.pth' % epoch))
        tqdm.write(self.save_root)

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Adversarial Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.my_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    model_dir = '/nfs/AUTE/results/'
    base_dir = 'AUTE' if args.num_class == 10 else 'AUTE_cifar100'
    save_root = os.path.join(model_dir, base_dir, 'seed_{:d}'.format(args.seed),
                             '{:d}_{:s}'.format(args.model_num, args.arch),
                             'eps{:.3f}_steps{:d}'.format(args.eps, args.steps))

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

    models = utils.get_models(args, train=True, as_ensemble=False, model_file=None)

    # get data loaders
    trainloader, testloader = utils.get_loaders(args)

    # get optimizers and schedulers
    optimizers = utils.get_optimizers(args, models)
    schedulers = utils.get_schedulers(args, optimizers)

    # save the train file
    import shutil
    current_script_path = os.path.abspath(__file__)
    target_folder_path = save_root
    target_file_path = os.path.join(target_folder_path, os.path.basename(current_script_path))
    shutil.copy2(current_script_path, target_file_path)
    print("Script saved to: ", save_root)

    # train the ensemble
    trainer = Adversarial_Trainer(models, optimizers, schedulers,
                                  trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
