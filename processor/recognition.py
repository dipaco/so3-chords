#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from tools import NTU_CLASS_NAMES

def process_class_names(class_names):
    out_class_names = []
    for name in class_names:
        parts = name.split(' ')
        lines = [parts[0]]
        c_line = 0
        th = 20
        for p in parts[1:]:
            if len(f'{lines[c_line]} {p}') > th:
                lines[c_line] = f'{lines[c_line]}\n'
                lines.append(p)
                c_line += 1

            else:
                lines[c_line] = f'{lines[c_line]} {p}'
        out_class_names.append(''.join(lines))
    return out_class_names


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        ds_mode = self.arg.test_feeder_args['aug_mod']
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.dump_variables(self.tables_dir / f'{ds_mode}_results.txt', {f'top{k}': f'{accuracy}'}, append=True)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_confusion_matrix(self):

        ds_mode = self.arg.test_feeder_args['aug_mod']
        probs = np.exp(self.result) / np.exp(self.result).sum(axis=-1, keepdims=True)
        n_classes = probs.shape[-1]

        # Computes the confusion matrix containing average predicted probability per class given the gt class?
        conf_matrix = np.stack([probs[self.label == i].mean(axis=0) for i in range(n_classes)])

        # Computes the probability similarity matrix given the gt class
        probs_dist = (probs[np.arange(probs.shape[0]), self.label][:, None] - probs)
        conf_matrix_dist = np.stack([probs_dist[self.label == i].mean(axis=0) for i in range(n_classes)])

        fig = plt.figure()
        plt.imshow(conf_matrix, vmin=0.0, vmax=1.0, interpolation=None, cmap='coolwarm')
        plt.colorbar()
        plt.savefig(self.figures_dir / f'{ds_mode}_cm.pdf', bbox_inches="tight")
        plt.close(fig)

        plt.figure()
        plt.imshow(conf_matrix_dist, vmin=0.0, vmax=1.0, interpolation=None, cmap='coolwarm')
        plt.colorbar()
        plt.savefig(self.figures_dir / f'{ds_mode}_cm_dist.pdf', bbox_inches="tight")
        plt.close(fig)

        n_rows = n_classes // 10
        n_cols = 10
        diag_conf_matrix = np.diag(conf_matrix)[None].reshape(n_rows, n_cols)
        fig = plt.figure(figsize=(20, 10))
        plt.subplots_adjust(
            left=0.125,  # the left side of the subplots of the figure
            right=0.9,  # the right side of the subplots of the figure
            bottom=0.1,  # the bottom of the subplots of the figure
            top=0.9,  # the top of the subplots of the figure
            wspace=0.2,  # the amount of width reserved for blank space between subplots
            hspace=0.2,  # the amount of height reserved for white space between subplots
        )
        for i in range(n_rows):
            ax = plt.subplot(1, n_rows, i + 1)
            plt.imshow(diag_conf_matrix[i, :, None], vmin=0.0, vmax=1.0, interpolation=None, cmap='coolwarm')

            for j in range(n_cols):
                color = (1.0, 1.0, 1.0, 1.0)
                ax.text(0, j, f'{diag_conf_matrix[i, j]:.3f}', ha="center", va="center", color=color)

            ax.tick_params(axis='both', which='minor', labelsize=6)
            plt.yticks(np.arange(n_cols), process_class_names(NTU_CLASS_NAMES[i*n_cols:(i+1)*n_cols]), rotation=0)
            plt.xticks([0.5], [''])
            ax.xaxis.tick_top()
        #fig.tight_layout(pad=5.0)

        #plt.colorbar()
        plt.savefig(self.figures_dir / f'{ds_mode}_cm_diag.pdf', bbox_inches="tight")

        self.dump_variables(
            self.tables_dir / f'{ds_mode}_results.txt',
            {
                'conf_probs': diag_conf_matrix.tolist(),
            },
            append=False,
        )
        self.io.print_log(f'\tconf_probs: {[[f"{e:.3f}" for  e in r] for r in diag_conf_matrix.tolist()]}')

    def dump_variables(self, file, var_dict, append=False):

        mode = 'a' if append else 'w'
        with open(file, mode) as f:
            for k, v in var_dict.items():
                f.write(f'#{k}\n')
                f.write(f'{v}\n')

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        # add logs in tensor board
        self.tb_writer.add_scalar('train/loss', np.mean(loss_value), self.meta_info['iter'])
        self.tb_writer.add_scalar('train/lr', self.lr, self.meta_info['iter'])

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show tensorboard data
            self.tb_writer.add_scalar('test/loss', np.mean(loss_value), self.meta_info['iter'])

            # show confusion matrix
            self.show_confusion_matrix()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        parser.add_argument('--aug_mod', default='no_aug', help='Dataset augmentation mode. Apply transformtions to the xyz joint coordinates.', choices=['no_aug', 'azimuthal', 'so3'])
        parser.add_argument('--feat_mode', default='feat_mode', help='Type of features computed from the xyz joint coordinates.', choices=['xyz', 'so3_chains'])

        return parser
