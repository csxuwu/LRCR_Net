

import torch
import torch.nn as nn
import os
import numpy as np
import random
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from codes.utils.img_save_v1 import save_img_for_train


class BaseTester():

    def __init__(self, cfg):
        self.cfg_train = cfg
        # self.device = torch.device('cuda' if cfg.gpu_ids is not None else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.is_train = cfg.is_train
        self.schedulers = []
        self.optimizers = None
        self.model = None

    def initialize(self,):
        pass

    def tester(self,):
        pass

    def inference(self, cur_epoch, img_ll, img_org, **kwargs):
        pass

    def tester_for_real(self, cur_epoch):
        pass

    def tester_for_pair(self, cur_epoch):
        pass

    def name(self):
        return self.cfg_train.model_name


    def load_network(self, load_path, network, strict=True):
        '''

        :param load_path:
        :param network:
        :param strict:
        :return:
        '''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def print_network(self):
        '''

        :return:
        '''
        print(self.name())
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, DistributedDataParallel):
            network = self.model.module
            print(str(network))
        else:
            print(str(self.model))

        # total num
        totoal_num_params = 0
        for p in self.model.parameters():
            totoal_num_params += p.numel()

        # trainable num
        trainable_num_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                trainable_num_params += p.numel

        print("The total number of parameters: {}\n".format(totoal_num_params))
        print("The trainable number of parameters: {}\n".format(trainable_num_params))

    def save_data(self,results, running_results, epoch, i, img_enhance, img_org, img_ll, loss_val, lr_epoch, train_bar,
				  ssim_val, psnr_val, train_data_len):
        '''
		experiment data
		:param results: dict
		:param running_results: dict
		:param epoch:
		:param i: iteration
		:param img_enhance: enhanced image
		:param img_org: ground truth
		:param img_ll: low-light image
		:param loss_val:
		:param lr_epoch:
		:param train_bar:
		:return:
		'''
        step = epoch * train_data_len + i + 1
        running_results['epoch'] = epoch
        running_results['step'] = step

        running_results['losses'] += loss_val * self.cfg_train.batch_size
        running_results['loss_mean'] = running_results['losses'] / running_results['batch_size']

        running_results['psnrs'] += psnr_val * self.cfg_train.batch_size
        running_results['psnr_mean'] = running_results['psnrs'] / running_results['batch_size']
        running_results['ssims'] += ssim_val * self.cfg_train.batch_size
        running_results['ssim_mean'] = running_results['ssims'] / running_results['batch_size']
        train_bar.set_description(desc='[Training %s on %s]  Epoch [%d/%d], Step %d， '
                                       '| Vae_loss: %.4f,  '
                                       '| Lr: %.4f,  '
                                       '| PSNR: %.4f, SSIM: %.4f\n[Training %s on %s]  '
                                       % (self.cfg_train.model_name,
                                          self.cfg_train.train_dataset_name + '_' + str(self.cfg_train.patch_size),
                                          epoch, self.cfg_train.num_epochs, step, loss_val,
                                          lr_epoch,
                                          psnr_val, ssim_val, self.cfg_train.model_name,
                                          self.cfg_train.train_dataset_name + '_' + str(self.cfg_train.patch_size)))

        if (i + 1) % 1000 == 0 or self.cfg_train.debug :

            save_img_for_train(model_name=self.cfg_train.model_name,
                               imgs=[img_org, img_ll, img_enhance],
                               out_path=self.cfg_train.train_out_path,
                               step=step, epoch=epoch)

            # losses, psnr，ssim
            results['epoch'].append(epoch)
            results['step'].append(step)
            results['losses'].append(running_results['loss_mean'])
            results['lr_epoch'].append(lr_epoch)
            results['psnr'].append(running_results['psnr_mean'])
            results['ssim'].append(running_results['ssim_mean'])






