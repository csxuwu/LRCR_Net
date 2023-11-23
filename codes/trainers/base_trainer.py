

import torch
import torch.nn as nn
import os
import numpy as np
import random
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from codes.utils.img_save_v1 import save_img_for_train
from codes.utils.data_augmentation import data_augmentation

from torch.autograd import Variable


class BaseTrainer():

    def __init__(self, cfg):
        self.cfg_train = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_train = cfg.is_train
        self.schedulers = []
        self.optimizers = None
        self.model = None

    def initialize(self,):
        pass

    def trainer(self):
        pass

    def inference(self, cur_epoch, img_ll, img_org, **kwargs):
        pass

    def inference_pro(self, cur_epoch, img_ll, **kwargs):
        pass

    def tester_for_epoch(self, cur_epoch,**kwargs):
        pass

    def tester_for_pair(self, out_path, cur_epoch,**kwargs):
        pass

    def name(self):
        return self.cfg_train.model_name

    def optimize_parameters(self, loss):
        '''

        :return:
        '''
        loss_val = loss.mean().item()
        self.model.zero_grad()
        loss.backward()
        self.optimizers.step()
        return loss_val

    def _set_lr(self, lr_gourps_l):
        '''

        :param lr_gourps_l:
        :return:
        '''
        for optimizer, lr_gourps in zip(self.optimizers, lr_gourps_l):
            for param_group, lr in zip(optimizer.param_groups, lr_gourps):
                param_group['lr'] = lr


    def update_learning_rate(self, cur_epoch):
        '''

        :param cur_epoch:
        :param lr_decay:
        :return:
        '''
        init_lr_g_l = self._get_init_lr()
        new_lr_l = []
        for init_lr_g in init_lr_g_l:
            new_lr_l.append([v * pow(self.cfg_train.lr_decay, (cur_epoch + 1)) for v in init_lr_g])
        self._set_lr(new_lr_l)

    def _get_init_lr(self):
        '''

        '''
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])

        return init_lr_groups_l

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

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
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''

        :param epoch:
        :param iter_step:
        :return:
        '''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.cfg_train.train_out_path, save_filename)
        torch.save(state, save_path)

    def save_model(self, cur_epoch, cur_lr):

        net_path = os.path.join(self.cfg_train.train_log_path, '%s-ep%d-lr%.4f-lrdec%d.pth' % (
            self.cfg_train.model_name, cur_epoch, cur_lr, self.cfg_train.num_step_decay))
        opt_file = self.cfg_train.model_name + '_optimizer_ep' + repr(cur_epoch) + '.pth'
        opt_path = os.path.join(self.cfg_train.train_log_path, opt_file)

        print('Saving state, iter:', cur_epoch)

        if torch.__version__ >= '1.6.0':
            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(),net_path, _use_new_zipfile_serialization=False)
            else:
                torch.save(self.model.state_dict(), net_path, _use_new_zipfile_serialization=False)
            torch.save(self.optimizers.state_dict(),opt_path, _use_new_zipfile_serialization=False)
        else:
            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(),net_path)
            else:
                torch.save(self.model.state_dict(),net_path)
            torch.save(self.optimizers.state_dict(),opt_path)
        print('*' * 20)
        print('%s model ,epoch %d has saved.' % (self.cfg_train.model_type, cur_epoch))
        print('*' * 20)

    def resume_training(self, resume_state):

        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

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

		:param results: dict
		:param running_results: dict
		:param epoch:
		:param i: iteration
		:param img_enhance:
		:param img_org:
		:param img_ll:
		:param loss_val:
		:param lr_epoch:
		:param train_bar:
		:return:
		'''
        step = epoch * train_data_len + i + 1  #
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

            results['epoch'].append(epoch)
            results['step'].append(step)
            results['losses'].append(running_results['loss_mean'])
            results['lr_epoch'].append(lr_epoch)
            results['psnr'].append(running_results['psnr_mean'])
            results['ssim'].append(running_results['ssim_mean'])


    def data_augm(self, img_list, image_id):
        '''

        :param img_ll: low-light image
        :param image_id:
        :return:
        '''

        batch_list = []
        tp = img_list[0]
        _, h, w = tp[0].shape
        x = random.randint(0, h - self.cfg_train.patch_size)
        y = random.randint(0, w - self.cfg_train.patch_size)
        rand_mode = random.randint(0, 7)

        for i in range(len(img_list)):
            img = img_list[i]
            batch_container = batch_input_low = np.zeros((img.size(0), self.cfg_train.patch_size,
                                                          self.cfg_train.patch_size, 3), dtype="float32")
            if len(img.shape) == 3:
                img = img.unsqueeze(dim=1)
                batch_container = batch_input_low = np.zeros((img.size(0), self.cfg_train.patch_size,
                                                              self.cfg_train.patch_size, 1), dtype="float32")
            img = np.transpose(img, [0, 2, 3, 1])

            for patch_id in range(img.size(0)):
                img_aug = data_augmentation(
                    image=img[patch_id][x: x + self.cfg_train.patch_size, y: y + self.cfg_train.patch_size, :],
                    mode=rand_mode)
                batch_container[patch_id, :, :, :] = img_aug
            batch_list.append(torch.from_numpy(batch_container.transpose([0, 3, 1, 2])))

        return batch_list


    def merge_multigpu(self, tensor):
        if tensor is not None:
            if len(tensor.shape) == 0:
                return tensor
            return tensor.sum() / len(tensor)
        else:
            return None




