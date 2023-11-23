import os
import torch
import pandas as pd
import random
import numpy as np
import time

from torch import optim, nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from codes.utils.ops import create_folder, numerical_normalizaiton
from codes.utils.img_save import save_img_for_train, save_img_for_test
from codes.utils.build_LRCR import Build_LRCR as Build_Model
from codes.trainers.base_trainer import BaseTrainer
from codes.utils.img_save_v1 import *
from codes.utils import psnr, ssim, pytorch_msssim
from codes.utils import init_model


# -----------------------------------------------------
# trainer for LRCR_Net
# -----------------------------------------------------

class Trainer(BaseTrainer):
	def __init__(self, cfg, train_loader=None, test_loader=None):
		BaseTrainer.__init__(self, cfg=cfg)
		self.cfg = cfg
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.train_data_len = train_loader.__len__()
		self.old_lr = cfg.lr

		self.test_dataset_name_syn = 'NASA'
		self.test_dataset_name_syn2 = 'AGLLSet_test'

		# build network
		self.build_net = Build_Model(cfg=cfg)
		self.model = self.build_net.model
		self.initialize()

	def initialize(self):
		'''
		some thing used for the training step.
		:return:
		'''
		self.optimizers = optim.Adam(list(self.model.parameters()), self.cfg.lr, [self.cfg.beta1, self.cfg.beta2])
		self.scheduler_lr = optim.lr_scheduler.ExponentialLR(self.optimizers, gamma=self.cfg.lr_decay)   # 指数衰减学习率
		self.loss = self.build_net.loss
		self.model.apply(init_model.weights_init_kaiming)
		self.writer = SummaryWriter(self.cfg.train_log_tb_path)
		self.L2 = nn.MSELoss()

		if self.cfg.load_model_path != '':
			self.load_network(load_path=self.cfg.load_model_path, network=self.model)
			print('%s is Successfully Loaded from %s' % (self.cfg.model_type, self.cfg.load_model_path))

		self.cur_lr = self.cfg.lr
		self.results = {'epoch': [], 'step': [], 'losses': [], 'lr_epoch': [], 'psnr': [], 'ssim': []}
		self.test_results = {'epoch': [],  'psnr': [], 'ssim': [], 'msssim': [], 'mse': [], 'ave_run_time':[]}
		self.img_id = 0
		self.flag = True

	def trainer(self):
		'''
		trainer
		:return:
		'''
		# testing
		global_step = 0

		if self.cfg.test and self.cfg.load_model_path != '':
			epoch = 'test'
			self.model.eval()
			self.tester_for_epoch(epoch)
			return

		for epoch in range(self.cfg.start_epoch, self.cfg.num_epochs):
			self.model.train(True)
			train_bar = tqdm(self.train_loader)
			running_results = {'epoch': 0, 'step': 0, 'loss_mean': 0, 'losses': 0,
							   'lr_epoch': 0, 'psnr_mean': 0, 'psnrs': 0, 'ssim_mean': 0, 'ssims': 0, 'batch_size': 0}

			if self.cfg.debug:
				self.save_model(epoch, self.cur_lr)

				l = len(self.results['losses']) + 1
				data_frame = pd.DataFrame(data={'epoch': self.results['epoch'],
												'step': self.results['step'],
												'losses': self.results['losses'],
												'lr_epoch': self.results['lr_epoch'],
												'psnr': self.results['psnr'],
												'ssim': self.results['ssim']},
										  index=range(1, l))
				data_save_path = os.path.join(self.cfg.train_log_path,
											  'train_' + self.cfg.model_name + '_' + self.cfg.train_dataset_name + '_' + str(
												  self.cfg.img_h) + '.csv')
				data_frame.to_csv(data_save_path)
				self.tester_for_epoch(epoch)

			for i, (img_ll, img_ll_noise, img_org, img_org_en, y, noise_map, name) in enumerate(train_bar):
				global_step += 1
				running_results['batch_size'] += self.cfg.batch_size
				img_aug_list = self.data_augm(img_list=[img_ll, img_ll_noise, img_org, y, noise_map, img_org_en],
											  image_id=self.img_id)
				img_ll = img_aug_list[0]
				img_ll_noise = img_aug_list[1]
				img_org = img_aug_list[2]
				y = img_aug_list[3]
				noise_map = img_aug_list[4]
				img_org_en = img_aug_list[5]

				if self.cfg.start_epoch != 0 and self.flag:
					self.load_network(load_path=self.cfg.load_model_path, network=self.model)
					self.tester_for_epoch(epoch - 1)
					self.flag = False

				loss, img_enhance, ssim_val, psnr_val = self.inference(epoch,
																	   img_ll=img_ll,
																	   img_ll_noise=img_ll_noise,
																	   img_org=img_org,
																	   img_org_en=img_org_en,
																	   y=y,
																	   noise_map=noise_map,
																	   global_step=global_step)
				loss_val = self.optimize_parameters(loss)

				self.save_data(results=self.results,
							   running_results=running_results,
							   epoch=epoch,
							   i=i,
							   train_bar=train_bar,
							   img_org=img_org,
							   img_ll=img_ll,
							   img_enhance=img_enhance,
							   lr_epoch=self.cur_lr,
							   loss_val=loss_val,
							   ssim_val=ssim_val,
							   psnr_val=psnr_val,
							   train_data_len=self.train_data_len)

			self.save_model(epoch, self.cur_lr)

			l = len(self.results['losses']) + 1
			data_frame = pd.DataFrame(data={'epoch': self.results['epoch'],
											'step': self.results['step'],
											'losses': self.results['losses'],
											'lr_epoch': self.results['lr_epoch'],
											'psnr': self.results['psnr'],
											'ssim': self.results['ssim']},
									  index=range(1, l))
			data_save_path = os.path.join(self.cfg.train_log_path,
										  'train_' + self.cfg.model_name + '_' + self.cfg.train_dataset_name + '_' + str(
											  self.cfg.img_h) + '.csv')
			data_frame.to_csv(data_save_path)

			self.scheduler_lr.step(epoch)
			self.model.eval()
			self.tester_for_epoch(epoch)

	def inference(self, cur_epoch, img_ll,img_ll_noise, img_org, img_org_en, y, noise_map,global_step):

		img_ll = Variable(img_ll)
		img_ll = img_ll.to(self.device)
		img_ll_noise = Variable(img_ll_noise)
		img_ll_noise = img_ll_noise.to(self.device)
		img_org = Variable(img_org)
		img_org = img_org.to(self.device)
		img_org_en = Variable(img_org_en)
		img_org_en = img_org_en.to(self.device)
		y = Variable(y)
		y = y.to(self.device)

		out, out2 = self.model({'img_ll_noise':img_ll, 'illu_map':y, 'img_org_en': img_org_en, 'is_test':False})
		out['img_ll'] = img_ll
		GT={'img_org':img_org,
			'img_org_en':img_org_en}

		loss = self.loss(input=out, GT=GT)

		ssim_val = ssim.ssim(out['img_enhance1'], img_org).item()
		psnr_val = psnr.psnr(out['img_enhance1'], img_org)

		ssim_val2 = ssim.ssim(out['img_enhance2'], img_org).item()
		psnr_val2 = psnr.psnr(out['img_enhance2'], img_org)

		if global_step % 10 == 0:
			self.writer.add_scalar(tag='ssim_train1', scalar_value=ssim_val, global_step=global_step)
			self.writer.add_scalar(tag='psnr_train1', scalar_value=psnr_val, global_step=global_step)
			self.writer.add_scalar(tag='ssim_train2', scalar_value=ssim_val2, global_step=global_step)
			self.writer.add_scalar(tag='psnr_train2', scalar_value=psnr_val2, global_step=global_step)
			for i in range(len(loss)):
				if i == 0:
					tag = self.cfg.loss_type + '/loss_sum'
					self.writer.add_scalar(tag=tag, scalar_value=loss[i], global_step=global_step)
				else:
					tag = self.cfg.loss_type + '/losses' + str(i)
					self.writer.add_scalar(tag=tag, scalar_value=loss[i], global_step=global_step)

		return loss[0], out['img_enhance2'], ssim_val, psnr_val


	def tester_for_epoch(self, cur_epoch):
		'''

		:param cur_epoch:
		:return:
		'''
		torch.cuda.empty_cache()
		test_out_path_real = os.path.join(self.cfg.test_out_path, self.cfg.test_dataset_name, 'epoch_' + str(cur_epoch))
		test_out_path_syn = os.path.join(self.cfg.test_out_path, self.test_dataset_name_syn, 'epoch_' + str(cur_epoch))
		test_out_path_syn2 = os.path.join(self.cfg.test_out_path, self.test_dataset_name_syn2, 'epoch_' + str(cur_epoch))
		create_folder(test_out_path_real)
		create_folder(test_out_path_syn)

		if cur_epoch == 9:
			test_out_path_syn2 = os.path.join(self.cfg.test_out_path,
											  self.test_dataset_name_syn2,
											  'epoch_' + str(cur_epoch))
			self.tester_for_pair(out_path=test_out_path_syn2, cur_epoch=cur_epoch, test_loader=self.test_loader['AGLLSet_test'])

		self.tester_for_pair(out_path=test_out_path_syn, cur_epoch=cur_epoch, test_loader=self.test_loader['NASA'], is_save_dict=True)

		with torch.no_grad():
			test_bar_real = tqdm(self.test_loader['real'])
			test_bar_real.set_description(desc='[Testing real %s]' % self.cfg.model_name)
			for j, (img_ll_test, y, name) in enumerate(test_bar_real):
				img_ll_test = Variable(img_ll_test)
				img_ll_test = img_ll_test.to(self.device)
				y = Variable(y)
				y = y.to(self.device)
				img_name = name[0] + '_' + self.cfg.model_name

				try:
					out, out2  = self.model({'img_ll_noise': img_ll_test, 'illu_map': y, 'is_test': True})

					test_out_path_real2 = test_out_path_real + '2'
					create_folder(test_out_path_real2)
					save_img_for_test(model_name_son=self.cfg.model_name, imgs=[out['img_enhance1']],
									  out_path=test_out_path_real2, index=j, epoch=cur_epoch, img_name=img_name)
					save_img_for_test(model_name_son=self.cfg.model_name,
									  imgs=[out['img_enhance2']],
									  out_path=test_out_path_real2, index=j, epoch=cur_epoch, img_name=img_name + '_2')

				except RuntimeError as exception:
					if "out of memory" in str(exception) or "illegal memory access" in str(exception):
						print("WARNING: out of memory")
						if hasattr(torch.cuda, 'empty_cache'):
							torch.cuda.empty_cache()
					else:
						raise exception

				for key in out:
					tag = 'test/' + key
					if out[key] is not None and len(out[key].size()) == 4:
						if key == 'slice_coeffs':
							self.writer.add_image(tag='test/slice_coeffs_r',
												  img_tensor=out["slice_coeffs"][:, 0:3, :, :],
												  dataformats='NCHW')
							self.writer.add_image(tag='test/slice_coeffs_g',
												  img_tensor=out["slice_coeffs"][:, 4:7, :, :],
												  dataformats='NCHW')
							self.writer.add_image(tag='test/slice_coeffs_b',
												  img_tensor=out["slice_coeffs"][:, 8:11, :, :],
												  dataformats='NCHW')
						elif key != 'coeffs_out':
							self.writer.add_image(tag, out[key], dataformats='NCHW')
							if cur_epoch > 4:
								save_img_for_test(model_name_son=self.cfg.model_name,
												  imgs=out[key],
												  out_path=test_out_path_real, index=j, epoch=cur_epoch, img_name=img_name +'_' +key)
					elif out[key] is not None and len(out[key].size()) == 3:
						self.writer.add_image(tag, out[key], dataformats='CHW')


	def tester_for_pair(self, out_path, cur_epoch, test_loader, is_save_dict=False):
		'''
		paired testset. low-light image -- normal-light image
		:param out_path:
		:param cur_epoch:
		:param test_loader:
		:param is_save_dict:
		:return:
		'''

		results = {'step': [],
				   'ssim': [], 'msssim': [], 'psnr': [], 'mse_val': [],
				   'ssim2': [], 'msssim2': [], 'psnr2': [], 'mse_val2': []}
		test_out_path = out_path
		test_log_path = os.path.join(self.cfg.test_log_path,
										 'test_' + '_' + self.test_dataset_name_syn + '_' + str(cur_epoch) + '.csv')
		test_avg_log_path = os.path.join(self.cfg.test_log_path,
									 'test_AVG_' + '_' + self.test_dataset_name_syn + '_' + str(cur_epoch) + '.csv')
		create_folder(test_out_path)
		total_run_time = 0.0

		ssim_sum = 0.0
		msssim_sum = 0.0
		psnr_sum = 0.0
		mse_sum = 0.0

		ssim_sum2 = 0.0
		msssim_sum2 = 0.0
		psnr_sum2 = 0.0
		mse_sum2 = 0.0

		with torch.no_grad():
			test_bar = tqdm(test_loader)
			test_data_len = test_loader.__len__()
			print('test imgs:{}'.format(test_data_len))

			for step, (img_ll_test, img_org, y, img_name) in enumerate(test_bar):
				img_ll_test = Variable(img_ll_test)
				img_ll_test = img_ll_test.to(self.device)
				img_org = Variable(img_org)
				img_org = img_org.to(self.device)
				img_org = torch.tensor(img_org, dtype=torch.float32)
				y = Variable(y)
				y = y.to(self.device)
				img_name = img_name[0] + '_' + self.cfg.model_name

				try:
					st = time.time()
					out, out2 = self.model({'img_ll_noise': img_ll_test, 'illu_map': y, 'is_test': True})
					total_run_time += time.time() - st
				except RuntimeError as exception:
					if "out of memory" in str(exception) or "illegal memory access" in str(exception):
						print("WARNING: out of memory")
						if hasattr(torch.cuda, 'empty_cache'):
							torch.cuda.empty_cache()
					else:
						raise exception

				ssim_val = pytorch_msssim.ssim(out['img_enhance1'], img_org).item()
				msssim_val = pytorch_msssim.msssim(out['img_enhance1'], img_org).item()
				psnr_val = psnr.psnr(out['img_enhance1'], img_org)
				mseloss = nn.MSELoss()
				mse_val = mseloss(out['img_enhance1'], img_org).item()

				ssim_val2 = pytorch_msssim.ssim(out['img_enhance2'], img_org).item()
				msssim_val2 = pytorch_msssim.msssim(out['img_enhance2'], img_org).item()
				psnr_val2 = psnr.psnr(out['img_enhance2'], img_org)
				mse_val2 = mseloss(out['img_enhance2'], img_org).item()

				test_out_path2 = test_out_path + '2'
				create_folder(test_out_path2)
				save_img_for_test(model_name_son=self.cfg.model_name,
								  imgs=[out['img_enhance1']],
								  out_path=test_out_path2, index=step, epoch=cur_epoch, img_name=img_name)
				save_img_for_test(model_name_son=self.cfg.model_name,
								  imgs=[out['img_enhance2']],
								  out_path=test_out_path2, index=step, epoch=cur_epoch, img_name=img_name + '_2')

				self.writer.add_scalar('test/ssim_test', scalar_value=ssim_val, global_step=step)
				self.writer.add_scalar('test/msssim_test', scalar_value=msssim_val, global_step=step)
				self.writer.add_scalar('test/psnr_test', scalar_value=psnr_val, global_step=step)
				self.writer.add_scalar('test/mse_test', scalar_value=mse_val, global_step=step)

				self.writer.add_scalar('test/ssim_test2', scalar_value=ssim_val2, global_step=step)
				self.writer.add_scalar('test/msssim_test2', scalar_value=msssim_val2, global_step=step)
				self.writer.add_scalar('test/psnr_test2', scalar_value=psnr_val2, global_step=step)
				self.writer.add_scalar('test/mse_test2', scalar_value=mse_val2, global_step=step)

				for key in out:
					tag = 'test/' + key + '_p'
					if out[key] is not None and len(out[key].size()) == 4:
						if key == 'slice_coeffs':
							self.writer.add_image(tag='test/slice_coeffs_r_p',
												  img_tensor=out["slice_coeffs"][:, 0:3, :, :],
												  dataformats='NCHW')
							self.writer.add_image(tag='test/slice_coeffs_g_p',
												  img_tensor=out["slice_coeffs"][:, 4:7, :, :],
												  dataformats='NCHW')
							self.writer.add_image(tag='test/slice_coeffs_b_p',
												  img_tensor=out["slice_coeffs"][:, 8:11, :, :],
												  dataformats='NCHW')
						elif key != 'coeffs_out':
							self.writer.add_image(tag, out[key], dataformats='NCHW')
							if is_save_dict and cur_epoch > 4:
								save_img_for_test(model_name_son=self.cfg.model_name,
												  imgs=out[key],
												  out_path=test_out_path, index=step, epoch=cur_epoch, img_name=img_name+'_'+key)
					elif out[key] is not None and len(out[key].size()) == 3:
						self.writer.add_image(tag, out[key], dataformats='CHW')

				results['step'].append(step)
				results['psnr'].append(psnr_val)
				results['ssim'].append(ssim_val)
				results['msssim'].append(msssim_val)
				results['mse_val'].append(mse_val)

				results['psnr2'].append(psnr_val2)
				results['ssim2'].append(ssim_val2)
				results['msssim2'].append(msssim_val2)
				results['mse_val2'].append(mse_val2)

				ssim_sum += ssim_val
				msssim_sum += msssim_val
				psnr_sum += psnr_val
				mse_sum += mse_val

				ssim_sum2 += ssim_val2
				msssim_sum2 += msssim_val2
				psnr_sum2 += psnr_val2
				mse_sum2 += mse_val2

				test_bar.set_description(desc='[Testing %s]  Step [%d/%d], '
											  '| PSNR: %.4f, SSIM: %.4f, MSSSIM: %.4f, MSE: %.4f, '
											  '| PSNR2: %.4f, SSIM2: %.4f, MSSSIM2: %.4f, MSE2: %.4f'
											  '\n[Testing %s]  '
											  % (self.cfg.model_name, test_data_len, step,
												 psnr_val, ssim_val, msssim_val, mse_val,
												 psnr_val2, ssim_val2, msssim_val2, mse_val2,
												 self.cfg.model_name))

			l = len(results['psnr']) + 1
			data_frame = pd.DataFrame(data={'step': results['step'],
											'ssim': results['ssim'],
											'msssim': results['msssim'],
											'psnr': results['psnr'],
											'mse_val': results['mse_val2'],

											'ssim2': results['ssim2'],
											'msssim2': results['msssim2'],
											'psnr2': results['psnr2'],
											'mse_val2': results['mse_val2']
											},
									  index=range(1, l))
			data_frame.to_csv(test_log_path)

			ave_run_time = total_run_time / float(self.test_loader.__len__())
			print("[*] Average run time: %.4f" % ave_run_time)
			print("[*] Average PSNR: %f, SSIM: %f, MSSSIM: %f, MSE: %f " % (
				psnr_sum2 / float(test_data_len),
				ssim_sum2 / float(test_data_len),
				msssim_sum2 / float(test_data_len),
				mse_sum2 / float(test_data_len)))

			self.test_results['epoch'].append(cur_epoch)
			self.test_results['psnr'].append(psnr_sum2 / float(test_data_len))
			self.test_results['ssim'].append(ssim_sum2 / float(test_data_len))
			self.test_results['msssim'].append(msssim_sum2 / float(test_data_len))
			self.test_results['mse'].append(mse_sum2 / float(test_data_len))
			self.test_results['ave_run_time'].append(ave_run_time)

			l2 = len(self.test_results['epoch']) + 1
			data_frame_avg = pd.DataFrame(data={'epoch': self.test_results['epoch'],
												'ssim': self.test_results['ssim'],
												'msssim': self.test_results['msssim'],
												'psnr': self.test_results['psnr'],
												'mse': self.test_results['mse'],
												'ave_run_time': self.test_results['ave_run_time']},
										  index=range(1, l2))
			data_frame_avg.to_csv(test_avg_log_path)
			print('=' * 50)
			print('All data have saved!')






































