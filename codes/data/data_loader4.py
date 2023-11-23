

import os
import glob
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.distributed import DistributedSampler
from codes.utils import img_processing
from codes.data import data_utils
import math
import numpy as np

class Load_Data(Dataset):
	'''
	读取图片
	获取低照度图像的Y通道
	读取低照度图像、低照度+噪声图像、正常照度图像
	'''
	def __init__(self, data_root, data_son=None, img_type='jpg', is_resize=False, is_long_resize=False, resize_h=512, resize_w=512):
		if data_son is not '':
			# 如果非None，则读取成对的低照度-正常照度图像
			imgs_ll = glob.glob(os.path.join(data_root, data_son['ll'], '*.*' ))
			imgs_ll_noise = glob.glob(os.path.join(data_root, data_son['ll_noise'], '*.*' ))
			imgs_org = glob.glob(os.path.join(data_root, data_son['org'], '*.*'))
			imgs_org_enhance = glob.glob(os.path.join(data_root, data_son['org_en'], '*.*'))
			self.imgs_ll = imgs_ll
			self.imgs_org = imgs_org
			self.imgs_org_enhance = imgs_org_enhance
		else:
			imgs_ll_noise = glob.glob(os.path.join(data_root, '*.*'))

		self.imgs_ll_noise = imgs_ll_noise
		self.data_son = data_son
		self.is_resize = is_resize
		self.resize_h = resize_h
		self.resize_w = resize_w
		self.is_long_resize = is_long_resize

		# 对图片的操作
		self.img_ll_transform = data_utils.train_ll_transforms()
		self.img_org_transform = data_utils.train_org_transforms()


	def __getitem__(self, index):
		'''
		读取图片，并对图片进行相应的处理
		:param index:
		:return:
		'''
		imgs_ll_path = self.imgs_ll[index]      # 低照度，返回下标为index的低照度图片路径
		imgs_ll_noise_path = imgs_ll_path.replace(self.data_son['ll'], self.data_son['ll_noise'])	# 低照度 + noise

		[_, name] = os.path.split(imgs_ll_path)
		suffix = name[name.find('.') + 1:]  # 图片类型
		name = name[:name.find('.')]

		img_ll = img_processing.read_image(imgs_ll_path, is_resize=self.is_resize, resize_height=self.resize_h,
		                                   resize_width=self.resize_w, normalization=True,
										   is_long_resize=self.is_long_resize)
		img_ll_noise, y = img_processing.read_image(imgs_ll_noise_path, is_resize=self.is_resize, resize_height=self.resize_h,
											  resize_width=self.resize_w, normalization=True,
											  is_long_resize=self.is_long_resize, is_cvtColor='YCrCb')
		# t0 = abs(img_ll_noise - img_ll)
		# t = abs(img_ll_noise - img_ll) / (img_ll + 1e-7)
		# r_max = t[:,:,0].max()
		# noise_map = np.max(abs(img_ll_noise - img_ll) / img_ll_noise, axis=(0,1,2))
		# noise = self.noise_map(img_ll_noise)

		noise_map = img_ll_noise - img_ll		# 对于非加性噪声，这种求法不对
		noise_map = self.img_org_transform(noise_map)
		img_ll = self.img_org_transform(img_ll)
		img_ll_noise = self.img_org_transform(img_ll_noise)

		if self.data_son is not '':   # 读取正常照度图像
			imgs_org_path = imgs_ll_path.replace(self.data_son['ll'], self.data_son['org'])
			imgs_org_path = imgs_org_path.replace('png', 'jpg')		# org集的图片格式为jpg
			img_org = img_processing.read_image(imgs_org_path, is_resize=self.is_resize, resize_height=self.resize_h,
			                                    resize_width=self.resize_w, normalization=False,
												is_long_resize=self.is_long_resize)
			img_org = self.img_org_transform(img_org)

			imgs_org_en_path = imgs_ll_path.replace(self.data_son['ll'], self.data_son['org_en'])
			img_org_en, y_en = img_processing.read_image(imgs_org_en_path, is_resize=self.is_resize, resize_height=self.resize_h,
												resize_width=self.resize_w, normalization=False,
												is_long_resize=self.is_long_resize,  is_cvtColor='YCrCb')
			img_org_en = self.img_org_transform(img_org_en)

			return img_ll, img_ll_noise, img_org, img_org_en, y, noise_map, name
		else:
			return img_ll, y, name



	def __len__(self):
		return len(self.imgs_ll)   # 总图片数量


def get_loader(data_root, data_son, batch_size, is_resize=False,resize_h=384, resize_w=384, img_type='jpg', is_long_resize=False):
	dataset = Load_Data(data_root, data_son, is_resize=is_resize, resize_h=resize_h, resize_w=resize_w, img_type=img_type, is_long_resize=is_long_resize)
	data_loader = DataLoader(dataset=dataset,
		                         batch_size=batch_size,
		                         shuffle=False,
		                         num_workers=1,
		                         pin_memory=True)   # 锁页内存，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，
													# 这样将内存的Tensor转义到GPU的显存就会更快一些
													# 显卡中的显存全部是锁页内存,当计算机的内存充足的时候，可以设置pin_memory=True
													# 省掉将数据从CPU传入到RAM中，再传到GPU上的过程。而是直接将数据映射到GPU的相关内存上，节省数据传输的时间
	return data_loader