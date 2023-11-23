

import os
import glob
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.distributed import DistributedSampler
from codes.utils import img_processing
from codes.data import data_utils


class Load_Data(Dataset):
	'''
	读取图片
	获取低照度图像的Y通道
	'''
	def __init__(self, data_root, data_son=None, img_type='jpg', is_resize=False, is_long_resize=False, resize_h=512, resize_w=512):
		if data_son is not '':
			# 如果非None，则读取成对的低照度-正常照度图像
			# t = os.path.join(data_root, data_son['ll'], '*.'+img_type)
			# t2 = os.path.join(data_root, data_son['org'], '*.'+img_type)
			# imgs_ll = glob.glob(os.path.join(data_root, data_son['ll'], '*.'+img_type))
			# imgs_org = glob.glob(os.path.join(data_root, data_son['org'], '*.'+img_type))
			imgs_ll = glob.glob(os.path.join(data_root, data_son['ll'], '*.*' ))
			imgs_org = glob.glob(os.path.join(data_root, data_son['org'], '*.*'))
			self.imgs_org = imgs_org
		else:
			imgs_ll = glob.glob(os.path.join(data_root, '*.*'))

		self.imgs_ll = imgs_ll
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
		imgs_ll_path = self.imgs_ll[index]      # 返回下标为index的低照度图片路径

		[_, name] = os.path.split(imgs_ll_path)
		suffix = name[name.find('.') + 1:]  # 图片类型
		name = name[:name.find('.')]

		img_ll, y = img_processing.read_image(imgs_ll_path, is_resize=self.is_resize, resize_height=self.resize_h,
		                                   resize_width=self.resize_w, normalization=True,
										   is_long_resize=self.is_long_resize, is_cvtColor='YCrCb')
		img_ll = self.img_org_transform(img_ll)


		if self.data_son is not '':   # 读取正常照度图像
			imgs_org_path = self.imgs_org[index]
			img_org = img_processing.read_image(imgs_org_path, is_resize=self.is_resize, resize_height=self.resize_h,
			                                    resize_width=self.resize_w, normalization=True,
												is_long_resize=self.is_long_resize)
			img_org = self.img_org_transform(img_org)
			return img_ll, img_org, y, name
		else:
			return img_ll, y, name


	def __len__(self):
		return len(self.imgs_ll)   # 总图片数量


def get_loader(data_root, data_son, batch_size, is_resize=False,resize_h=384, resize_w=384, img_type='jpg', is_long_resize=False):
	dataset = Load_Data(data_root, data_son, is_resize=is_resize, resize_h=resize_h, resize_w=resize_w, img_type=img_type, is_long_resize=is_long_resize)
	data_loader = DataLoader(dataset=dataset,
		                         batch_size=batch_size,
		                         shuffle=False,
		                         num_workers=0,
		                         pin_memory=True)   # 锁页内存，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，
													# 这样将内存的Tensor转义到GPU的显存就会更快一些
													# 显卡中的显存全部是锁页内存,当计算机的内存充足的时候，可以设置pin_memory=True
													# 省掉将数据从CPU传入到RAM中，再传到GPU上的过程。而是直接将数据映射到GPU的相关内存上，节省数据传输的时间
	return data_loader