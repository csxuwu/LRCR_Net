

import os
import torch
from torchvision import utils
from tqdm import tqdm


def save_img_for_train(model_name_son, img_org, img_ll, img_enhance, out_path, step, epoch):
	'''
	存储训练图像
	:param self:
	:param img_org:
	:param img_ll:
	:param img_enhance:
	:param out_path:
	:param step:
	:return:
	'''
	train_imgs = []
	train_imgs.extend([img_org.data.cpu().squeeze(0),
	                   # img_org.data将Variable中的tensor取出来，.cpu() 放cpu上 img_org.data.cpu()等同于 img_org.cpu().data
	                   img_ll.data.cpu().squeeze(0),
	                   img_enhance.data.cpu().squeeze(0)])
	train_imgs = torch.stack(train_imgs)
	train_imgs = torch.chunk(train_imgs,
	                         train_imgs.size(0) // 1)  # 作用与concat想法，将输入按照某个维度，拆成n个子模块，返回的是list
	train_save_bar = tqdm(train_imgs, desc='[saving training results]')
	tp = 1
	for img in train_save_bar:
		img = utils.make_grid(img, nrow=3, padding=5)  # 将多张图拼成一张图
		img_save_path = os.path.join(out_path, model_name_son + '_train_' + str(epoch) + '_' + str(step) + '_' + str(tp) + '.png')
		utils.save_image(img, filename=img_save_path)
		tp += 1


def save_img_for_test(model_name_son, img_ll, img_enhance, out_path, index, epoch):
	'''
	存储测试图像
	:param self:
	:param img_ll:
	:param img_enhance:
	:param out_path:
	:param index:
	:return:
	'''
	train_imgs = []
	train_imgs.extend([img_ll.data.cpu().squeeze(0),
	                   img_enhance.data.cpu().squeeze(0)])
	train_imgs = torch.stack(train_imgs)
	train_imgs = torch.chunk(train_imgs,
	                         train_imgs.size(0) // 1)  # 作用与concat想法，将输入按照某个维度，拆成n个子模块，返回的是list
	# train_save_bar = tqdm(train_imgs, desc='[saving training results]')
	tp = 1
	for img in train_imgs:
		img = utils.make_grid(img, nrow=3, padding=5)  # 将多张图拼成一张图
		# print(out_path)
		img_save_path = os.path.join(out_path, model_name_son + '_test_' + str(epoch) + '_'+ str(index) + '_' + str(tp) + '.png')
		utils.save_image(img, filename=img_save_path)
		tp += 1