
import os
import torch
from torchvision import utils
from tqdm import tqdm

def save_img_for_train(model_name, imgs, out_path, step, epoch):
	'''

	:param self:
	:param img_org:
	:param img_ll:
	:param img_enhance:
	:param out_path:
	:param step:
	:return:
	'''
	train_imgs = []
	temp_imgs = []
	for i, img in enumerate(imgs):
		temp_imgs.append(img.cpu().squeeze(0))

	train_imgs.extend(temp_imgs)
	train_imgs = torch.stack(train_imgs)
	train_imgs = torch.chunk(train_imgs,
	                         train_imgs.size(0) // 1)
	train_save_bar = tqdm(train_imgs, desc='[saving training results]')
	tp = 1
	for img in train_save_bar:
		img = utils.make_grid(img, nrow=3, padding=5)
		img_save_path = os.path.join(out_path, model_name + '_train_' + str(epoch) + '_' + str(step) + '_' + str(tp) + '.png')
		utils.save_image(img, fp=img_save_path)
		tp += 1

def save_img_for_test(imgs, out_path, img_name, model_name_son='', index=0, epoch=0):
	'''

	:param self:
	:param img_ll:
	:param img_enhance:
	:param out_path:
	:param index:
	:return:
	'''
	train_imgs = []
	temp_imgs = []
	for i ,img in enumerate(imgs):
		try:
			temp_imgs.append(img.cpu().squeeze(0))
		except:
			print('testing erro.')
	train_imgs.extend(temp_imgs)
	index=0
	for img in train_imgs:
		img_save_path = os.path.join(out_path, img_name + str(index)+'.jpg')
		utils.save_image(img, fp=img_save_path)
		index += 1

