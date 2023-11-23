


import numpy as np

def data_augmentation(image, mode):
	if mode == 0:
		# original
		return image
	elif mode == 1:
		# flip up and down
		return np.flipud(image)
	elif mode == 2:
		# rotate counterwise 90 degree
		return np.rot90(image)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		image = np.rot90(image)
		return np.flipud(image)
	elif mode == 4:
		# rotate 180 degree
		return np.rot90(image, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		image = np.rot90(image, k=2)
		return np.flipud(image)
	elif mode == 6:
		# rotate 270 degree
		return np.rot90(image, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		image = np.rot90(image, k=3)    # 逆时针旋转90度
		return np.flipud(image)         #  将第一个维度的值倒序
