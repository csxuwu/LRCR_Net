

import cv2
import numpy as np
import math


def array_mu(array):
	'''

	:param array:
	:return:
	'''
	sum = 0
	for i in range(len(array)):
		sum += array[i]
	return sum / len(array)

def psnr(img1,img2):
	'''

	:param img1:
	:param img2:
	:return:
	'''
	psnr = []
	for i in range(len(img1)):
		img1_cpu,img2_cpu = img1.cpu(),img2.cpu()
		img1_np = img1_cpu[i].detach().numpy()
		img2_np = img2_cpu[i].detach().numpy()

		mse = np.mean((img1_np/1.0 - img2_np/1.0)**2)
		psnr_val = 10 * math.log10((1.0 ** 2)/mse)
		psnr.append(psnr_val)
	return array_mu(psnr)

def psrn2(img1,img2):
	mse = ()