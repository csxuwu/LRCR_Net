

from torchvision import transforms as T

def train_org_transforms():
	'''
	对正常照度图片进行处理
	:return:
	'''
	tranforms = T.Compose([
		# T.ToPILImage(),
		T.ToTensor()
	])
	return tranforms


def train_ll_transforms():
	'''
	对低照度图片处理
	:return:
	'''
	tranforms = T.Compose([
		T.ToPILImage(),
		T.ToTensor()
	])
	return tranforms


def test_transforms():
	'''
	对正常照度图片进行处理
	:return:
	'''
	tranforms = T.Compose([
		T.ToTensor()
	])
	return tranforms

def display_transforms():
	'''
	对输出图片处理
	:return:
	'''
	tranforms = T.Compose([
		T.ToPILImage(),
		# T.ToTensor()
	])
	return tranforms