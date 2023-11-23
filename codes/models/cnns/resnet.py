

import torch

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


# ---------------------------
# 20200326
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch
# ---------------------------

__all__ = ['ResNet', 'resnet50']

model_urls = {
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class Bottleneck(nn.Module):
	'''
	瓶颈状的残差卷积块
	'''
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)     # 调整通道数到瓶颈通道数
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,    # 卷积
							   padding=rate, dilation=rate, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)  # 调整通道至输出通道数
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)

		self.relu = nn.ReLU(inplace=True)

		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			# 调整通道数
			residual = self.downsample(x)

		out += residual     # 跳跃连接
		out = self.relu(out)

		return out

class ResNet(nn.Module):
	'''
	将第4个卷积组替换成dilated conv
	'''
	def __init__(self, block, layers, num_classes=1000):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 64, layers[0])                # 64*4=256,256*128*128,64为瓶颈层的通道数，输出通道数为瓶颈层的4倍
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)     # 128*4=512,512*64*64
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

		rates = [1, 2, 4]
		self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		'''
		构建resnet的大模块，两个池化层的中间，称作为大模块
		1个layer的组成：
			下采样：步长为2的卷积，第一个conv进行下采样
			卷积模块：除第1个conv，其他conv都进行卷积操作
		:param block:
		:param planes:bottleneck瓶颈的通道数
		:param blocks:block的数量
		:param stride:步长
		:return:
		'''
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:        # planes * block.expansion  bottleneck残差模块的输出通道数
			downsample = nn.Sequential(     # 如果有下采样，则调整输入的通道数，再做跳跃连接
				nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))     # 第一个block可能是要进行下采样的
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):      # 卷积模块
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
		'''
		与layer不同的是，将标准卷积换成dilated conv
		:param block:
		:param planes:
		:param blocks:
		:param rates:
		:param stride:
		:return:
		'''
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, rate=rates[i]))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)       # convk7s2c64，下采样
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)     # 下采样

		x = self.layer1(x)      # 无下采样
		x = self.layer2(x)      # 有采样
		x = self.layer3(x)      # 有采样
		x = self.layer4(x)      # 标准卷积换成dilated conv，无下采样

		x = self.avgpool(x)     # 平均池化
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
		model.load_state_dict(torch.load(r'..\pretrains\resnet50-19c8e357.pth'))
	return model
