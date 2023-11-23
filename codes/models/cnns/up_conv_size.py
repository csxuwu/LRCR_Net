

from torch import nn
import torch.nn.functional as F

class Up_conv_size(nn.Module):
	'''
	 up + conv + bn + relu
	'''
	def __init__(self, ch_in, ch_out):
		super(Up_conv_size, self).__init__()
		self.up_conv = nn.Sequential(
			nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x, en_x):
		h = en_x.size()[2]  # NCHW
		w = en_x.size()[3]
		size = (h, w)
		up = F.interpolate(input=x, size=size, mode='bilinear')
		up = self.up_conv(up)
		return up