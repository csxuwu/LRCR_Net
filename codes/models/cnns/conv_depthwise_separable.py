

from torch import nn
import torch.nn.functional as F


class Conv_DW(nn.Module):
	'''

	'''
	def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, dilation=1, padding=1):
		super(Conv_DW, self).__init__()
		self.depth_conv = nn.Conv2d(in_channels=ch_in,
									out_channels=ch_in,
									kernel_size=kernel_size,
									stride=stride,
									padding=padding,
		                            dilation=dilation,
									groups=ch_in)

		self.point_conv = nn.Conv2d(in_channels=ch_in,
									out_channels=ch_out,
									kernel_size=1,
									stride=1,
									padding=0,
									groups=1)
		self.act = nn.ReLU(True)


	def forward(self, x):
		x = self.depth_conv(x)
		x = self.point_conv(x)
		# x = self.act(x)
		x = F.relu(x)
		return x