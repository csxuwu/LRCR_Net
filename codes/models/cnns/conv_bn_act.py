

from torch import nn


class Conv_bn_act(nn.Module):
	'''
	'''
	def __init__(self,ch_in, ch_out, kernel_size=3, stride=1, padding=1, bn=True, activation='relu'):
		super(Conv_bn_act, self).__init__()
		self.bn = bn
		self.act_type = activation

		self.conv = nn.Conv2d(ch_in,ch_out,kernel_size,stride,padding,bias=True)
		self.bn2d = nn.BatchNorm2d(ch_out)

		if self.act_type == 'relu':
			self.act = nn.ReLU(inplace=True)
		elif self.act_type == 'prelu':
			self.act = nn.PReLU(num_parameters=1,init=0.25)
		elif self.act_type == 'lrelu':
			self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
		elif self.act_type == 'tanh':
			self.act = nn.Tanh()


	def forward(self, x):
		x = self.conv(x)

		if self.bn:
			x = self.bn2d(x)

		if self.act_type is not None:
			x = self.act(x)

		return x




