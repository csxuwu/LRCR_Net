
import torch
from torch import nn

# from codes.utils.losses import Loss
from codes.losses.loss_rewrite import Loss
import random
import numpy as np

class Build_LRCR():

	def __init__(self, cfg, ):
		self.cfg = cfg
		# ---------------------
		# Models
		self.model = None
		self.model_type = cfg.model_type
		self.loss_type = cfg.loss_type
		self.l2_loss = nn.MSELoss()

		self.lr = cfg.lr
		self.lr_decay = cfg.lr_decay
		self.beta1 = cfg.beta1
		self.beta2 = cfg.beta2
		self.num_step_decay = cfg.num_step_decay
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = cfg.model_type

		self.build_loss()
		self.build_model()

	def build_model(self):

		if self.model_type == 'LRCR':
			from codes.models.LRCR.LRCR import LRCR
			self.model = LRCR(cfg=self.cfg)
		else:
			print('Error model building.')

		if torch.cuda.device_count() > 1:
			self.model = nn.DataParallel(self.model)

		self.model.to(self.device)

		self.print_network(self.model, self.model_type)

	def build_loss(self):
		# self.losses = Loss(loss_type=self.loss_type)
		self.loss = Loss(loss_type=self.loss_type, cfg=self.cfg)

	def print_network(self, model, name):
		"""Print out the network information."""

		print(name)
		# total num
		totoal_num_params = 0
		for p in model.parameters():  # sum(p.numel() for p in  self.model.parameters())
			totoal_num_params += p.numel()

		# trainable num
		trainable_num_params = 0
		for p in self.model.parameters():  # sum(p.numel() for p in  self.model.parameters() if p.requires_grad)
			if p.requires_grad:
				trainable_num_params += p.numel()

		print("The total number of parameters: {}".format(totoal_num_params))
		print("The trainable number of parameters: {}\n".format(trainable_num_params))

	def set_seed(self, seed):

		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
