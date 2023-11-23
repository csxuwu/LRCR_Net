
import argparse
import os

class BaseConfigs():
	def __init__(self,
				 model_type='',
				 model_version=0,
				 basic_path='../logs',
	             patch_size2=16,
				 num_heads=4,
	             en_dim=128,
				 en_depth=0,
				 de_dim=128,
				 de_depth=0,
		         loss_type='l2',
				 sam_type='',
				 debug=False):

		# basic information of model
		self.model_type = model_type
		self.model_version = model_version
		self.model_name = self.model_type + '_' + str(self.model_version)
		self.basic_path = basic_path

		# Transformer parameters
		self.patch_size2 = patch_size2
		self.num_heads = num_heads

		# model parameters
		self.en_dim = en_dim
		self.en_depth = en_depth
		self.de_dim = de_dim
		self.de_depth = de_depth

		self.loss_type = loss_type
		self.sam_type = sam_type
		self.debug = debug

		self.parser = argparse.ArgumentParser(description=self.model_name + ' Training with Pytorch.')
		self.initialized = False

	def initialize(self):

		self.parser.add_argument('--Model configs BEGIN', type=str, default='--------------------------- : Model configs BEGIN')
		# model-parameters
		# -----------------------------------------------------------------------------------------
		self.parser.add_argument('--model_type', type=str, default=self.model_type)
		self.parser.add_argument('--model_version', type=str, default=self.model_version)
		self.parser.add_argument('--model_name', type=str, default=self.model_name)

		self.parser.add_argument('--en_dim', type=str, default=self.en_dim, help='channel dimensions of encoder')
		self.parser.add_argument('--encoder_depth', type=int, default=self.en_depth, help='depth of encoder')
		self.parser.add_argument('--de_dim', type=str, default=self.de_dim, help='channel dimensions of decoder')
		self.parser.add_argument('--de_depth', type=int, default=self.de_depth, help='depth of encoder')
		self.parser.add_argument('--loss_type', type=str, default=self.loss_type, help='type of losses function')
		self.parser.add_argument('--sam_type', type=str, default=self.sam_type)

		# Transformer-parameter
		# -----------------------------------------------------------------------------------------
		self.parser.add_argument('--patch_size2', type=int, default=self.patch_size2, help='patch size of transformer')
		self.parser.add_argument('--num_heads', type=int, default=self.num_heads, help='number of attention heads')

		# hype-parameters
		# -----------------------------------------------------------------------------------------
		self.parser.add_argument('--gpu_id', type=int, default=[0, 1], help='gpu_id')
		self.parser.add_argument('--debug', type=bool, default=self.debug, help='debug?')
		self.parser.add_argument('--local_rank', type=int, default=-1)
		self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
		self.parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay')
		self.parser.add_argument('--beta1', type=float, default=0.9, help='parameter of ADAM')

		self.parser.add_argument('--beta2', type=float, default=0.999, help='parameter of ADAM')
		self.parser.add_argument('--num_step_decay', type=float, default=10000, help='cycle of learning rate decay')
		self.parser.add_argument('--basic_path', type=str, default=self.basic_path)
		self.parser.add_argument('--Model configs END', type=str, default='--------------------------- : Model configs BEGIN\n')

		self.initialized = True

	def parse(self, out_path, file_name):

		# saving all parameters
		if not self.initialized:
			self.initialize()

		self.args = self.parser.parse_args()
		args_vars = vars(self.args)
		file_path = os.path.join(out_path, file_name)

		print('------------ Options -------------')
		# for k, v in sorted(args_vars.items()):
		for k, v in (args_vars.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		with open(file_path, 'wt') as opt_file:
			opt_file.write('------------ Configs -------------\n')
			# for k, v in sorted(args_vars.items()):
			for k, v in (args_vars.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')

	def get_config(self, config):
		import yaml
		with open(config, 'r') as stream:
			return yaml.load(stream)
