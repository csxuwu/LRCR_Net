
import os
from codes.utils import ops
from .base_configs import BaseConfigs


# ------------------------------
# 20210609
# WX
# 训练时用到的参数
# ------------------------------

class TrainConfigs(BaseConfigs):
	def __init__(self,
	             # model
	             model_type='', model_version=0, basic_path='G:\Code\ACMMM2022',
	             patch_size2=16, num_heads=4,
	             en_dim=128, en_depth=0, de_dim=128, de_depth=0,
		         loss_type='l2', sam_type='', debug=False, filters_param_ch=None,filters='C',
	             # training
	             batch_size=4, num_epochs=8, start_epoch=0, load_model_path='',train_dataset_name='',test_dataset_name='',
	             is_resize=True, img_h=256, img_w=256, patch_size=256, is_data_augm=True, depth=[1,1,1,1],
				 # testing
				 gamma=3, noise=5,test=False,):
		BaseConfigs.__init__(self, model_type=model_type, model_version=model_version, basic_path=basic_path,
		                     patch_size2=patch_size2, num_heads=num_heads,
		                     en_dim=en_dim, en_depth=en_depth, de_dim=de_dim, de_depth=de_depth,
		                     loss_type=loss_type, sam_type=sam_type, debug=debug)

		self.start_epoch = start_epoch
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.load_model_path = load_model_path

		self.filters_param_ch = filters_param_ch
		self.filters = filters

		# dataset setting
		self.train_dataset_name = train_dataset_name
		self.test_dataset_name = test_dataset_name
		self.is_resize = is_resize
		self.img_h = img_h
		self.img_w = img_w
		self.depth = depth	# depth of network
		self.patch_size = patch_size
		self.is_data_augm = is_data_augm
		self.gamma = gamma
		self.noise = noise
		self.test = test
		# self.initialize()


	def initialize(self):

		BaseConfigs.initialize(self)

		self.parser.add_argument('--Training configs BEGIN', type=str, default='--------------------------- : Training configs BEGIN')
		self.parser.add_argument('--start_epoch', type=int, default=self.start_epoch, help='starting epoch for training')
		self.parser.add_argument('--load_model_path', type=str, default=self.load_model_path)
		self.parser.add_argument('--is_train', type=bool, default=True)
		self.parser.add_argument('--num_epochs', type=int, default=self.num_epochs)
		self.parser.add_argument('--batch_size', type=int, default=self.batch_size)

		self.parser.add_argument('--filters_param_ch', type=dict, default=self.filters_param_ch)
		self.parser.add_argument('--filters', type=str, default=self.filters)

		# training dataset
		# -----------------------------------------------------------------------------------------
		train_dataset_path, train_dataset_son, test_dataset_path, test_dataset_son  = self.set_dataset(self.train_dataset_name, self.test_dataset_name)
		self.parser.add_argument('--train_dataset_name', type=str, default=self.train_dataset_name)
		self.parser.add_argument('--train_dataset_path', type=str, default=train_dataset_path)
		self.parser.add_argument('--train_dataset_son', type=str, default=train_dataset_son)

		self.parser.add_argument('--test_dataset_name', type=str, default=self.test_dataset_name)
		self.parser.add_argument('--test_dataset_path', type=str, default=test_dataset_path)
		self.parser.add_argument('--test_dataset_son', type=str, default=test_dataset_son)

		self.parser.add_argument('--is_resize', type=bool, default=self.is_resize, help='resizing images or Not')
		self.parser.add_argument('--img_h', type=int, default=self.img_h, help='high of resized images')
		self.parser.add_argument('--img_w', type=int, default=self.img_w, help='wide of resized images')
		self.parser.add_argument('--img_size', type=int, default=self.patch_size, help='size of cropping images')
		self.parser.add_argument('--patch_size', type=int, default=self.patch_size, help='size of cropping images')
		self.parser.add_argument('--is_data_augm', type=bool, default=self.is_data_augm)
		self.parser.add_argument('--depth', type=list, default=self.depth)

		self.parser.add_argument('--gamma', type=int, default=self.gamma)
		self.parser.add_argument('--noise', type=int, default=self.noise)
		self.parser.add_argument('--test', type=bool, default=self.test)

		# saving path
		# -----------------------------------------------------------------------------------------
		path_main = os.path.join(self.basic_path,'logs', self.model_name)
		foder_name = 'train_' + str(self.model_name) + '_' + self.train_dataset_name + '_' + str(self.patch_size) + '_' + str(self.loss_type)
		foder_name_test = 'test_' + str(self.model_name) + '_' + self.train_dataset_name + '_' + str(self.patch_size) + '_' + str(self.loss_type)
		train_log_path = os.path.join(path_main, foder_name, 'logs')
		train_log_tb_path = os.path.join(path_main, foder_name, 'logs_tb')
		train_out_path = os.path.join(path_main ,foder_name, 'out')
		test_log_path = os.path.join(path_main, foder_name_test, 'logs')
		test_out_path = os.path.join(path_main ,foder_name_test, 'out')
		self.parser.add_argument('--train_log_tb_path', type=str, default=train_log_tb_path)
		self.parser.add_argument('--foder_name', type=str, default=foder_name)
		self.parser.add_argument('--train_log_path', type=str, default= train_log_path)
		self.parser.add_argument('--train_out_path', type=str, default= train_out_path)
		self.parser.add_argument('--test_log_path', type=str, default= test_log_path)
		self.parser.add_argument('--test_out_path', type=str, default= test_out_path)
		self.parser.add_argument('--Training configs END', type=str, default='--------------------------- : Training configs END')

		ops.create_folder(train_log_path)
		ops.create_folder(train_out_path)
		ops.create_folder(test_log_path)
		ops.create_folder(test_out_path)
		file_name = 'train_params_'+ self.model_name + '.txt'
		self.parse(out_path=train_log_path, file_name=file_name)


	def set_dataset(self, train_dataset_name, test_dataset_name):
		train_dataset_path = ''
		train_dataset_son = ''

		# ----------------------------------------------------------------------------
		# 训练数据
		if train_dataset_name == 'ExDark120':
			train_dataset_path = 'G:\Dataset\LL_Set\ExDark120'
		elif train_dataset_name == 'Syn5':
			train_dataset_path = r'G:\Dataset\Syn'
			train_dataset_son = {'ll': 'syn5', 'org': 'high'}
		elif train_dataset_name == 'LOL_train':
			train_dataset_path = r'G:\Dataset\LOL\LOLdataset\our485'
			train_dataset_son = {'ll': 'low', 'org': 'high'}
		elif train_dataset_name == 'LOL_train2':
			train_dataset_path = r'G:\Dataset\LOL\BrighteningTrain\BrighteningTrain'
			train_dataset_son = {'ll': 'low', 'org': 'high'}
		# elif train_dataset_name == 'AGLLSet':
		# 	train_dataset_path = r'G:\WUXU\Dataset\Synthetic Lowlight Dataset'
		# 	# train_dataset_son = {'org': 'train', 'll': 'low',}
		elif train_dataset_name == 'AGLLSet':
			train_dataset_path = r'G:\WUXU\Dataset\Synthetic Lowlight Dataset'
			train_dataset_son = {'ll': 'train_lowlight', 'org': 'train'}  # train_lowlight：低照度+噪声
		elif train_dataset_name == 'AGLLSet2':
			train_dataset_path = r'G:\WUXU\Dataset\Synthetic Lowlight Dataset'
			train_dataset_son = {'ll': 'train_dark', 'org': 'train'}  # train _dark: 低照度
		elif train_dataset_name == 'AGLLSet3':
			train_dataset_path = r'G:\WUXU\Dataset\Synthetic Lowlight Dataset'
			train_dataset_son = {'ll': 'train_dark', 'll_noise':'train_lowlight', 'org': 'train'}  # train _dark: 低照度
		elif train_dataset_name == 'AGLLSet4':
			train_dataset_path = r'G:\WUXU\Dataset\Synthetic Lowlight Dataset'
			train_dataset_son = {'ll': 'train_dark', 'll_noise':'train_lowlight',
								 'org': 'train', 'org_en':'train_enhance'}  # train _dark: 低照度
		elif train_dataset_name == 'AGLLSet41':
			train_dataset_path = r'G:\WUXU\Dataset\Synthetic Lowlight Dataset'
			train_dataset_son = {'ll': 'train_dark', 'll_noise':'train_dark',
								 'org': 'train', 'org_en':'train_enhance'}  # train _dark: 低照度


		test_dataset_path = ''
		test_dataset_son = ''

		# ----------------------------------------------------------------------------
		# 训练数据
		if test_dataset_name == 'MEF':
			# 单张图片
			test_dataset_path = 'G:\Dataset\LL_Set\MEF'
		elif test_dataset_name == 'LOL':
			test_dataset_path = 'G:\Dataset\LL_Set'
			test_dataset_son = {'ll': 'LOL', 'org': 'LOL-high'}  # 低照度
		elif test_dataset_name == 'LL_all':
			test_dataset_path = r'G:\Dataset\LL_Set\LL_all'
		elif test_dataset_name == 'LL_all2':
			test_dataset_path = r'G:\Dataset\LL_Set\LL_all2'
		elif test_dataset_name == 'syn_test':
			test_dataset_path = r'G:\Dataset\LL_Set\syn_test'




			# 单张真实低照度
			# -------------------------------------------
		elif test_dataset_name == 'NASA':
			test_dataset_path = 'G:\Dataset\LL_Set'
			test_dataset_son = {'ll': 'NASA', 'org': 'NASA-high'}  # 低照度
		elif test_dataset_name == 'LIME':
			test_dataset_path = 'G:\Dataset\LL_Set\LIME'
		elif test_dataset_name == 'MEF':
			test_dataset_path = 'G:\Dataset\LL_Set\MEF'
		elif test_dataset_name == 'NPE':
			test_dataset_path = r'G:\Dataset\LL_Set\NPE'
		elif test_dataset_name == 'TID2013':
			test_dataset_path = 'G:\Dataset\LL_Set\TID2013'
		elif test_dataset_name == 'VV':
			test_dataset_path = 'G:\Dataset\LL_Set\VV'
		elif test_dataset_name == 'Nirscene':
			test_dataset_path = r'G:\Dataset\LL_Set\Nirscene'
		elif test_dataset_name == 'LOL_test':
			test_dataset_path = r'G:\Dataset\LL_Set\LOL'
		elif test_dataset_name == 'DICM':
			test_dataset_path = r'G:\Dataset\LL_Set\DICM'
		# -------------------------------------------


		return train_dataset_path, train_dataset_son, test_dataset_path, test_dataset_son
