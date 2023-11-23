import os
from codes.utils import ops
from .base_configs import BaseConfigs


class TestConfigs(BaseConfigs):
	def __init__(self,
	             # model
	             model_type='',
				 model_version=0,
				 basic_path='../logs',
	             patch_size2=16,
				 num_heads=4,
				 depth=[1,1,1,1],
				 filters_param_ch={'Color': None, 'Contrast': 3, 'Saturation': None, 'WB': None},
				 filters='ContrastClip',
	             loss_type='l2',
	             # testing
	             train_dataset_name='',
				 cur_epoch=49,
	             batch_size=1,
				 load_model_path='',
				 test_dataset_name='',
				 is_resize=True,
				 is_real_img=True,
	             img_h=256,
				 img_w=256,
				 patch_size=256):
		BaseConfigs.__init__(self, model_type=model_type, model_version=model_version, basic_path=basic_path,
		                     patch_size2=patch_size2, num_heads=num_heads,
		                     loss_type=loss_type, )

		self.batch_size = batch_size
		self.cur_epoch = cur_epoch
		self.load_model_path = load_model_path

		# dataset setting
		self.train_dataset_name = train_dataset_name
		self.test_dataset_name = test_dataset_name
		self.is_real_img = is_real_img
		self.is_resize = is_resize
		self.img_h = img_h
		self.img_w = img_w
		self.patch_size = patch_size
		self.depth = depth
		self.filters = filters
		self.filters_param_ch = filters_param_ch


	def initialize(self):

		BaseConfigs.initialize(self)

		self.parser.add_argument('--Testing configs', type=str, default='---------------------------------------- : Begin')
		self.parser.add_argument('--load_model_path', type=str, default=self.load_model_path)

		# test dataset
		# -----------------------------------------------------------------------------------------
		test_dataset_path, test_dataset_son = self.set_dataset(self.test_dataset_name)
		self.parser.add_argument('--train_dataset_name', type=str, default=self.train_dataset_name)
		self.parser.add_argument('--test_dataset_name', type=str, default=self.test_dataset_name)
		self.parser.add_argument('--test_dataset_path', type=str, default=test_dataset_path)
		self.parser.add_argument('--test_dataset_son', type=str, default=test_dataset_son)
		self.parser.add_argument('--is_real_img', type=str, default=self.is_real_img)
		self.parser.add_argument('--is_resize', type=bool, default=self.is_resize, help='resizing images or Not')
		self.parser.add_argument('--img_h', type=int, default=self.img_h, help='high of resized images')
		self.parser.add_argument('--img_w', type=int, default=self.img_w, help='wide of resized images')
		self.parser.add_argument('--img_size', type=int, default=self.patch_size, help='size of cropping images')
		self.parser.add_argument('--patch_size', type=int, default=self.patch_size, help='size of cropping images')
		self.parser.add_argument('--cur_epoch', type=bool, default=self.cur_epoch)
		self.parser.add_argument('--depth', type=list, default=self.depth)
		self.parser.add_argument('--filters', type=list, default=self.filters)
		self.parser.add_argument('--filters_param_ch', type=dict, default=self.filters_param_ch)

		# saving path
		# -----------------------------------------------------------------------------------------
		path_main = os.path.join(r'..\logs', self.model_name)
		foder_name_test = 'test_' + str(self.model_name) + '_' + self.train_dataset_name + '_' + str(self.patch_size) + '_' + self.loss_type
		test_log_path = os.path.join(path_main, foder_name_test, 'logs')
		test_out_path = os.path.join(path_main, foder_name_test, 'out', self.model_name+'_'+str(self.cur_epoch))
		self.parser.add_argument('--foder_name_test', type=str, default=foder_name_test)
		self.parser.add_argument('--test_log_path', type=str, default=test_log_path)
		self.parser.add_argument('--test_out_path', type=str, default=test_out_path)
		self.parser.add_argument('--Testing configs END', type=str, default='--------------------------- : Testing configs END')

		ops.create_folder(test_log_path)
		ops.create_folder(test_out_path)

		file_name = 'test_params_' + self.model_name + '_epoch'+str(self.cur_epoch) + '.txt'
		self.parse(out_path=test_log_path, file_name=file_name)

	def set_dataset(self, dataset_name):
		test_dataset_path = ''
		test_dataset_son = ''

		if dataset_name == 'NASA':
			test_dataset_path = 'E:\Dataset_LL\LLTest_Set'
			test_dataset_son = {'ll': 'NASA', 'org': 'NASA-high'}
		elif dataset_name == 'LIME':
			test_dataset_path = 'E:\Dataset_LL\LLTest_Set\LIME'
		elif dataset_name == 'MEF':
			test_dataset_path = 'E:\Dataset_LL\LLTest_Set\MEF'
		elif dataset_name == 'NPE':
			test_dataset_path = r'E:\Dataset_LL\LLTest_Set\NPE'
		elif dataset_name == 'TID2013':
			test_dataset_path = 'E:\Dataset_LL\LLTest_Set\TID2013'
		elif dataset_name == 'VV':
			test_dataset_path = 'E:\Dataset_LL\LLTest_Set\VV'
		elif dataset_name == 'Nirscene':
			test_dataset_path = r'E:\Dataset_LL\LLTest_Set\Nirscene'
		elif dataset_name == 'LOL_test':
			test_dataset_path = r'E:\Dataset_LL\LOLdataset\eval15\test_x'
		else:
			print('Dataset is None!')

		return test_dataset_path, test_dataset_son
