
import os
from codes.configs.train_configs import TrainConfigs
from codes.data import data_loader4
from codes.data import data_loader_tp
from codes.trainers.trainer_for_LRCR import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    trainConfigs = TrainConfigs(
        # model setting
        model_type='LRCR',
        model_version='debug',
        basic_path=r'../',
        patch_size2=4,
        num_heads=4,
        loss_type='rec_tex_vis',
        depth=[2, 2, 3, 2],
        filters='ContrastClip_Saturation',
        filters_param_ch={'Color': None, 'Contrast': 3, 'Saturation': 3, 'WB': None},

        # training setting
        debug=False,
        batch_size=8,
        num_epochs=10,
        start_epoch=0,
        train_dataset_name='AGLLSet4',
        test_dataset_name='LL_all2',
        is_resize=True,
        img_h=384,
        img_w=384,
        patch_size=256,
        is_data_augm=True,
        load_model_path = '',
        test=False,
    )
    trainConfigs.initialize()
    args = trainConfigs.args

    train_loader = data_loader4.get_loader(data_root=args.train_dataset_path,
                                           data_son=args.train_dataset_son,
                                           batch_size=args.batch_size,
                                           is_resize=args.is_resize,
                                           resize_w=args.img_w,
                                           resize_h=args.img_h)
    test_loader_real = data_loader_tp.get_loader(data_root=args.test_dataset_path,
                                                 data_son=args.test_dataset_son,
                                                 batch_size=1,
                                                 is_resize=False,
                                                 resize_h=512, resize_w=512,
                                                 is_long_resize=True)
    test_loader_pair = data_loader_tp.get_loader(data_root=r'G:\Dataset\LL_Set',  # test dataset path
                                                 data_son={'ll': 'NASA', 'org': 'NASA-high'},
                                                 batch_size=1,
                                                 is_resize=False,
                                                 resize_h=512, resize_w=512,
                                                 is_long_resize=True,
                                                 img_type='jpg')
    test_loader_pair2 = data_loader_tp.get_loader(data_root=r'G:\WUXU\Dataset\Synthetic Lowlight Dataset',  # test dataset path
                                                  data_son={'ll': 'test_lowlight', 'org': 'test'},
                                                  batch_size=1,
                                                  is_resize=False,
                                                  resize_h=512,
                                                  resize_w=512,
                                                  is_long_resize=True,
                                                  img_type='jpg')
    test_loader = {'real': test_loader_real,
                   'NASA': test_loader_pair,
                   'AGLLSet_test': test_loader_pair2}

    t = Trainer(cfg=args, train_loader=train_loader, test_loader=test_loader)
    t.trainer()
