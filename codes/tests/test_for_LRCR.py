import logging
import tqdm
from torch.autograd import Variable
from collections import OrderedDict

from codes.utils.build_LRCR import Build_LRCR
from codes.data import data_loader2 as data_loader
from codes.utils.ops import create_folder
from codes.utils.img_save_v1 import *
from torch.utils.tensorboard import SummaryWriter

from codes.configs.train_configs import TrainConfigs

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(model,
         model_name,
         dataset_root,
         dataset,
         base_path,
         model_pth_path):

    test_out_path = os.path.join(base_path, 'out', dataset + '_CR')       # outputs of the color refinement
    test_out_path2 = os.path.join(base_path, 'out', dataset + '_LR_CR')   # outputs of the light-restoration and the color refinement
    test_out_path3 = os.path.join(base_path, 'out', dataset + '_all')

    test_out_path_log = os.path.join(base_path, 'logs', dataset)
    create_folder(test_out_path)
    create_folder(test_out_path2)
    create_folder(test_out_path3)
    create_folder(test_out_path_log)
    writer = SummaryWriter(test_out_path_log)

    logging.info('{} is loade from {}.'.format(model_name, model_pth_path))

    load_net = torch.load(model_pth_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean)
    model.to(device)

    dataroot = os.path.join(dataset_root, dataset)
    dataloader = data_loader.get_loader(data_root=dataroot,
                                        data_son='',
                                        batch_size=1,
                                        is_resize=False)

    test_bar = tqdm(dataloader)
    with torch.no_grad():
        test_bar.set_description(desc='[Testing real %s | %s]' % (model_name, dataset))
        for j, (img_ll_test, y, name) in enumerate(test_bar):
            img_ll_test = Variable(img_ll_test)
            img_ll_test = img_ll_test.cuda()
            y = Variable(y)
            y = y.cuda()
            img_name = name[0]

            try:
                out, out2 = model({'img_ll_noise': img_ll_test, 'illu_map': y, 'is_test': True})
                save_img_for_test(model_name_son=model_name, imgs=[out['img_enhance2']],
                                  out_path=test_out_path, index=j, epoch=9, img_name=img_name)
                save_img_for_test(model_name_son=model_name, imgs=[out['img_enhance1'], out['img_enhance2']],
                                  out_path=test_out_path2, index=j, epoch=9, img_name=img_name)
            except RuntimeError as exception:
                if "out of memory" in str(exception) or "illegal memory access" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            for key in out:
                tag = 'test/' + key
                if out[key] is not None and len(out[key].size()) == 4:
                    if key == 'slice_coeffs':
                        writer.add_image(tag='test/slice_coeffs_r',
                                         img_tensor=out["slice_coeffs"][:, 0:3, :, :],
                                         dataformats='NCHW')
                        writer.add_image(tag='test/slice_coeffs_g',
                                         img_tensor=out["slice_coeffs"][:, 4:7, :, :],
                                         dataformats='NCHW')
                        writer.add_image(tag='test/slice_coeffs_b',
                                         img_tensor=out["slice_coeffs"][:, 8:11, :, :],
                                         dataformats='NCHW')
                    elif key != 'coeffs_out':
                        writer.add_image(tag, out[key], dataformats='NCHW')
                        save_img_for_test(model_name_son=model_name,
                                          imgs=out[key],
                                          out_path=test_out_path3, index=j, epoch=9,
                                          img_name=img_name + '_' + key)
                elif out[key] is not None and len(out[key].size()) == 3:
                    writer.add_image(tag, out[key], dataformats='CHW')

if __name__ == '__main__':

    model_name = 'LRCR'
    dataset_list = ['LIME']   # LIME, NPE, MEF, NASA
    dataset_root = r'G:\Dataset\LL_Set'     # root path of testing dataset
    save_path = r'../results/'
    model_pth_path = r'../../pths/LRCR.pth'

    trainConfigs = TrainConfigs(
        model_type=model_name,
        patch_size2=4,
        depth=[2, 2, 3, 2],
        filters='ContrastClip_Saturation',
        filters_param_ch={'Color': None, 'Contrast': 3, 'Saturation': 3, 'WB': None})
    trainConfigs.initialize()
    args = trainConfigs.args
    build = Build_LRCR(args)
    from codes.models.LRCR.LRCR import LRCR
    model = LRCR(args)

    for dataset in dataset_list:
        eval(model,
             model_name=model_name,
             base_path=save_path,
             model_pth_path=model_pth_path,
             dataset_root=dataset_root,
             dataset=dataset)




