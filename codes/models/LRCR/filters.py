
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2

from codes.models.LRCR.filters_functions import *

# --------------------
# https://github.com/yuanming-hu/exposure
# Exposure: A White-Box Photo Post-Processing Framework
# rewriting the operators by pytorch.
# --------------------

class Filter(nn.Module):

    def __init__(self):
        super(Filter, self).__init__()
        self.num_filter_params = None
        self.short_name = None
        self.filter_params = None

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_params(self):
        assert self.num_filter_params
        return self.num_filter_params

    def extract_params(self):
        pass

    def filter_param_regressor(self, features, **kwargs):

        assert False

    def process(self, img, param, **kwargs):

        assert False

    def debug_info_batches(self):
        return False

    def no_high_res(self):
        return False

    def visualize_filter(self, debug_info, canvas):
        assert False

    def draw_high_res_text(self, text, canvas):
        # save image
        cv2.putText(
            canvas,     # image
            text,       # image name
            (30, 128),  # 图片名字的位置
            cv2.FONT_HERSHEY_SIMPLEX,   # type of font
            0.8,        # size of font
            (0,0,0),    # color of font
            thickness=5
        )
        return canvas


class Exposure_Filter(Filter):

    def __init__(self):
        Filter.__init__()
        self.short_name = 'E'


    def process(self, img, param):
        if len(param.size()) == 4:
            out = img * torch.exp(param * np.log(2))
        else:
            out = img * torch.exp(param[:, :, None, None] * np.log(2))

        return out

    def visualize_filter(self, debug_info, canvas):

        exposure = debug_info['filter_params'][0]

        if canvas.shape[0] == 64:
            # cv2.rectangle：在任何图像上绘制矩形
            cv2.rectangle(canvas,
                          (8, 40),      # 起始坐标
                          (56, 52),     # 结束坐标
                          (1, 1, 1),    # 矩形边界线的颜色
                          cv2.FILLED)
            cv2.putText(canvas, 'EV %+.2f' % exposure, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
        else:
            self.draw_high_res_text('Exposure %+.2f' % exposure, canvas)


class Gamma_Filter(Filter):

    def __init__(self):
        Filter.__init__()
        self.short_name = 'G'


    def process(self, img, param):
        down = torch.zeros_like(img).cuda()  # 下限
        down = down + 0.001
        img = torch.where(img < torch.as_tensor(0.001).cuda(), down, img)

        if len(param.size()) == 2:
            out = torch.pow(img, param[:, :, None, None])  # param 是 2D tensor
        else:
            out = torch.pow(img, param)  # param 是4D tensor

        return out

    def visualize_filter(self, debug_info, canvas):
        gamma = debug_info['filter_params']
        cv2.rectangle(canvas,
                      (8, 40),      # start coordinates
                      (56, 52),     # end coordinates
                      (1, 1, 1),    # color
                      cv2.FILLED)
        cv2.putText(canvas, 'G 1%.2f' % (1.0 / gamma), (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


class Color_Filter(Filter):

    def __init__(self, curve_steps):
        Filter.__init__()
        self.curve_steps = curve_steps
        self.short_name = 'C'

    def process(self, img, param, color_curve_range=(0.90, 1.10),):
        param = param[:, None, None, None, :]
        param = tanh_range(*color_curve_range, initial=1)(param)

        param_sum = torch.sum(param, dim=4) + 1e-30
        total_img = img * 0

        for i in range(self.curve_steps):
            total_img += clip(x=img - 1.0 * i / self.curve_steps,
                              down=0.0,
                              down_value=0.0,
                              top=1.0 / self.curve_steps,
                              top_value=1.0 / self.curve_steps) * param[:, :, :, :, i]

        total_img *= self.curve_steps / param_sum

        return total_img



