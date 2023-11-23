

import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.models.vgg import vgg16
from codes.models.cnns.vgg import vgg16
from codes.utils import ops,ssim, pytorch_msssim, gaussianFilter

import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


class Loss(nn.Module):
    '''
    '''
    def __init__(self, loss_type='l2_kl_per', cfg=None):
        super(Loss, self).__init__()

        self.loss_type = loss_type
        self.cfg = cfg

        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.loss_network.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.mse_loss = nn.MSELoss()
        self.smooth_L1 = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss()
        self.color_loss = L_color()
        self.space_loss = L_spa()
        self.exp_loss = L_exp(10, 0.6)
        self.gaussianFilter = gaussianFilter.get_gaussian_kernel(128, 10)
        self.contrast_loss = ContrastLoss()

    def forward(self, input, GT):
        '''

        :param input:
        :param GT:
        :return:
        '''
        loss = 0.0
        if self.loss_type == 'l1':
            l1 = self.l1_loss(input['img_enhance'], GT['img_org'])
            loss = l1
            return loss, l1

        elif self.loss_type == 'rec_tex_vis':
            rec1 = self.mse_loss(input['img_enhance1'], GT['img_org'])
            tex1 = self.mse_loss(self.loss_network(input['img_enhance1']), self.loss_network(GT['img_org']))
            vis1 = 1 - pytorch_msssim.msssim(input['img_enhance1'], GT['img_org']).item()

            rec2 = self.mse_loss(input['img_enhance2'], GT['img_org_en'])
            tex2 = self.mse_loss(self.loss_network(input['img_enhance2']), self.loss_network(GT['img_org_en']))
            vis2 = 1 - pytorch_msssim.msssim(input['img_enhance2'], GT['img_org_en']).item()

            loss = rec1 + vis1 + 0.35 * tex1 + \
                   rec2 + vis2 + 0.35 * tex2

            return loss, rec1, vis1, tex1, rec2, tex2, vis2

        else:
            return loss

    def KL_Loss(self, mu, log_var):
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
        # mean_sq = z_mean * z_mean
        # stddev_sq = z_stddev * z_stddev
        # 0.5 * torch.mean(mu + log_var - log_var.exp() - 1)
        return kld_loss

    def color_map(self, img):
        color_map = nn.functional.avg_pool2d(img, 11, 1, 5)  # kernel size=11, strid=1, padding=5
        color_map = color_map / torch.sum(color_map, 1, keepdim=True)

        return color_map


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class L_color(nn.Module):
    '''
    基于一个假设：每个通道中的颜色在整个图像上平均为灰色。
    设计了颜色保持损失，用于矫正潜在的色差问题，同时构建调整三个通道颜色的关系。
    '''
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)  # 求均值
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)  # 分离三通道
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    '''
    空间一致性损失主要保持增强后的图像仍然能够保持区域间的差异
    （这个差异应与输入图像的区域间差异一致），促进增强后图像的空间连贯性。
    '''
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

class L_exp(nn.Module):
    '''
    为了抑制 under/over 曝光区域，提出了曝光控制损失用于控制曝光水平，
    计算局部（16x16）平均强度值与适当曝光水平E（E=0.6）之间的均方误差，
    将E设置为RGB颜色空间的灰度水平
    '''
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d

class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()

    # print(1)
    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k

class LaplacianRegularizer2D(nn.Module):
    def __init__(self):
        super(LaplacianRegularizer2D, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
    def forward(self,f):
        loss = 0.
        for i in range(f.shape[2]):
            for j in range(f.shape[3]):
                up = max(i-1,0)
                down = min(i+1,f.shape[2] - 1)
                left = max(j-1,0)
                right = min(j+1,f.shape[3] - 1)
                term = f[:,:,i,j].view(f.shape[0],f.shape[1],1,1).expand(f.shape[0],f.shape[1],down - up+1,right-left+1)
                loss += self.mse_loss(term,f[:,:,up:down+1,left:right+1])
        return loss

class LaplacianRegularizer3D(nn.Module):
    def __init__(self):
        super(LaplacianRegularizer3D, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, f):
        loss = 0.
        # f: [B, 12, 8, H, W]
        f = f.reshape(f.shape[0],12,8,f.shape[2],f.shape[-1])
        B, C, D, H, W = f.shape
        for k in range(D):
            for i in range(H):
                for j in range(W):
                    front = max(k - 1, 0)
                    back = min(k + 1, D - 1)
                    up = max(i - 1, 0)
                    down = min(i + 1, H - 1)
                    left = max(j - 1, 0)
                    right = min(j + 1, W - 1)
                    term = f[:, :, k, i, j].view(B, C, 1, 1, 1).expand(B, C, back - front + 1, down - up + 1, right - left + 1)
                    loss += self.mse_loss(term, f[:, :, front:back + 1, up:down + 1, left:right + 1])
        return loss



# ------------------------------
# 2022.7.2
# 参考：Wu H, Qu Y, Lin S, et al. Contrastive learning for compact single image dehazing[C]
# //Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 10551-10560.
# https://github.com/GlassyWu/AECR-Net
# ------------------------------

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.chs = [64, 128, 256, 512, 512]
        self.fn = 5

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

# ------------------------------------------------------------------------

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

# ------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    def __init__(self,):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0

        per = 0
        for i in range(len(x_vgg)):
            per = self.l1(x_vgg[i], y_vgg[i].detach())
            loss += self.weights[i] * per
        return loss
























