

import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.models.vgg import vgg16
from codes.models.cnns.vgg import vgg16
from codes.utils import ops,ssim, pytorch_msssim, gaussianFilter



class Loss(nn.Module):
    '''
    联合损失
    '''
    def __init__(self, loss_type='l2_kl_per'):
        super(Loss, self).__init__()

        self.loss_type = loss_type

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


    def forward(self, out_images, target_images):
        '''
        定义各类损失
        :param out_images: 网络输出
        :param target_images: 标签
        :param is_first_out: 是否有first_out
        :param is_fusion_loss: 是否计算fusion_loss
        :return:
        '''
        loss = 0.0
        # 感知损失
        if 'per' in self.loss_type:
            perception_loss = self.mse_loss(self.loss_network(out_images['img_enhance']), self.loss_network(target_images['img_org']))
        # MSE损失
        if 'l2' in self.loss_type:
            l2_loss = self.mse_loss(out_images['img_enhance'], target_images['img_org'])
        # TV Loss
        if 'tv' in self.loss_type:
            tv_loss = self.tv_loss(out_images['img_enhance'])

        if 'supexp' in self.loss_type:
            supexp = self.mse_loss(out_images['img_enhance'], out_images['img_ll'])     # suppress exposure

        # SSIM Loss
        if 'ssim' in self.loss_type:
            ssim_val = ssim.ssim(out_images['img_enhance'], target_images['img_org']).item()  # 计算ssim
            ssim_loss = 1 - ssim_val

        if 'msssim' in self.loss_type:
            msssim_val = pytorch_msssim.msssim(out_images['img_enhance'], target_images['img_org']).item()
            msssim_loss = 1 - msssim_val

        # L1 Loss
        if 'l1' in self.loss_type:
            l1_loss = self.l1_loss(out_images['img_enhance'], target_images['img_org'])

        # KL losses
        if out_images['mu'] is not None and out_images['logvar'] is not None:
            kl_loss = ops.numerical_normalizaiton(self.KL_Loss(out_images['mu'], out_images['logvar']))
            if 'mlt' in self.loss_type:
                mu_log_tv = self.tv_loss(out_images['mu']) + self.tv_loss(out_images['logvar'])


        # 无参考损失
        # 空间一致性损失，通过保留输入图像与其增强版本之间相邻区域的差异，促进了增强图像的空间连贯性
        if 'spa' in self.loss_type:
            spa_loss = torch.mean(self.space_loss(out_images['img_enhance'], out_images['img_ll']))

        # 颜色一致性损失，通道中的颜色，在整张图中平均为灰色。该损失用于矫正输出图像中潜在的色差问题，同时建立调整三通道间的关系
        if 'color' in self.loss_type:
            color_loss = torch.mean(self.color_loss(out_images['img_enhance']))

        # 曝光控制损失，表示局部区域的平均强度值于正常曝光值E之间的差异。
        if 'exp' in self.loss_type:
            exp_loss = torch.mean(self.exp_loss(out_images['img_enhance']))

        if 'L2_de_en' in self.loss_type:
            noise_loss = self.mse_loss(out_images['de_noise'], target_images['img_ll'])
            illu_loss = self.mse_loss(out_images['img_enhance'], target_images['img_org'])
            return noise_loss + illu_loss

        if 'light' in self.loss_type:
            # light_map1 = self.gaussianFilter(target_images['img_org'].cpu())
            # light_map2 = self.gaussianFilter(out_images['img_enhance'].cpu())
            #
            # light_loss = self.mse_loss(light_map1.cuda(), light_map2.cuda())

            light_map1 = self.gaussianFilter(target_images['img_org'])
            light_map2 = self.gaussianFilter(out_images['img_enhance'])

            light_loss = self.mse_loss(light_map1, light_map2)

        if 'localM' in self.loss_type:
            localM1 = F.max_pool2d(target_images['img_org'], 16, 16)
            localM2 = F.max_pool2d(out_images['img_enhance'], 16, 16)

            localM_loss = self.mse_loss(localM1, localM2)

        if 'localMTV' in self.loss_type:
            localM2 = F.max_pool2d(out_images['img_enhance'], 16, 16)

            localMTV_loss = self.tv_loss(localM2)

        if 'localA' in self.loss_type:
            localA1 = F.avg_pool2d(target_images['img_org'], 16, 16)
            localA2 = F.avg_pool2d(out_images['img_enhance'], 16, 16)

            localA_loss = self.mse_loss(localA1, localA2)
        if 'localAM' in self.loss_type:
            localM1 = F.max_pool2d(target_images['img_org'], 16, 16)
            localM2 = F.max_pool2d(out_images['img_enhance'], 16, 16)

            localA1 = F.avg_pool2d(target_images['img_org'], 16, 16)
            localA2 = F.avg_pool2d(out_images['img_enhance'], 16, 16)

            localA_loss = self.mse_loss(localA1, localA2)
            localM_loss = self.mse_loss(localM1, localM2)

            localAM_loss = localA_loss + localM_loss

        if 'localAM2' in self.loss_type:
            localM1 = F.max_pool2d(target_images['img_org'], 16, 16)
            localM2 = F.max_pool2d(out_images['img_enhance'], 16, 16)

            localA1 = F.avg_pool2d(target_images['img_org'], 16, 16)
            localA2 = F.avg_pool2d(out_images['img_enhance'], 16, 16)

            localAM_loss2 = self.mse_loss(localA1+localM1, localA2+localM2)


        # ------------------------------------------------------
        # 基础的1项损失
        if self.loss_type == 'l2':
            return l2_loss
        elif self.loss_type == 'l1':
            return l1_loss

        # ------------------------------------------------------
        # 2项损失
        elif self.loss_type == 'l2_per':
            loss = l2_loss  + 0.006 * perception_loss
        elif self.loss_type == 'l2_kl':
            loss = l2_loss + kl_loss
        elif self.loss_type == 'l2_per_kl':
            loss = l2_loss  + 0.006 * perception_loss + kl_loss
        elif self.loss_type == 'l2_kl_tv':
            loss = l2_loss  + kl_loss + 2e-8 * tv_loss
        elif self.loss_type == 'l2_per_supexp_kl':
            loss = 0.9 * l2_loss  + 0.006 * perception_loss + kl_loss + 0.1 * supexp

        elif self.loss_type == 'l2_tv':
            # losses = l2_loss  + 2e-8 * tv_loss
            loss = l2_loss  + 0.006 * tv_loss
        elif self.loss_type == 'l2_ssim':
            loss = l2_loss  + ssim_loss
        elif self.loss_type == 'l2_color':
            loss = l2_loss  + 5 * color_loss

        elif self.loss_type == 'l1_per':
            loss = l1_loss  + 0.006 * perception_loss
        elif self.loss_type == 'l1_tv':
            loss = l1_loss  + 2e-8 * tv_loss
        elif self.loss_type == 'l1_ssim':
            loss = l1_loss  + ssim_loss
        # elif self.loss_type == 'l1_per_tv':
        # 	losses = l1_loss  + 0.1 * perception_loss + 0.1 * tv_loss

        # ------------------------------------------------------
        # 3项损失
        elif self.loss_type == 'l2_per_tv':
            loss = l2_loss  + 0.006 * perception_loss  + 2e-8 * tv_loss
        elif self.loss_type == 'l2_per_color':
            loss = l2_loss  + 0.006 * perception_loss  + 0.05 * color_loss
        elif self.loss_type == 'l2_per_exp':
            loss = l2_loss  + 0.006 * perception_loss  + 0.05 * exp_loss
        elif self.loss_type == 'l2_per_spa':
            loss = l2_loss  + 0.006 * perception_loss  + 0.05 * spa_loss
        elif self.loss_type == 'l2_per_ssim':
            loss = l2_loss  + 0.006 * perception_loss + ssim_loss

        elif self.loss_type == 'l2_per_color_exp':
            loss = l2_loss  + 0.006 * perception_loss  + color_loss + exp_loss

        elif self.loss_type == 'l1_per_tv':
            loss = l1_loss  + 0.006 *perception_loss  + 2e-8 * tv_loss
        elif self.loss_type == 'l1_per_ssim':
            loss = l1_loss  + 0.006 * perception_loss + ssim_loss
        elif self.loss_type == 'l1_per_msssim':
            loss = 0.16 * l1_loss  + 0.006 * perception_loss + 0.84*msssim_loss
            return loss, l1_loss, perception_loss, msssim_loss
        elif self.loss_type == 'l1_per_msssim_tv':
            loss = 0.16 * l1_loss  + 0.006 * perception_loss + 0.84*msssim_loss + 0.15*tv_loss
            return loss, l1_loss, perception_loss, msssim_loss, tv_loss

        elif self.loss_type == 'l1_per_msssim_tv_LR':
            # LR: laplacian regularizer
            loss = 0.16 * l1_loss  + 0.006 * perception_loss + 0.84*msssim_loss + 0.15*tv_loss
            lr = LaplacianRegularizer3D()
            lr_loss = lr(out_images[''])
            return loss, l1_loss, perception_loss, msssim_loss, tv_loss

        elif self.loss_type == 'l1_ssim_tv':
            loss = 0.85 * l1_loss + 0.1*ssim_loss + 0.05*tv_loss
        elif self.loss_type == 'l1_msssim':
            loss = 0.16 * l1_loss + 0.84*msssim_loss
        elif self.loss_type == 'l1_msssim_tv':
            loss = 0.85 * l1_loss + 0.1*msssim_loss + 0.05*tv_loss
        elif self.loss_type == 'l1_msssim_tv2':
            loss = 0.1 * l1_loss + 0.85*msssim_loss + 0.05*tv_loss
        elif self.loss_type == 'l1_msssim_tv3':
            loss = 0.5 * l1_loss + 0.4*msssim_loss + 0.1*tv_loss
        elif self.loss_type == 'l1_msssim_tv3_light':
            loss = 0.5 * l1_loss + 0.4 * msssim_loss + 0.1 * tv_loss + 0.1 * light_loss

        elif self.loss_type == 'l1_msssim_localM':
            loss = 0.5 * l1_loss + 0.25 * msssim_loss + 0.25 * localM_loss

        elif self.loss_type == 'l1_msssim_localMTV':
            loss = 0.5 * l1_loss + 0.25 * msssim_loss + 0.25 * localMTV_loss

        elif self.loss_type == 'l1_msssim_tv3_localM':
            loss = 0.5 * l1_loss + 0.4 * msssim_loss + 0.1 * tv_loss + 0.1 * localM_loss
        elif self.loss_type == 'l1_msssim_tv3_localAM':
            loss = 0.5 * l1_loss + 0.4 * msssim_loss + 0.1 * tv_loss + 0.1 * localAM_loss
        elif self.loss_type == 'l1_msssim_tv3_localAM2':
            loss = 0.5 * l1_loss + 0.4 * msssim_loss + 0.1 * tv_loss + 0.1 * localAM_loss2



        elif self.loss_type == 'l1_msssim_localA':
            loss = 0.5 * l1_loss + 0.25 * msssim_loss + 0.25 * localA_loss

        elif self.loss_type == 'l1_msssim_tv3_localA':
            loss = 0.5 * l1_loss + 0.4 * msssim_loss + 0.1 * tv_loss + 0.1 * localA_loss

        elif self.loss_type == 'l1_msssim_localAM':
            loss = 0.5 * l1_loss + 0.25 * msssim_loss + 0.25 * localAM_loss
        elif self.loss_type == 'l1_msssim_localAM2':
            loss = 0.5 * l1_loss + 0.25 * msssim_loss + 0.25 * localAM_loss2



        # ------------------------------------------------------
        # 4项损失
        elif self.loss_type == 'l2_per_tv_ssim':
            loss = l2_loss  + 0.006 * perception_loss  + 2e-8 * tv_loss + ssim_loss

        elif self.loss_type == 'l1_per_tv_ssim':
            loss = l1_loss  + 0.006 * perception_loss  + 2e-8 * tv_loss + ssim_loss

        elif self.loss_type == 'l2_per_tv_dc_per':
            loss = l2_loss + 0.006 * perception_loss + 2e-8 * tv_loss
            # de_cycle 损失
            dc_loss2 = self.mse_loss(self.loss_network(out_images['de_2']), self.loss_network(out_images['dc_2']))
            dc_loss3 = self.mse_loss(self.loss_network(out_images['de_3']), self.loss_network(out_images['dc_3']))
            dc_loss4 = self.mse_loss(self.loss_network(out_images['de_4']), self.loss_network(out_images['dc_4']))
            dc_loss5 = self.mse_loss(self.loss_network(out_images['de_5']), self.loss_network(out_images['dc_5']))
            return loss + dc_loss2 + dc_loss3 + dc_loss4 + dc_loss5


        elif self.loss_type == 'l2_kl_per_tv':
            # losses = l2_loss + kl_loss + 0.006 * perception_loss + 2e-8 * tv_loss   # 一开始采用的权重比
            loss = l2_loss + kl_loss + 0.006 * perception_loss + 2e-3 * tv_loss     # v91 4号实验
        elif self.loss_type == 'l2_kl_per_tv_mlt':
            # losses = l2_loss + kl_loss + 0.006 * perception_loss + 2e-8 * tv_loss   # 一开始采用的权重比
            loss = l2_loss + 0.1 * kl_loss + 0.006 * perception_loss + 2e-8 * tv_loss  +  0.1 * mu_log_tv# v91 4号实验


        return loss

    def KL_Loss(self, mu, log_var):
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
        # mean_sq = z_mean * z_mean
        # stddev_sq = z_stddev * z_stddev
        # 0.5 * torch.mean(mu + log_var - log_var.exp() - 1)
        return kld_loss

    # def grad_loss(self, input_r_low, input_r_high):
    # 	input_r_low_gray = tf.image.rgb_to_grayscale(input_r_low)
    # 	input_r_high_gray = tf.image.rgb_to_grayscale(input_r_high)
    # 	x_loss = tf.square(gradient(input_r_low_gray, 'x') - gradient(input_r_high_gray, 'x'))
    # 	y_loss = tf.square(gradient(input_r_low_gray, 'y') - gradient(input_r_high_gray, 'y'))
    # 	grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    # 	return grad_loss_all


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


