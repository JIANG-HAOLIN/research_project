import math
import numpy as np
import random

import torch
from torch import nn
from torch.nn import functional as F

from models.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

import gc

from models.clade_encoder_decoder import encoder_decoder
from models.trans_encoder import trans_encoder,Img2Token
##ganformer begin
from tqdm import trange
# from models.ganformer.networks import Generator
from models.ganformer import NetworkS as Ganformernetworks
import torch



class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class PixelNorm_multi_class(nn.Module):
    def __init__(self,latent_size,num_classes):
        super().__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
    def forward(self, input):
        input = input.view(input.shape[0],self.num_classes,self.latent_size)
        return (input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)).view(input.shape[0],-1,1)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class separate_mapping_layer(nn.Module):
    def __init__(self, style_dim, num_classes):
        super(separate_mapping_layer,self).__init__()
        self.weight = nn.Parameter(torch.randn(num_classes,style_dim,style_dim))
        self.bias = nn.Parameter(torch.zeros(style_dim))

    def forward(self, input):
        out = torch.einsum('bli,loi -> blo',input,self.weight) + self.bias
        return out



class EqualConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True,groups=35):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(groups*out_channel, in_channel, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.zeros(groups*out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv1d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            # self.weight = self.weight.to('cuda')
            # print(input.device, self.weight.device, )
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        approach= -1,
        add_dist = False
    ):
        super().__init__()
        self.approach = approach
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.add_dist = add_dist

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        if self.approach == 0:
            self.weight = nn.Parameter(
                torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
            )
            self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
            self.clade_weight_modulation = EqualLinear(style_dim, self.out_channel, bias_init=1) #jhl
            self.clade_bias_modulation = EqualLinear(style_dim, self.out_channel, bias_init=1) #jhl
        elif self.approach == 1:
            self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
            self.clade_weight_modulation = EqualLinear(style_dim*2, self.out_channel, bias_init=1)  # jhl
            self.clade_bias_modulation = EqualLinear(style_dim*2, self.out_channel, bias_init=1)  # jhl
        elif self.approach == 1.3 :
            self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
            self.encoder_decoder_w = encoder_decoder()
            self.encoder_decoder_b = encoder_decoder()
        elif self.approach == 1.4 :##clade weight modulation
            if self.out_channel != 3 :
                self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
                self.clade_weight_modulation = EqualLinear(style_dim * 2, 512, bias_init=1)  # jhl
                self.clade_bias_modulation = EqualLinear(style_dim * 2, 512, bias_init=1)  # jhl
                self.i2t = Img2Token(dim=self.out_channel)
                self.w_encoder = nn.TransformerEncoderLayer(d_model=self.out_channel,nhead=8)
                self.b_encoder = nn.TransformerEncoderLayer(d_model=self.out_channel,nhead=8)
            else:
                self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
                self.clade_weight_modulation = EqualLinear(style_dim * 2,self.out_channel, bias_init=1)  # jhl
                self.clade_bias_modulation = EqualLinear(style_dim * 2,self.out_channel, bias_init=1)  # jhl
        elif self.approach == 1.5 :##clade weight modulation
            if self.out_channel != 3 :
                self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
                self.clade_weight_modulation = EqualLinear(style_dim * 2, 512, bias_init=1)  # jhl
                self.clade_bias_modulation = EqualLinear(style_dim * 2, 512, bias_init=1)  # jhl
                self.i2t = Img2Token(dim=self.out_channel)
                self.w_encoder = nn.TransformerEncoderLayer(d_model=self.out_channel,nhead=8)
                self.b_encoder = nn.TransformerEncoderLayer(d_model=self.out_channel,nhead=8)
            else:
                self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
                self.clade_weight_modulation = EqualLinear(style_dim * 2,self.out_channel, bias_init=1)  # jhl
                self.clade_bias_modulation = EqualLinear(style_dim * 2,self.out_channel, bias_init=1)  # jhl
        elif self.approach == 1.6:
            if self.out_channel != 3:
                self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
                self.clade_weight_modulation = EqualLinear(style_dim * 2, 512, bias_init=1)  # jhl
                self.clade_bias_modulation = EqualLinear(style_dim * 2, 512, bias_init=1)  # jhl
            else:
                self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
                self.clade_weight_modulation = EqualLinear(style_dim * 2, self.out_channel, bias_init=1)  # jhl
                self.clade_bias_modulation = EqualLinear(style_dim * 2, self.out_channel, bias_init=1)  # jhl

        elif self.approach == 1.7:

            self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
            self.clade_weight_modulation = EqualLinear(style_dim * 2, self.out_channel, bias_init=1)  # jhl
            self.clade_bias_modulation = EqualLinear(style_dim * 2, self.out_channel, bias_init=1)  # jhl
        elif self.approach == 'SPADE_like':
            if self.out_channel == 3:
                self.param_free_norm = nn.InstanceNorm2d(self.out_channel, affine=False)  ##jhl
                self.spade_weight_modulation = EqualLinear(self.in_channel, self.out_channel, bias_init=1) #jhl
                self.spade_bias_modulation = EqualLinear(self.in_channel, self.out_channel, bias_init=1) #jhl
            else:
                self.param_free_norm = nn.InstanceNorm2d(self.out_channel, affine=False)  ##jhl
                self.spade_weight_modulation = EqualLinear(self.out_channel, self.out_channel, bias_init=1) #jhl
                self.spade_bias_modulation = EqualLinear(self.out_channel, self.out_channel, bias_init=1) #jhl
        elif self.approach == 'SPADE_like_modulation':
            if self.out_channel == 3:
                self.latent_modulation = EqualLinear(style_dim, self.in_channel, bias_init=1)
                self.param_free_norm = nn.InstanceNorm2d(self.out_channel, affine=False)  ##jhl
                self.spade_weight_modulation = EqualLinear(self.in_channel, self.out_channel, bias_init=1) #jhl
                self.spade_bias_modulation = EqualLinear(self.in_channel, self.out_channel, bias_init=1) #jhl
            else:
                self.latent_modulation = EqualLinear(style_dim, self.out_channel, bias_init=1)
                self.param_free_norm = nn.InstanceNorm2d(self.out_channel, affine=False)  ##jhl
                self.spade_weight_modulation = EqualLinear(self.out_channel, self.out_channel, bias_init=1) #jhl
                self.spade_bias_modulation = EqualLinear(self.out_channel, self.out_channel, bias_init=1) #jhl
        elif self.approach == 'SPADE_like_multi_class_modulation':
            if self.out_channel == 3:
                self.latent_modulation = EqualLinear(128, self.in_channel, bias_init=1)
                self.param_free_norm = nn.InstanceNorm2d(self.out_channel, affine=False)  ##jhl
                self.spade_weight_modulation = EqualLinear(self.in_channel, self.out_channel, bias_init=1) #jhl
                self.spade_bias_modulation = EqualLinear(self.in_channel, self.out_channel, bias_init=1) #jhl
            else:
                self.latent_modulation = EqualLinear(128, self.out_channel, bias_init=1)
                self.param_free_norm = nn.InstanceNorm2d(self.out_channel, affine=False)  ##jhl
                self.spade_weight_modulation = EqualLinear(self.out_channel, self.out_channel, bias_init=1) #jhl
                self.spade_bias_modulation = EqualLinear(self.out_channel, self.out_channel, bias_init=1) #jhl
        elif self.approach == 'styleGan_like':
            if self.out_channel == 3:
                self.latent_modulation = EqualLinear(style_dim, self.in_channel, bias_init=1)
                self.instance_norm = nn.InstanceNorm2d(self.out_channel, affine=True)  ##jhl
                self.weight_modulation = EqualLinear(self.in_channel*2, self.in_channel, bias_init=1) #jhl
            else:
                self.latent_modulation = EqualLinear(style_dim, self.out_channel, bias_init=1)
                self.instance_norm = nn.InstanceNorm2d(self.out_channel, affine=True)  ##jhl
                self.weight_modulation = EqualLinear(self.out_channel*2, self.in_channel, bias_init=1) #jhl

        elif self.approach == 2:
            pass
        else:
            pass

        if self.add_dist :
            if (    self.approach == 0 or
                    self.approach == 1 or
                    self.approach == 1.3 or
                    self.approach ==1.4 or
                    self.approach==1.5 or
                    self.approach==1.6
            ) :
                self.dist_conv_w = nn.Conv2d(2, 1, kernel_size=1, padding=0)
                nn.init.zeros_(self.dist_conv_w.weight)
                nn.init.zeros_(self.dist_conv_w.bias)
                self.dist_conv_b = nn.Conv2d(2, 1, kernel_size=1, padding=0)
                nn.init.zeros_(self.dist_conv_b.weight)
                nn.init.zeros_(self.dist_conv_b.bias)

            elif self.approach ==2 :
                pass
            else:
                pass




    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style,label_class_dict=None,label = None,
                class_style = None,dist_map = None, clade_param = None,style_conv_dict=None):
        batch, in_channel, height, width = input.shape

        # style = self.modulation(style).view(batch, 1, in_channel, 1, 1)##[1,1,1024,1,1,]
        # ###modulation(style):35x512=>35x1024
        # weight = self.scale * self.weight * style
        # ##[1,512,1024,1,1]
        # if self.demodulate:
        #     demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        #     weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        #     ##[1,512,1024,1,1]
        # weight = weight.view(
        #     batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        # )##[1,512,1024,1,1]

        if self.upsample:
            print('upsample')
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
            ###modulation(style):35x512=>35x1024
            weight = self.scale * self.weight * style
            ##[1,512,1024,1,1]
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
                ##[1,512,1024,1,1]
            weight = weight.view(
                batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )  ##[1,512,1024,1,1]


            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)


        elif self.downsample:
            print('downsample')
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
            ###modulation(style):35x512=>35x1024
            weight = self.scale * self.weight * style
            ##[1,512,1024,1,1]
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
                ##[1,512,1024,1,1]
            weight = weight.view(
                batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )  ##[1,512,1024,1,1]


            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:

            # ###pixel interation
            # input = input.reshape(height, width, batch,1, in_channel)
            # out   = []
            # for h in range(height):
            #     for w in range(width):
            #         print(h)
            #         pixel_input = input[h][w].reshape(1, batch * in_channel, 1, 1)
            #         LUT=int(label_class_dict[h][w]-1)
            #         class_style = style[LUT]
            #         class_style = self.modulation(class_style).view(batch, 1, in_channel, 1, 1)
            #         weight = self.scale * self.weight * class_style
            #         if self.demodulate:
            #             demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            #             weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
            #             ##[1,512,1024,1,1]
            #         weight = weight.view(
            #             batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)  ##[1,512,1024,1,1]
            #         pixel_input = pixel_input.reshape(1, batch * in_channel, 1, 1)
            #
            #         out.append(F.conv2d(pixel_input, weight, padding=self.padding, groups=batch))
            # out = torch.Tensor(out)
            #         # print(out)
            # _,height, width, _ , _ = out.shape
            # out = out.view(batch, self.out_channel, height, width)


            # ###class iteration
            # out = torch.zeros(batch, self.out_channel, height, width).cuda(0)
            # # style = style.reshape(35,512)
            # for class_index in range(35):
            #     print(class_index)
            #     class_map = label_class_dict
            #     class_map[class_map != class_index] = -1.000
            #     class_map[class_map == class_index] = 1.000
            #     class_map[class_map == -1] = 0.000
            #     class_map = class_map.view(batch, 1, 256, 512).cuda(0)
            #     input = (input * class_map).cuda(0)
            #     class_style = style[class_index].view(batch, 1, 512).cuda(0)
            #     class_style = self.modulation(class_style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
            #     ###modulation(style):35x512=>35x1024
            #     weight1 = self.scale * self.weight * class_style.cuda(0)
            #     ##[1,512,1024,1,1]
            #     if self.demodulate:
            #         demod = torch.rsqrt(weight1.pow(2).sum([2, 3, 4]) + 1e-8)
            #     weight1 = weight1 * demod.view(batch, self.out_channel, 1, 1, 1).cuda(0)
            #     ##[1,512,1024,1,1]
            #     weight1 = weight1.view(
            #         batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)  ##[1,512,1024,1,1]
            #     print(input.device,input.double().device,weight1.device,weight1.double().device,out.device)
            #     input = input.view(1, batch * in_channel, height, width)
            #     out = out + F.conv2d(input.double().cuda(0), weight1.double().cuda(0), padding=self.padding, groups=batch)
            #     del weight1

            #     gc.collect()
            #     print("xxxxxxxxxxxxxxxxx")
            # _, _, height, width = out.shape
            # out = out.view(batch, self.out_channel, height, width)


            ## Modulation+CLADE layer
            if self.approach == 'SPADE_like':
                weight = self.weight.view(self.out_channel, self.in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)

                spade_weight = self.spade_weight_modulation(style_conv_dict.permute(0,2,3,1)).permute(0,3,1,2)
                spade_bias = self.spade_bias_modulation(style_conv_dict.permute(0,2,3,1)).permute(0,3,1,2)
                out = out * spade_weight + spade_bias
                return out
            elif self.approach == 'SPADE_like_modulation':
                weight = self.weight.view(self.out_channel, self.in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)
                style = self.latent_modulation(style).view(batch,-1,1,1)
                style_conv_dict = style_conv_dict*style
                style_conv_dict = style_conv_dict*torch.rsqrt(torch.sum(style_conv_dict.pow(2),dim=1,keepdim=True)+1e-8)
                spade_weight = self.spade_weight_modulation(style_conv_dict.permute(0,2,3,1)).permute(0,3,1,2)
                spade_bias = self.spade_bias_modulation(style_conv_dict.permute(0,2,3,1)).permute(0,3,1,2)
                out = out * spade_weight + spade_bias
                return out
            elif self.approach == 'SPADE_like_multi_class_modulation':
                weight = self.weight.view(self.out_channel, self.in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)
                style = self.latent_modulation(style)
                style = torch.einsum('nci,nchw->nihw', style, label)
                style_conv_dict = style_conv_dict*style
                style_conv_dict = style_conv_dict*torch.rsqrt(torch.sum(style_conv_dict.pow(2),dim=1,keepdim=True)+1e-8)
                spade_weight = self.spade_weight_modulation(style_conv_dict.permute(0,2,3,1)).permute(0,3,1,2)
                spade_bias = self.spade_bias_modulation(style_conv_dict.permute(0,2,3,1)).permute(0,3,1,2)
                out = out * spade_weight + spade_bias
                return out
            elif self.approach == 'styleGan_like':
                weight = self.weight.view(self.out_channel, self.in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                style = self.latent_modulation(style).view(batch,-1,1,1).expand(-1,-1,height,width)
                style_conv_dict = self.weight_modulation(torch.cat([style_conv_dict,style],dim=1).permute(0,2,3,1)).permute(0,3,1,2)
                weight = self.scale * weight.unsqueeze(0) * style_conv_dict.unsqueeze(1)
                # style_conv_dict = style_conv_dict*torch.rsqrt(torch.sum(style_conv_dict.pow(2),dim=1,keepdim=True)+1e-8)
                demod = torch.rsqrt(torch.sum(weight.pow(2),dim=2,keepdim=True)+1e-8)
                weight = weight * demod
                out = torch.einsum('nihw,noihw->nohw', input, weight)
                return out

            elif self.approach == 0:
                style = self.modulation(style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
                ###modulation(style):35x512=>35x1024
                weight = self.scale * self.weight * style
                ##[1,512,1024,1,1]
                if self.demodulate:
                    demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                    weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
                ##[1,512,1024,1,1]
                weight = weight.view(
                    batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)  ##[1,512,1024,1,1]

                input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight, padding=self.padding, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)
                ##apply CLADE layer
                clade_weight = self.clade_weight_modulation(class_style)
                clade_bias = self.clade_bias_modulation(class_style)
                out = self.param_free_norm(out)
                clade_weight = F.embedding(label_class_dict.long(), clade_weight).permute(0, 3, 1, 2)
                #before permute:[n, h, w, c] after permute [n, c, h, w]
                clade_bias = F.embedding(label_class_dict.long(), clade_bias).permute(0, 3, 1, 2)
                # before permute:[n, h, w, c] after permute [n, c, h, w]

                if self.add_dist:
                    # input_dist = F.interpolate(dist_map, size=input.size()[2:], mode='nearest')
                    # class_weight = class_weight * (1 + self.dist_conv_w(input_dist))
                    # class_bias = class_bias * (1 + self.dist_conv_b(input_dist))

                    # input_dist = dist_map
                    # class_weight= class_weight.to('cpu')
                    # class_bias= class_bias.to('cpu')
                    # alpha_weight = (1 + self.dist_conv_w(input_dist)).to('cpu')
                    # print(alpha_weight.shape)
                    # alpha_bias = 1 + self.dist_conv_b(input_dist).to('cpu')
                    # class_weight = (class_weight * alpha_weight).to('cuda')
                    # class_bias = (class_bias * alpha_bias).to('cuda')

                    clade_weight = (clade_weight * (1 + self.dist_conv_w(dist_map)))
                    clade_bias = (clade_bias * (1 + self.dist_conv_b(dist_map)))

                out = out * clade_weight + clade_bias


            ##class_label_dict = [N, H, W]

            ## use only CLADE layer, no mod or demod
            elif self.approach == 1:
                # print('approach 1 activated')
                weight = self.weight.view(self.out_channel,in_channel,self.kernel_size,self.kernel_size) ##[512,1024,1,1]

                # print(input.shape,weight.shape)


                # input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight, padding=self.padding)
                # _, _, height, width = out.shape
                # out = out.view(batch, self.out_channel, height, width)

                out = self.param_free_norm(out)

                # print(torch.cuda.memory_allocated())
                # print(torch.cuda.memory_reserved())

                style = style.view(batch, 1, 512).expand(batch, 35, 512)
                # class_style = class_style.view(1, 35, 512).expand(batch, 35, 512)
                # style = torch.cat((style, class_style), dim=2).view(batch*35, 1024)
                style = torch.cat((style,class_style.view(1, 35, 512).expand(batch, 35, 512)), dim=2).view(batch * 35, 1024)
                # print(torch.cuda.memory_allocated())
                # print(torch.cuda.memory_reserved())


                clade_weight = self.clade_weight_modulation(style).view(batch, 35, self.out_channel)
                clade_bias = self.clade_bias_modulation(style).view(batch, 35, self.out_channel)
                clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)


                # print(torch.cuda.memory_allocated())
                # print(torch.cuda.memory_reserved())



                if self.add_dist:

                    # input_dist = F.interpolate(dist_map, size=input.size()[2:], mode='nearest')
                    # class_bias = class_bias * (1 + self.dist_conv_b(input_dist))



                    # input_dist = dist_map
                    # class_weight = class_weight.to('cpu')
                    # class_bias = class_bias.to('cpu')
                    # alpha_weight = (1 + self.dist_conv_w(input_dist)).to('cpu')
                    # print(alpha_weight.shape)
                    # alpha_bias = 1 + self.dist_conv_b(input_dist).to('cpu')
                    # class_weight = (class_weight * alpha_weight).to('cuda')
                    # class_bias = (class_bias * alpha_bias).to('cuda')


                    clade_weight = (clade_weight * (1 + self.dist_conv_w(dist_map)))
                    clade_bias = (clade_bias * (1 + self.dist_conv_b(dist_map)))

                    # print(torch.cuda.memory_allocated())
                    # print(torch.cuda.memory_reserved())

                out = out * clade_weight + clade_bias


                torch.cuda.empty_cache()

                # print(torch.cuda.memory_allocated())
                # print(torch.cuda.memory_reserved())

            elif self.approach == 1.3:
                # print('approach 1 activated')
                weight = self.weight.view(self.out_channel, in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)

                # style = style.view(batch, 1, 512).expand(batch, 35, 512)
                # style = torch.cat((style, class_style.view(1, 35, 512).expand(batch, 35, 512)), dim=2).view(batch * 35,1024)
                # clade_weight = self.clade_weight_modulation(style).view(batch, 35, self.out_channel)
                # clade_bias = self.clade_bias_modulation(style).view(batch, 35, self.out_channel)
                # clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                # clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                # if self.add_dist:
                #     clade_weight = (clade_weight * (1 + self.dist_conv_w(dist_map)))
                #     clade_bias = (clade_bias * (1 + self.dist_conv_b(dist_map)))
                # out = out * clade_weight + clade_bias

                DE_w, DE_w_list = self.encoder_decoder_w(label)
                DE_b, DE_b_list = self.encoder_decoder_b(label)


                if self.add_dist:
                    DE_w = (DE_w * (1 + self.dist_conv_w(dist_map)))
                    DE_b = (DE_b * (1 + self.dist_conv_b(dist_map)))
                out = DE_w*out + DE_b
                torch.cuda.empty_cache()

            elif self.approach == 1.4:
                # print('approach 1 activated')
                weight = self.weight.view(self.out_channel, in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)
                style = style.view(batch, 1, 512).expand(batch, 35, 512)
                style = torch.cat((style, class_style.view(1, 35, 512).expand(batch, 35, 512)), dim=2)

                if self.out_channel != 3:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, 512)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, 512)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                    label_token = self.i2t(label)
                    w_modulation = self.w_encoder(label_token).view(self.out_channel,512,1,1)
                    b_modulation = self.b_encoder(label_token).view(self.out_channel,512,1,1)
                    clade_weight = F.conv2d(clade_weight,w_modulation)
                    clade_bias = F.conv2d(clade_bias,b_modulation)
                else:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, self.out_channel)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, self.out_channel)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                if self.add_dist:
                    clade_weight = (clade_weight * (1 + self.dist_conv_w(dist_map)))
                    clade_bias = (clade_bias * (1 + self.dist_conv_b(dist_map)))
                out = out * clade_weight + clade_bias

            elif self.approach == 1.5:
                # print('approach 1 activated')
                weight = self.weight.view(self.out_channel, in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)
                style = torch.cat((style, class_style.view(1, 35, 512).expand(batch, 35, 512)), dim=2)

                if self.out_channel != 3:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, 512)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, 512)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                    label_token = self.i2t(label)
                    w_modulation = self.w_encoder(label_token).view(self.out_channel,512,1,1)
                    b_modulation = self.b_encoder(label_token).view(self.out_channel,512,1,1)
                    clade_weight = F.conv2d(clade_weight,w_modulation)
                    clade_bias = F.conv2d(clade_bias,b_modulation)
                else:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, self.out_channel)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, self.out_channel)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                if self.add_dist:
                    clade_weight = (clade_weight * (1 + self.dist_conv_w(dist_map)))
                    clade_bias = (clade_bias * (1 + self.dist_conv_b(dist_map)))
                out = out * clade_weight + clade_bias



            elif self.approach == 1.6:
                # print('approach 1 activated')
                weight = self.weight.view(self.out_channel, in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)
                style = style.view(batch, 1, 512).expand(batch, 35, 512)
                style = torch.cat((style, class_style.view(1, 35, 512).expand(batch, 35, 512)), dim=2)

                if self.out_channel != 3:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, 512)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, 512)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                    w_modulation = clade_param[0].view(self.out_channel,512,1,1)
                    b_modulation = clade_param[1].view(self.out_channel,512,1,1)
                    clade_weight = F.conv2d(clade_weight,w_modulation)
                    clade_bias = F.conv2d(clade_bias,b_modulation)
                else:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, self.out_channel)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, self.out_channel)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                if self.add_dist:
                    clade_weight = (clade_weight * (1 + self.dist_conv_w(dist_map)))
                    clade_bias = (clade_bias * (1 + self.dist_conv_b(dist_map)))
                out = out * clade_weight + clade_bias

            elif self.approach == 1.61:
                # print('approach 1 activated')
                weight = self.weight.view(self.out_channel, in_channel, self.kernel_size,
                                          self.kernel_size)  ##[512,1024,1,1]
                out = F.conv2d(input, weight, padding=self.padding)
                out = self.param_free_norm(out)
                style = style.view(batch, 1, 512).expand(batch, 35, 512)
                style = torch.cat((style, class_style.view(1, 35, 512).expand(batch, 35, 512)), dim=2)

                if self.out_channel != 3:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, 512)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, 512)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                    w_modulation = clade_param[0].view(self.out_channel,512,1,1)
                    b_modulation = clade_param[1].view(self.out_channel,512,1,1)
                    clade_weight = F.conv2d(clade_weight,w_modulation)
                    clade_bias = F.conv2d(clade_bias,b_modulation)
                else:
                    clade_weight = self.clade_weight_modulation(style).view(batch, 35, self.out_channel)
                    clade_bias = self.clade_bias_modulation(style).view(batch, 35, self.out_channel)
                    clade_weight = torch.einsum('nic,nihw->nchw', clade_weight, label)
                    clade_bias = torch.einsum('nic,nihw->nchw', clade_bias, label)
                out = out * clade_weight + clade_bias






            ## brutal Matrix Computation approach
            elif self.approach == 2:
                # print('apply approach 2 : Matrix Computation')
                style = style.view(batch, 1, 512)
                class_style = class_style.view(1, 35, 512)
                style_addition = style + class_style ##[N, 35, 512]
                style_addition = style_addition.view(batch*35, 512)
                style_dict = self.modulation(style_addition).view(batch, 35, self.in_channel)

                label = F.interpolate(label,size=(64,128),mode='nearest')

                pixel_class_style = torch.einsum('nci,nchw->nihw',style_dict,label)
                ###此处必须einsum,若用embedding,label必须为二维map，不能有batchsize这个维度！！
                # pixel_class_style = F.embedding(label_class_dict, style_init).permute(0, 3, 1, 2)  # [n, c, h, w]
                weight = self.weight.view(self.out_channel, self.in_channel)

                weight = weight.to('cpu')
                pixel_class_style = pixel_class_style.to('cpu')

                weight_per_pixel = self.scale * (torch.einsum('oi,nihw->noihw',weight , pixel_class_style))

                weight_per_pixel = weight_per_pixel.to('cuda')

                weight_per_pixel = F.interpolate(weight_per_pixel,size=(256,512),mode='nearest')
                if self.demodulate:
                    demod = torch.rsqrt(torch.sum(weight_per_pixel.pow(2), dim=2, keepdim=True) + 1e-8)
                    weight_per_pixel = weight_per_pixel * demod
                out = torch.einsum('nihw,noihw->nohw', input, weight_per_pixel)
                print('good')


        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def  __init__(self, channel, size=None,):
        super().__init__()
        # print(size)
        self.learnable_vectors = nn.Parameter(torch.randn(1, channel, size[0], size[1]))
        # print(self.learnable_vectors.shape)

    def forward(self, input):
        batch = input.shape[0]
        out = self.learnable_vectors.repeat(batch, 1, 1, 1)
        # print(out.shape)
        ##output = [batch,channel,size.512]
        return out



class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        activation=None,
        downsample=False,
        approach=-1,
        add_dist = False
    ):
        super().__init__()
        self.add_dist = add_dist
        self.approach = approach
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            downsample=downsample,
            approach=self.approach,
            add_dist= self.add_dist
        )

        self.activation = activation
        self.noise = NoiseInjection()
        if activation == 'sinrelu':
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
            self.activate = ScaledLeakyReLUSin()
        elif activation == 'sin':
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
            self.activate = SinActivation()
        else:
            self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style= None, noise=None,label_class_dict=None,
                label=None,class_style=None,dist_map = None,
                clade_param = None,style_conv_dict=None):
        out = self.conv(input, style,label_class_dict=label_class_dict,
                        label=label,class_style=class_style,dist_map=dist_map,
                        clade_param = clade_param,style_conv_dict=style_conv_dict)
        out = self.noise(out, noise=noise)
        if self.activation == 'sinrelu' or self.activation == 'sin':
            out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1],approach=-1,add_dist = False):#jhl
        super().__init__()
        self.add_dist = add_dist
        self.approach = approach
        self.upsample = upsample
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False,
                                    approach=self.approach,add_dist=self.add_dist)#jhl
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style=None, skip=None,label_class_dict=None,label=None,
                class_style=None,dist_map=None,style_conv_dict=None):
        out = self.conv(input, style,
                        label_class_dict=label_class_dict,label=label,class_style=class_style,dist_map=dist_map,style_conv_dict=style_conv_dict)
        out = out + self.bias

        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out


class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        upsample=False,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], kernel_size=3, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
        self.conv2 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        # print('initialsation:',self.conv.weight.device)

        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        # print('forwarding',self.conv.weight.device,x.device)
        return self.conv(x)


class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class LFF(nn.Module):
    def __init__(self, hidden_size, ):
        super(LFF, self).__init__()
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x


class ScaledLeakyReLUSin(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out_lr = F.leaky_relu(input[:, ::2], negative_slope=self.negative_slope)
        out_sin = torch.sin(input[:, 1::2])
        out = torch.cat([out_lr, out_sin], 1)
        return out * math.sqrt(2)


class StyledResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, blur_kernel=[1, 3, 3, 1], demodulate=True,
                 activation=None, upsample=False, downsample=False):
        super().__init__()

        self.conv1 = StyledConv(in_channel, out_channel, kernel_size, style_dim,
                                demodulate=demodulate, activation=activation)
        self.conv2 = StyledConv(out_channel, out_channel, kernel_size, style_dim,
                                demodulate=demodulate, activation=activation,
                                upsample=upsample, downsample=downsample)

        if downsample or in_channel != out_channel or upsample:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False, upsample=upsample,
            )
        else:
            self.skip = None

    def forward(self, input, latent):
        out = self.conv1(input, latent)
        out = self.conv2(out, latent)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out = (out + skip) / math.sqrt(2)

        return out

## @jhl new

# class upscale_Interpolate(nn.Module):
#     def __init__(self, size, mode):
#         super(upscale_Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode
#
#     def forward(self, x):
#         x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
#         return x

def make_dist_train_val_cityscapes_datasets(mask_batch=None,dir = '/home/tzt/dataset/cityscapes/',norm='norm'):
    # label_dir = os.path.join(dir, 'gtFine')
    # phases = ['val','train']
    # for phase in phases:
    #     if 'test' in phase:
    #         continue
    #     print('process',phase,'dataset')
    #     citys = sorted(os.listdir(os.path.join(label_dir,phase)))
    #     for city in citys:
    #         label_path = os.path.join(label_dir, phase, city)
    #         label_names_all = sorted(os.listdir(label_path))
    #         label_names = [p for p in label_names_all if p.endswith('_labelIds.png')]
    #         for label_name in label_names:
    #             print(label_name)
    #             mask = np.array(Image.open(os.path.join(label_path, label_name)))
                # check_mask(mask)
    batch_size = mask_batch.shape[0]
    dist_cat_np_batch = []
    for i in range(batch_size):
        mask = np.array(mask_batch[i,:,:])##(256,512)
        h_offset, w_offset = cal_connectedComponents(mask, norm)
        dist_cat_np = np.concatenate((h_offset[np.newaxis, ...], w_offset[np.newaxis, ...]), 0)
    # dist_name = label_name[:-12]+'distance.npy'
    # np.save(os.path.join(label_path, dist_name), dist_cat_np)
        dist_cat_np_batch.append(dist_cat_np)
    return torch.Tensor(np.array(dist_cat_np_batch))


def cal_connectedComponents(mask, normal_mode='norm'):
    label_idxs = np.unique(mask)
    H, W = mask.shape
    out_h_offset = np.float32(np.zeros_like(mask))
    out_w_offset = np.float32(np.zeros_like(mask))
    for label_idx in label_idxs:
        if label_idx == 0:
            continue
        tmp_mask = np.float32(mask.copy())
        tmp_mask[tmp_mask!=label_idx] = -1
        tmp_mask[tmp_mask==label_idx] = 255
        tmp_mask[tmp_mask==-1] = 0
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(tmp_mask))
# image(输入)：也就是输入图像，必须是二值图，即8位单通道图像。（因此输入图像必须先进行二值化处理才能被这个函数接受）
# num_labels：所有连通域的数目
# labels：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
# stats：每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和面积，示例如下： 0 0 720 720 291805
# centroids：连通域的中心点
        connected_numbers = len(centroids)-1
        for c_idx in range(1,connected_numbers+1):
            tmp_labels = np.float32(labels.copy())
            tmp_labels[tmp_labels!=c_idx] = 0
            tmp_labels[tmp_labels==c_idx] = 1
            h_offset = (np.repeat(np.array(range(H))[...,np.newaxis],W,1) - centroids[c_idx][1])*tmp_labels
            w_offset = (np.repeat(np.array(range(W))[np.newaxis,...],H,0) - centroids[c_idx][0])*tmp_labels
            h_offset = normalize_dist(h_offset, normal_mode)
            w_offset = normalize_dist(w_offset, normal_mode)
            out_h_offset += h_offset
            out_w_offset += w_offset

    return out_h_offset, out_w_offset

def normalize_dist(offset, normal_mode):
    if normal_mode == 'no':
        return offset
    else:
        return offset / np.max(np.abs(offset)+1e-5)##original 1e-5

def show_results(ins):
    plt.imshow(ins)
    plt.show()

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def check_mask(mask, check_idx = 255):
    idx = np.unique(mask)
    if check_idx in idx:
        print(idx)








def make_dist_train_val_cityscapes_datasets_multichannel(mask_batch=None,norm='norm'):
    batch_size = mask_batch.shape[0]
    dist_cat_np_batch = []
    for i in range(batch_size):
        mask = np.array(mask_batch[i,:,:,:])##(35,256,512)
        # h_offset, w_offset = cal_connectedComponents_multichannel(mask, norm)
        # dist_cat_np = np.concatenate((h_offset[np.newaxis, ...], w_offset[np.newaxis, ...]), 0)

        dist_cat_np = cal_connectedComponents_multichannel(mask, norm)

        dist_cat_np_batch.append(dist_cat_np)
    return torch.Tensor(np.array(dist_cat_np_batch))



def cal_connectedComponents_multichannel(mask, normal_mode='norm'):
    # label_idxs = np.unique(mask)
    C, H, W = mask.shape
    # zeros = np.float32(np.zeros((H,W)))

    # out_w_offset_set = []
    # out_h_offset_set = []
    out_offset_set = []

    for layer_idx in range(C):
        tmp_mask = np.float32(mask[layer_idx].copy())
        if tmp_mask.sum() == 0:
            h_offset = tmp_mask + np.random.normal(0.0,0.1,size=(tmp_mask.shape))
            w_offset = tmp_mask + np.random.normal(0.0,0.1,size=(tmp_mask.shape))
        else:
            tmp_mask[tmp_mask!=1] = -1
            tmp_mask[tmp_mask==1] = 255
            tmp_mask[tmp_mask==-1] = 0
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(tmp_mask))
# image(输入)：也就是输入图像，必须是二值图，即8位单通道图像。（因此输入图像必须先进行二值化处理才能被这个函数接受）
# num_labels：所有连通域的数目
# labels：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
# stats：每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和面积，示例如下： 0 0 720 720 291805
# centroids：连通域的中心点
            connected_numbers = len(centroids)-1
            for c_idx in range(1,connected_numbers+1):
                tmp_labels = np.float32(labels.copy())
                tmp_labels[tmp_labels!=c_idx] = 0
                tmp_labels[tmp_labels==c_idx] = 1
                h_offset = (np.repeat(np.array(range(H))[...,np.newaxis],W,1) - centroids[c_idx][1])*tmp_labels
                w_offset = (np.repeat(np.array(range(W))[np.newaxis,...],H,0) - centroids[c_idx][0])*tmp_labels
                h_offset = normalize_dist(h_offset, normal_mode) + np.random.normal(0.0,0.1,size=(tmp_mask.shape))
                w_offset = normalize_dist(w_offset, normal_mode) + np.random.normal(0.0,0.1,size=(tmp_mask.shape))


        # out_h_offset_set.append(h_offset.reshape(1,H,W))
        # out_w_offset_set.append(w_offset.reshape(1,H,W))
        out_offset_set.append(h_offset.reshape(1,H,W))
        out_offset_set.append(w_offset.reshape(1,H,W))


    # out_h_offset = np.concatenate(out_h_offset_set)
    # out_w_offset = np.concatenate(out_w_offset_set)
    out_offset_set = np.concatenate(out_offset_set)

    # return out_h_offset, out_w_offset
    return out_offset_set


class conv_bn_relu(nn.Module):
    def __init__(self,in_channels=None,out_channels=None,kernel_size=None,padding=None,):
        super(conv_bn_relu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                kernel_size=kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
    def forward(self,x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class conv_layernorm_relu(nn.Module):
    def __init__(self,in_channels=None,out_channels=None,kernel_size=None,padding=None,h=None,w=None):
        super(conv_layernorm_relu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                kernel_size=kernel_size,padding=padding)
        self.norm = nn.LayerNorm([out_channels,h,w])
        self.act = nn.ReLU()
    def forward(self,x):
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class conv_bn(nn.Module):
    def __init__(self,in_channels=None,out_channels=None,kernel_size=None,padding=None,):
        super(conv_bn, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                kernel_size=kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    def forward(self,x):
        x = self.conv2d(x)
        x = self.bn(x)
        return x


class simple_resblock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3 ):
        super().__init__()
        fin = in_channels
        fout = out_channels
        padding = kernel_size//2
        self.conv_0 = nn.Conv2d(fin, fout, kernel_size=kernel_size, padding=padding)
        self.bn_0 = nn.BatchNorm2d(num_features=fout)
        self.conv_1 = nn.Conv2d(fout, fout, kernel_size=kernel_size, padding=padding)
        self.bn_1 = nn.BatchNorm2d(num_features=fout)

        self.skip = nn.Conv2d(fin, fout, kernel_size=1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.skip(x)
        x = self.bn_0(self.conv_0(x))
        x = self.activ(x)
        x = self.bn_1(self.conv_1(x))
        x = y + x
        x = self.activ(x)
        return x


class simple_resblock_encoder(nn.Module):
    def __init__(self,block_resolutions=None,
                 channels_nums=None,
                 in_channel=67):
        super(simple_resblock_encoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.block_resolutions = block_resolutions  ## input resolution
        self.channels_nums = channels_nums  ## input channel
        in_channel_en = in_channel
        for idx, resolution in enumerate(self.block_resolutions):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels_nums[idx]

            if resolution == 256:
                encoder_block.append(simple_resblock(in_channels=in_channel_en,
                                                  out_channels=out_channel_en, kernel_size=3,))
                encoder_block.append(simple_resblock(in_channels=out_channel_en,
                                                  out_channels=out_channel_en, kernel_size=3,))
                encoder_block.append(simple_resblock(in_channels=out_channel_en,
                                                  out_channels=out_channel_en, kernel_size=3,))
            else:
                encoder_block.append(simple_resblock(in_channels=in_channel_en,
                                                  out_channels=out_channel_en, kernel_size=3,))



            in_channel_en = out_channel_en
            encoder_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            setattr(self, f'encoder_block_res{resolution}', encoder_block)
        print('encoder constructed')

    def forward(self, x):
        blockss_outputs = {}
        for idx, resolution in enumerate(self.block_resolutions):
            block_outputs = []
            block = getattr(self, f'encoder_block_res{resolution}')
            for layer in block:
                x = layer(x)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs[f'res{resolution}'] = block_outputs
        return x, blockss_outputs




class Conv_encoder(nn.Module):
    def __init__(self,scales,out_channels,in_channel):
        super(Conv_encoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_en = in_channel
        for idx, scale in enumerate(self.scales):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels[-idx - 1]
            scale = self.scales[-idx-1]
            if scale >= 128:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            else:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            in_channel_en = out_channel_en
            encoder_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder.append(encoder_block)
    def forward(self,x):
        blockss_outputs = []
        for block in self.encoder:
            block_outputs = []
            for layer in block:
                x = layer(x)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs.append(block_outputs)
        return x,blockss_outputs








class Conv_encoder_resblock(nn.Module):
    def __init__(self,scales,out_channels,in_channel):
        super(Conv_encoder_resblock, self).__init__()
        self.encoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_en = in_channel
        for idx, scale in enumerate(self.scales):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels[-idx - 1]
            scale = self.scales[-idx-1]
            if scale >= 128:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            else:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            in_channel_en = out_channel_en
            encoder_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder.append(encoder_block)
    def forward(self,x):
        blockss_outputs = []
        for num1, block in enumerate(self.encoder):
            block_outputs = []
            for num2,layer in enumerate(block):
                x = layer(x)
                if layer == block[-2]:
                    input = block_outputs[0]
                    x = F.relu(x + input)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs.append(block_outputs)
        return x,blockss_outputs




class Conv_encoder_resblock_tree(nn.Module):
    def __init__(self,scales,out_channels,in_channel):
        super(Conv_encoder_resblock_tree, self).__init__()
        self.encoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_en = in_channel
        for idx, scale in enumerate(self.scales):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels[-idx - 1]
            scale = self.scales[-idx-1]
            if scale >= 128:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            else:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            in_channel_en = out_channel_en
            encoder_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder.append(encoder_block)
    def forward(self,x):
        blockss_outputs = []
        for num1, block in enumerate(self.encoder):
            block_outputs = []
            for num2,layer in enumerate(block):
                if num2 < len(block) - 3 :
                    x = layer(x)
                    block_outputs.append(x)
                elif layer == block[-3]:
                    input = block_outputs[0]
                    x = F.relu(x + input)
                    input = x
                    block_outputs.append(x)
                elif layer == block[-2]:
                    rgb_feature = layer(x)
                    rgb_feature += input
                    rgb_feature = F.relu(rgb_feature)
                    block_outputs.append(rgb_feature)
                elif layer == block[-1]:
                    x = layer(x)
            blockss_outputs.append(block_outputs)
        return x,blockss_outputs











class Conv_encoder_avepool(nn.Module):
    def __init__(self,scales,out_channels,in_channel):
        super(Conv_encoder_avepool, self).__init__()
        self.encoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_en = in_channel
        for idx, scale in enumerate(self.scales):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels[-idx - 1]
            scale = self.scales[-idx-1]
            if scale >= 128:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            else:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            in_channel_en = out_channel_en
            encoder_block.append(nn.AvgPool2d(kernel_size=2, stride=2))
            self.encoder.append(encoder_block)
    def forward(self,x):
        blockss_outputs = []
        for block in self.encoder:
            block_outputs = []
            for layer in block:
                x = layer(x)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs.append(block_outputs)
        return x,blockss_outputs




class Conv_encoder_avepool_for_bipartite_decoder(nn.Module):
    def __init__(self,block_resolutions=None,
                 channels_nums=None,
                 in_channel=67):
        super(Conv_encoder_avepool_for_bipartite_decoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.block_resolutions = block_resolutions## input resolution
        self.channels_nums = channels_nums## input channel
        in_channel_en = in_channel
        for idx, resolution in enumerate(self.block_resolutions):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels_nums[idx]

            encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same'))
            encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same'))
            encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same'))
            if resolution == 256:
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))

            in_channel_en = out_channel_en
            encoder_block.append(nn.AvgPool2d(kernel_size=2, stride=2))
            setattr(self, f'encoder_block_res{resolution}', encoder_block)
    def forward(self,x):
        blockss_outputs = {}
        for idx,resolution in enumerate(self.block_resolutions):
            block_outputs = []
            block = getattr(self, f'encoder_block_res{resolution}')
            for layer in block:
                x = layer(x)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs[f'res{resolution}']=block_outputs
        return x,blockss_outputs

class Conv_encoder_avepool_for_bipartite_decoder_layernorm(nn.Module):
    def __init__(self,block_resolutions=None,
                 channels_nums=None,
                 in_channel=67):
        super(Conv_encoder_avepool_for_bipartite_decoder_layernorm, self).__init__()
        self.encoder = nn.ModuleList()
        self.block_resolutions = block_resolutions## input resolution
        self.channels_nums = channels_nums## input channel
        in_channel_en = in_channel
        for idx, resolution in enumerate(self.block_resolutions):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels_nums[idx]

            encoder_block.append(conv_layernorm_relu(in_channels=in_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same',h=resolution,w=2*resolution))
            encoder_block.append(conv_layernorm_relu(in_channels=out_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same',h=resolution,w=2*resolution))
            encoder_block.append(conv_layernorm_relu(in_channels=out_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same',h=resolution,w=2*resolution))
            if resolution == 256:
                encoder_block.append(conv_layernorm_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same',h=resolution,w=2*resolution))
                encoder_block.append(conv_layernorm_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same',h=resolution,w=2*resolution))

            in_channel_en = out_channel_en
            encoder_block.append(nn.AvgPool2d(kernel_size=2, stride=2))
            setattr(self, f'encoder_block_res{resolution}', encoder_block)
    def forward(self,x):
        blockss_outputs = {}
        for idx,resolution in enumerate(self.block_resolutions):
            block_outputs = []
            block = getattr(self, f'encoder_block_res{resolution}')
            for layer in block:
                x = layer(x)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs[f'res{resolution}']=block_outputs
        return x,blockss_outputs



class Conv_encoder_avgpool_for_bipDC(nn.Module):
    def __init__(self,block_resolutions=None,
                 channels_nums=None,
                 in_channel=67):
        super(Conv_encoder_avgpool_for_bipDC, self).__init__()
        self.encoder = nn.ModuleList()
        self.block_resolutions = block_resolutions## input resolution
        self.channels_nums = channels_nums## input channel
        in_channel_en = in_channel
        for idx, resolution in enumerate(self.block_resolutions):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels_nums[idx]

            encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same'))
            encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same'))
            encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                out_channels=out_channel_en, kernel_size=3,
                                                padding='same'))
            encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                              out_channels=out_channel_en, kernel_size=3,
                                              padding='same'))
            if resolution == 256:
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))


            in_channel_en = out_channel_en
            encoder_block.append(nn.AvgPool2d(kernel_size=2, stride=2))
            setattr(self, f'encoder_block_res{resolution}', encoder_block)
    def forward(self,x):
        blockss_outputs = {}
        for idx,resolution in enumerate(self.block_resolutions):
            block_outputs = []
            block = getattr(self, f'encoder_block_res{resolution}')
            for layer in block:
                x = layer(x)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs[f'res{resolution}']=block_outputs
        return x,blockss_outputs








class Conv_encoder_convdownsampling(nn.Module):
    def __init__(self,scales,out_channels,in_channel):
        super(Conv_encoder_convdownsampling, self).__init__()
        self.encoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_en = in_channel
        for idx, scale in enumerate(self.scales):
            encoder_block = nn.ModuleList()
            ############### Encoder ##############################
            out_channel_en = self.channels[-idx - 1]
            scale = self.scales[-idx-1]
            if scale >= 128:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            else:
                encoder_block.append(conv_bn_relu(in_channels=in_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
                encoder_block.append(conv_bn_relu(in_channels=out_channel_en,
                                                                out_channels=out_channel_en, kernel_size=3,
                                                                padding='same'))
            in_channel_en = out_channel_en
            encoder_block.append(nn.Conv2d(in_channels=out_channel_en,out_channels=out_channel_en,kernel_size=2,stride=2))
            self.encoder.append(encoder_block)
    def forward(self,x):
        blockss_outputs = []
        for block in self.encoder:
            block_outputs = []
            for layer in block:
                x = layer(x)
                if layer != block[-1]:
                    block_outputs.append(x)
            blockss_outputs.append(block_outputs)
        return x,blockss_outputs





class conv1x1_decoder(nn.Module):
    def __init__(self,scales,out_channels,decoder_param=None):
        super(conv1x1_decoder, self).__init__()
        style_dim=decoder_param['style_dim']
        demodulate=decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.decoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_de = 2*out_channels[0]
        for idx, scale in enumerate(self.scales):
            decoder_block = nn.ModuleList()
            out_channel_de = self.channels[idx]
            if scale >= 128:
                decoder_block.append(StyledConv(in_channel_de,
                                                  out_channel_de,
                                                  1,
                                                  style_dim,
                                                  demodulate=demodulate,
                                                  activation=activation,
                                                  approach=approach,
                                                  add_dist=add_dist  # jhl
                                                  ))
                decoder_block.append(StyledConv(2*out_channel_de,
                                                  out_channel_de,
                                                  1,
                                                  style_dim,
                                                  demodulate=demodulate,
                                                  activation=activation,
                                                  approach=approach,
                                                  add_dist=add_dist  # jhl
                                                  ))
            else:
                decoder_block.append(StyledConv(in_channel_de,
                                                  out_channel_de,
                                                  1,
                                                  style_dim,
                                                  demodulate=demodulate,
                                                  activation=activation,
                                                  approach=approach,
                                                  add_dist=add_dist  # jhl
                                                  ))
                decoder_block.append(StyledConv(2*out_channel_de,
                                                  out_channel_de,
                                                  1,
                                                  style_dim,
                                                  demodulate=demodulate,
                                                  activation=activation,
                                                  approach=approach,
                                                  add_dist=add_dist  # jhl
                                                  ))
                decoder_block.append(StyledConv(2*out_channel_de,
                                                  out_channel_de,
                                                  1,
                                                  style_dim,
                                                  demodulate=demodulate,
                                                  activation=activation,
                                                  approach=approach,
                                                  add_dist=add_dist  # jhl
                                                  ))
            in_channel_de = out_channel_de
            decoder_block.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder.append(decoder_block)
    def forward(self,x,ffs,connectors,outputs_from_encoder,latent=None):
        blockss_outputs = []
        labels = latent[1]
        latent = latent[0]
        for idx,block in enumerate(self.decoder):
            block_dict = outputs_from_encoder[-1-idx]
            block_outputs = []
            for layer_idx,layer in enumerate(block):
                if layer == block[0]:
                    x = layer(x,style_conv_dict=block_dict[-1-layer_idx],style = latent,label = labels[idx])
                    block_outputs.append(x)
                elif layer != block[-1]:
                    x = torch.cat((x,ffs[idx]),dim=1)
                    x = layer(x,style_conv_dict=block_dict[-1-layer_idx],style = latent,label = labels[idx])
                    block_outputs.append(x)
                else:
                    if block != self.decoder[-1]:
                        x = layer(x)
                        x = x + connectors[idx+1]
            blockss_outputs.append(block_outputs)
        return x,blockss_outputs










class conv1x1_decoder_for_ganformerencoder(nn.Module):
    def __init__(self,scales,out_channels,decoder_param=None):
        super(conv1x1_decoder_for_ganformerencoder, self).__init__()
        style_dim=decoder_param['style_dim']
        demodulate=decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.decoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_de = 2*out_channels[0]
        for idx, scale in enumerate(self.scales):
            decoder_block = nn.ModuleList()
            out_channel_de = self.channels[idx]

            decoder_block.append(StyledConv(in_channel_de,
                                              out_channel_de,
                                              1,
                                              style_dim,
                                              demodulate=demodulate,
                                              activation=activation,
                                              approach=approach,
                                              add_dist=add_dist  # jhl
                                              ))

            in_channel_de = out_channel_de
            decoder_block.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder.append(decoder_block)
    def forward(self,x,ffs,connectors,outputs_from_encoder,latent=None):
        blockss_outputs = []
        labels = latent[1]
        latent = latent[0]
        for idx,block in enumerate(self.decoder):
            block_dict = outputs_from_encoder[-1-idx]
            block_outputs = []
            for layer_idx,layer in enumerate(block):
                if layer == block[0]:
                    x = layer(x,style_conv_dict=block_dict[-1-layer_idx],style = latent,label = labels[idx])
                    block_outputs.append(x)
                elif layer != block[-1]:
                    x = torch.cat((x,ffs[idx]),dim=1)
                    x = layer(x,style_conv_dict=block_dict[-1-layer_idx],style = latent,label = labels[idx])
                    block_outputs.append(x)
                else:
                    if block != self.decoder[-1]:
                        x = layer(x)
                        x = x + connectors[idx+1]
            blockss_outputs.append(block_outputs)
        return x,blockss_outputs










class conv1x1_decoder_ordered(nn.Module):
    def __init__(self, scales, out_channels, decoder_param=None):
        super(conv1x1_decoder_ordered, self).__init__()
        style_dim = decoder_param['style_dim']
        demodulate = decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.decoder = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        in_channel_de = 2 * out_channels[0]
        for idx, scale in enumerate(self.scales):
            decoder_block = nn.ModuleList()
            out_channel_de = self.channels[idx]
            if scale >= 128:
                decoder_block.append(StyledConv(in_channel_de,
                                                out_channel_de,
                                                1,
                                                style_dim,
                                                demodulate=demodulate,
                                                activation=activation,
                                                approach=approach,
                                                add_dist=add_dist  # jhl
                                                ))
                decoder_block.append(StyledConv(2 * out_channel_de,
                                                out_channel_de,
                                                1,
                                                style_dim,
                                                demodulate=demodulate,
                                                activation=activation,
                                                approach=approach,
                                                add_dist=add_dist  # jhl
                                                ))
            else:
                decoder_block.append(StyledConv(in_channel_de,
                                                out_channel_de,
                                                1,
                                                style_dim,
                                                demodulate=demodulate,
                                                activation=activation,
                                                approach=approach,
                                                add_dist=add_dist  # jhl
                                                ))
                decoder_block.append(StyledConv(2 * out_channel_de,
                                                out_channel_de,
                                                1,
                                                style_dim,
                                                demodulate=demodulate,
                                                activation=activation,
                                                approach=approach,
                                                add_dist=add_dist  # jhl
                                                ))
                decoder_block.append(StyledConv(2 * out_channel_de,
                                                out_channel_de,
                                                1,
                                                style_dim,
                                                demodulate=demodulate,
                                                activation=activation,
                                                approach=approach,
                                                add_dist=add_dist  # jhl
                                                ))
            in_channel_de = out_channel_de
            decoder_block.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder.append(decoder_block)

    def forward(self, x, ffs, connectors, outputs_from_encoder, latent=None):
        blockss_outputs = []
        labels = latent[1]
        latent = latent[0]
        for idx, block in enumerate(self.decoder):
            block_dict = outputs_from_encoder[idx]
            block_outputs = []
            for layer_idx, layer in enumerate(block):
                if layer == block[0]:
                    x = layer(x, style_conv_dict=block_dict[layer_idx], style=latent, label=labels[idx])
                    block_outputs.append(x)
                elif layer != block[-1]:
                    x = torch.cat((x, ffs[idx]), dim=1)
                    x = layer(x, style_conv_dict=block_dict[layer_idx], style=latent, label=labels[idx])
                    block_outputs.append(x)
                else:
                    if block != self.decoder[-1]:
                        x = layer(x)
                        x = x + connectors[idx + 1]
            blockss_outputs.append(block_outputs)
        return x, blockss_outputs







class to_rgb_block(nn.Module):
    def __init__(self,scales,out_channels,decoder_param=None):
        super(to_rgb_block, self).__init__()
        style_dim=decoder_param['style_dim']
        demodulate=decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.to_rgb_blocks = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        for idx, scale in enumerate(self.scales):
            block = nn.ModuleList()
            in_channel = self.channels[idx]
            if scale != 256:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
                block.append(nn.Upsample(scale_factor=2,mode='nearest'))
            else:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
            self.to_rgb_blocks.append(block)
    def forward(self,outputs_from_decoder,outputs_from_encoder,latent):
        rgb=0
        for idx,block in enumerate(self.to_rgb_blocks):
            block_dict = outputs_from_encoder[-1 - idx]
            x = block[0](outputs_from_decoder[idx],style_conv_dict=block_dict[0],style=latent)
            x = rgb + x
            if block != self.to_rgb_blocks[-1]:
                rgb = block[1](x)
            else:
                rgb = x
        return rgb



class to_rgb_block_bilinear(nn.Module):
    def __init__(self,scales,out_channels,decoder_param=None):
        super(to_rgb_block_bilinear, self).__init__()
        style_dim=decoder_param['style_dim']
        demodulate=decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.to_rgb_blocks = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        for idx, scale in enumerate(self.scales):
            block = nn.ModuleList()
            in_channel = self.channels[idx]
            if scale != 256:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
                block.append(nn.Upsample(scale_factor=2,mode='bilinear'))
            else:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
            self.to_rgb_blocks.append(block)
    def forward(self,outputs_from_decoder,outputs_from_encoder,latent):
        rgb=0
        labels = latent[1]
        latent = latent[0]
        for idx,block in enumerate(self.to_rgb_blocks):
            block_dict = outputs_from_encoder[-1 - idx]
            x = block[0](outputs_from_decoder[idx],style_conv_dict=block_dict[0],style=latent,label=labels[idx])
            x = rgb + x
            if block != self.to_rgb_blocks[-1]:
                rgb = block[1](x)
            else:
                rgb = x
        return rgb





class to_rgb_block_trans_con2d(nn.Module):
    def __init__(self,scales,out_channels,decoder_param=None):
        super(to_rgb_block_trans_con2d, self).__init__()
        style_dim=decoder_param['style_dim']
        demodulate=decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.to_rgb_blocks = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        for idx, scale in enumerate(self.scales):
            block = nn.ModuleList()
            in_channel = self.channels[idx]
            if scale != 256:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
                # block.append(nn.Upsample(scale_factor=2,mode='bilinear'))
                # block.append(nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0))
                block.append(EqualConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0))
            else:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
            self.to_rgb_blocks.append(block)
    def forward(self,outputs_from_decoder,outputs_from_encoder,latent):
        rgb=0
        labels = latent[1]
        latent = latent[0]
        for idx,block in enumerate(self.to_rgb_blocks):
            block_dict = outputs_from_encoder[-1 - idx]
            x = block[0](outputs_from_decoder[idx],style_conv_dict=block_dict[0],style=latent,label=labels[idx])
            x = rgb + x
            if block != self.to_rgb_blocks[-1]:
                # rgb = block[1](x,output_size=(2*x.shape[-2],2*x.shape[-1]))
                rgb = block[1](x)
            else:
                rgb = x
        return rgb








class to_rgb_block_transpose2d_ordered(nn.Module):
    def __init__(self,scales,out_channels,decoder_param=None):
        super(to_rgb_block_transpose2d_ordered, self).__init__()
        style_dim=decoder_param['style_dim']
        demodulate=decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.to_rgb_blocks = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        for idx, scale in enumerate(self.scales):
            block = nn.ModuleList()
            in_channel = self.channels[idx]
            if scale != 256:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
                # block.append(nn.Upsample(scale_factor=2,mode='bilinear'))
                # block.append(nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0))
                block.append(EqualConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0))
            else:
                block.append(ToRGB(in_channel=in_channel, style_dim=style_dim, upsample=False, approach=approach,
                                                     add_dist=add_dist))
            self.to_rgb_blocks.append(block)
    def forward(self,outputs_from_decoder,outputs_from_encoder,latent):
        rgb=0
        labels = latent[1]
        latent = latent[0]
        for idx,block in enumerate(self.to_rgb_blocks):
            block_dict = outputs_from_encoder[idx]
            x = block[0](outputs_from_decoder[idx],style_conv_dict=block_dict[-1],style=latent,label=labels[idx])
            x = rgb + x
            if block != self.to_rgb_blocks[-1]:
                # rgb = block[1](x,output_size=(2*x.shape[-2],2*x.shape[-1]))
                rgb = block[1](x)
            else:
                rgb = x
        return rgb


















class to_rgb_block_SLE(nn.Module):
    def __init__(self,scales,out_channels,decoder_param=None):
        super( to_rgb_block_SLE, self).__init__()
        style_dim=decoder_param['style_dim']
        demodulate=decoder_param['demodulate']
        approach = decoder_param['approach']
        activation = decoder_param['activation']
        add_dist = decoder_param['add_dist']

        self.to_rgb_blocks = nn.ModuleList()
        self.channels = out_channels
        self.scales = scales
        self.sle32_128 = SLEBlock(1024,128)
        self.sle64_256 = SLEBlock(512,64)

        self.tRGB_64 = ToRGB(in_channel=256, style_dim=style_dim, upsample=False, approach=approach,add_dist=add_dist)
        self.tRGB_128 = ToRGB(in_channel=128, style_dim=style_dim, upsample=False, approach=approach,add_dist=add_dist)
        self.tRGB_256 = ToRGB(in_channel=64, style_dim=style_dim, upsample=False, approach=approach,add_dist=add_dist)

    def forward(self,outputs_from_decoder,outputs_from_encoder,latent):

        output64 = outputs_from_decoder[-3]
        output128 = self.sle32_128(outputs_from_decoder[-2],outputs_from_decoder[-5])
        output256 = self.sle64_256(outputs_from_decoder[-1],outputs_from_decoder[-4])
        rgb_64 = self.tRGB_64(output64,style_conv_dict=outputs_from_encoder[2][0],style=latent)
        rgb_128 = self.tRGB_128(output128,style_conv_dict=outputs_from_encoder[1][0],style=latent)+F.interpolate(rgb_64,scale_factor=2,mode='nearest')
        rgb_256 = self.tRGB_256(output256,style_conv_dict=outputs_from_encoder[0][0],style=latent)+F.interpolate(rgb_128,scale_factor=2,mode='nearest')

        return rgb_256






class SLEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4),
                                    nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
                                    nn.Sigmoid() )

    def forward(self, feat_big, feat_small):
        return feat_big * self.main(feat_small)







def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class compute_bipartite(nn.Module):
    def __init__(self):
        super(compute_bipartite, self).__init__()

    # TransformerLayer(
    #     (att_dp): Dropout(p=0.06, inplace=False)
    # (to_queries): FullyConnectedLayer()
    # (to_keys): FullyConnectedLayer()
    # (to_values): FullyConnectedLayer()
    # (from_pos_map): FullyConnectedLayer()
    # (to_pos_map): FullyConnectedLayer()
    # (to_gate_attention): GateAttention()
    # (from_gate_attention): GateAttention()
    # (modulation): FullyConnectedLayer()
    # )

    # dim,  # The layer dimension
    # pos_dim,  # Positional encoding dimension
    # from_len, to_len,  # The from/to tensors length (must be specified if from/to has 2 dims)
    # from_dim, to_dim,  # The from/to tensors dimensions
    # from_gate = False, to_gate = False,  # Add sigmoid gate on from/to, so that info may not be sent/received
    # # when gate is low (i.e. the attention probs may not sum to 1)
    # # Additional options
    # num_heads = 1,  # Number of attention heads
    # attention_dropout = 0.12,  # Attention dropout rate
    # integration = "add",  # Feature integration type: additive, multiplicative or both
    # norm = None,  # Feature normalization type (optional): instance, batch or layer
    #
    # # k-means options (optional, duplex)
    # kmeans = False,  # Track and update image-to-latents assignment centroids, used in the duplex attention
    # kmeans_iters = 1,  # Number of K-means iterations per transformer layer
    # iterative = False,  # Carry over attention assignments across transformer layers of different resolutions
    # # If True, centroids are carried from layer to layer
    # ** _kwargs

    # {'dim' = 512,
    #  'from_len': 16,
    #  'to_len': 16,
    #  'from_dim': 512,
    #  'to_dim': 32,
    #  'from_gate': False,
    #  'to_gate': False,
    #  'num_heads': 1,
    #  'attention_dropout': 0.12,
    #  'integration': 'mul',
    #  'norm': 'layer',
    #  'kmeans': True,
    #  'kmeans_iters': 1,
    #  'iterative': False,
    #  'pos_dim': 32,}

    # {'dim' = 512,
    #  'from_len': 64,
    #  'to_len': 16,
    #  'from_dim': 512,
    #  'to_dim': 32,
    #  'from_gate': False,
    #  'to_gate': False,
    #  'num_heads': 1,
    #  'attention_dropout': 0.12,
    #  'integration': 'mul',
    #  'norm': 'layer',
    #  'kmeans': True,
    #  'kmeans_iters': 1,
    #  'iterative': False,
    #  'pos_dim': 32,}

    # {'dim' = 512,
    #  'from_len': 256,
    #  'to_len': 16,
    #  'from_dim': 512,
    #  'to_dim': 32,
    #  'from_gate': False,
    #  'to_gate': False,
    #  'num_heads': 1,
    #  'attention_dropout': 0.12,
    #  'integration': 'mul',
    #  'norm': 'layer',
    #  'kmeans': True,
    #  'kmeans_iters': 1,
    #  'iterative': False,
    #  'pos_dim': 32}


        transformer_kwargs = {
                              'from_len': 256,
                              'from_dim': 512,
                              'to_dim': 32,
                              'from_gate': False,
                              'to_gate': False,
                              'num_heads': 1,
                              'attention_dropout': 0.12,
                              'integration': 'mul',
                              'norm': 'layer',
                              'kmeans': True,
                              'kmeans_iters': 1,
                              'iterative': False,
                              'pos_dim': 32
                              }
        self.transformer = Ganformernetworks.TransformerLayer(dim=512,to_len = 16, **transformer_kwargs)

    # setattr(self, f"b{res}", block)
    # block = getattr(self, f"b{res}")

    def forward(self,ws,encoder_outputs):

        x, img, att_maps = None, None, []
        att_vars = {"centroid_assignments": None}


        ws = ws.to(torch.float32)
        block_ws = [ws.narrow(2, 0,4),
                    ws.narrow(2, 4,4),
                    ws.narrow(2, 8,4),
                    ws.narrow(2, 12,3),
                    ws.narrow(2, 15,3)]
        for ws,feature in zip(ws,encoder_outputs):
            w_iter = iter(ws.unbind(dim = 2))

            if self.stem:
                batch_size = ws.shape[0]
                if self.latent_stem:
                    x = self.conv_stem(get_global(next(w_iter)))
                    x = x.reshape(batch_size, -1, *self.init_shape)
                else:
                    x = self.const.unsqueeze(0).repeat([batch_size, 1, 1, 1])
            else:
                torch_misc.assert_shape(x, [None, self.in_channels, self.res // 2, self.res // 2])
            x = convert(x)



            x, att_maps[0], att_vars = self.conv0(x, next(w_iter), att_vars, fused_modconv = fused_modconv, **layer_kwargs)
            att_map, noise = None, None
            if self.local_noise and noise_mode != "none":
                if noise_mode == "random":
                    noise = torch.randn([x.shape[0], 1, self.out_res, self.out_res], device=x.device)
                if noise_mode == "const":
                    noise = self.noise_const
                noise = noise * self.noise_strength




            att_maps += _att_maps
        att_maps = self.list2tensor(att_maps, ws.device)



class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.Sequential(
                    nn.BatchNorm2d(dim_in),
                    nn.ReLU()
                ),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

















































































