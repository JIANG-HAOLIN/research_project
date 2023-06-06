import math

import torch.nn as nn
import models.norms as norms
import torch
import torch.nn.functional as F
# from .blocks import *
from . import blocks as CIPblocks
from models.discriminator import make_kernel,upfirdn2d,InverseHaarTransform,HaarTransform

import models.tensor_transforms as tt
from models.trans_encoder import Img2Token
## ganformer_begin
from models.ganformer.NetworkS.networks import bipartite_attention_computer,BipartiteEncoder,BipartiteDecoder
from models.ganformer.NetworkS.bipaDEC_1spade import BipartiteDecoder_1spade
from models.ganformer.NetworkS.bipaDEC_1spade_CatFeature import BipartiteDecoder_1spade_catFeature
from models.ganformer.NetworkS.bipDEC_1spd_CatFeat_noSkip import BipartiteDecoder_1spd_catFeat_noSkip

from models.ganformer.NetworkS.bipaDEC_1spade_CatFeat_skip import BipartiteDecoder_1spade_catFeat_skip
from models.ganformer.NetworkS.bipaDEC_noENC_1spade_CatFeature import BipartiteDecoder_1spade_noENC_catFeature

from models.ganformer.NetworkS.bipaDEC_cat import BipartiteDecoder_cat
from models.ganformer.NetworkS.bipaDEC_cat_nospade import BipartiteDecoder_cat_nospade
from models.ganformer.NetworkS.bipaDEC_contrastive import BipartiteDecoder_contrastive
from models.ganformer.NetworkS.bipaDEC_catFeature_skipSPD import BipartiteDecoder_catFeature_skipSPD
from models.ganformer.NetworkS.bipaDEC_catLabel_skipSPD_3Dnoise import BipartiteDecoder_catlabel_skipSPD_3Dnoise
from models.ganformer.NetworkS.bipaDEC_shallow_skipSPD_3Dnoise import BipartiteDecoder_shallow_skipSPD_3Dnoise
from models.ganformer.NetworkS.bipaDEC_shallow2_skipSPD_3Dnoise import BipartiteDecoder_shallow2_skipSPD_3Dnoise

from models.ganformer.NetworkS.bipaDEC_catFeat_skipSPD_3Dnoise_noEC import BipDC_catFeat_skipSPD_3Dnoise_noEC

from models.ganformer.NetworkS.bipaDEC_catLbFeat_skipSPD import BipartiteDecoder_catLbFeat_skipSPD
from models.ganformer.NetworkS.bipaDEC_1spade_CatLabel import BipartiteDecoder_1spade_catLabel
from models.ganformer.NetworkS.bipaDEC_1spade_CatFeature_3D import BipartiteDecoder_1spade_catFeature_3D




class ImplicitGenerator_tanh(nn.Module):

    def __init__(self, opt=None, size=(256, 512), hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, z=None, **kwargs):
        super(ImplicitGenerator_tanh, self).__init__()

        self.opt = opt
        if opt.apply_MOD_CLADE:
            self.approach = 0
        elif opt.only_CLADE:
            self.approach = 1.4
        elif opt.Matrix_Computation:
            self.approach = 2
        else:
            self.approach = -1
        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = CIPblocks.LFF(int(hidden_size/2))
        self.emb = (CIPblocks.ConstantInput(hidden_size, size=size))

        self.channels = {
            0: 512,
            1: 512,  ##512
            2: 256,  ##512
            3: 256,  ##512
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])
        self.conv1 = CIPblocks.StyledConv(int(multiplier * hidden_size),  ##the real in_channel 1024
                                          in_channels,  ##actually is out_channel
                                          1,
                                          style_dim,
                                          demodulate=demodulate,
                                          activation=activation,
                                          approach=self.approach,
                                          add_dist=self.add_dist# jhl
                                          )
        ###kernel_size = 1===>first modFC layer!!only one layer!!input=embbed coords!!

        self.linears = nn.ModuleList()
        ##2xModFC for 2-8 Layers
        self.to_rgbs = nn.ModuleList()
        ##tRGB for 2-8 Layers
        self.log_size = int(CIPblocks.math.log(max(size), 2))
        ## 8 Layers

        self.n_intermediate = self.log_size - 6
        ## intermediate layer(7 layers except first layer)
        self.to_rgb_stride = 2
        ##how many ModFC between two tRGB==>in this case, 2 ModFC layers
        for i in range(0, self.log_size - 1):  ## for each layer in intermediate 7 Layers:
            out_channels = self.channels[i]
            self.linears.append(CIPblocks.StyledConv(in_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl
            self.linears.append(CIPblocks.StyledConv(out_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl
            self.to_rgbs.append(
                CIPblocks.ToRGB(out_channels, style_dim, upsample=False,
                                approach=self.approach,
                                add_dist=self.add_dist))  # jhl
            ###upsample turned off manually
            # print(out_channels)
            in_channels = out_channels
            ##2xModFC+tRGB for 2-8 Layers

        self.style_dim = style_dim
        ##dimension of style vector

        layers = [CIPblocks.PixelNorm()]
        ##layers for latent normalization

        for i in range(n_mlp):  ##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        ##mapping network that generate style w!!

        self.styleMatrix = nn.Parameter(torch.randn(35, 512))
        # self.styleMatrix.data.fill_(0.25)
        # self.alpha = nn.Parameter(torch.rand(1,512))
        # self.alpha.data.fill_(0.5)

    def forward(self,
                label,  ##[1,35,256,512]
                label_class_dict,
                coords,  ##[1,2,256,512]
                latent,  ##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                ):
        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        # print("input latent code:",latent)
        latent = latent[0]  ##[1,512]
        ##input noirse z
        # print("received latent[0] :",latent.shape,latent)
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)
        ##style w [1,512]

        # latent = self.alpha*latent + (1-self.alpha)*self.styleMatrix
        ##combined style vector [35,512]

        x = self.lff(coords)
        x = torch.cat((x,self.lff(dist_map)),dim = 1)
        ##Fourier Features:simple linear transformation with sin activation
        ##[N,512,256,512]
        # print(x)

        batch_size, _, h, w = coords.shape

        if self.training and h == self.size[0] and w == self.size[1]:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                # Given an input and a flow-field grid,
                # computes the output using input values and pixel locations from grid.
                # input(N,C,H_in,W_in),grid(N,2,H_out,W_out),out(N,C,H_out,W_out)
                self.emb.learnable_vectors.expand(batch_size, -1, -1, -1),
                # 调用emb class的self.input!!
                # -1 means not changing the size of that dimension!!!!
                (coords.permute(0, 2, 3, 1).contiguous()),
                padding_mode='border', mode='bilinear',
            )



        x = torch.cat([x, emb], 1)
        ##concatenation of Fourier Features and Coordinates Embeddings on channel dimension!!!
        ##[1,1024,256,512]

        rgb = 0

        x = self.conv1(x,latent,
                       label_class_dict=label_class_dict,
                       label=label,
                       class_style=self.styleMatrix,
                       dist_map=dist_map
                       )
        ##first ModFC layer
        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            # print(i)
            for j in range(self.to_rgb_stride):  ##2xModFC
                x = self.linears[i * self.to_rgb_stride + j](x, latent,
                                                             label_class_dict=label_class_dict,
                                                             label=label,
                                                             class_style=self.styleMatrix,
                                                             dist_map=dist_map)

            rgb = self.to_rgbs[i](x,latent,rgb,
                                  label_class_dict=label_class_dict,
                                  label=label,
                                  class_style=self.styleMatrix,
                                  dist_map=dist_map)
            ####skip=rgb ==> rgb image accumulation!!

        if return_latents:
            return rgb, latent
        else:

            # print("rgb size:",rgb.size())
            # return self.tanh(rgb), None
            return self.tanh(rgb), None




class ImplicitGenerator_tanh_no_emb(nn.Module):

    def __init__(self, opt=None, size=(256, 512), hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, z=None, **kwargs):
        super(ImplicitGenerator_tanh_no_emb, self).__init__()


        self.opt = opt
        if opt.apply_MOD_CLADE:
            self.approach = 0
        elif opt.only_CLADE:
            self.approach = 1.5
        elif opt.Matrix_Computation:
            self.approach = 2
        else:
            self.approach = -1
        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = CIPblocks.LFF(int(hidden_size/2))

        self.channels = {
            0: 512,
            1: 512,  ##512
            2: 256,  ##512
            3: 256,  ##512
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])
        self.conv1 = CIPblocks.StyledConv(512,  ##the real in_channel 1024
                                          in_channels,  ##actually is out_channel
                                          1,
                                          style_dim,
                                          demodulate=demodulate,
                                          activation=activation,
                                          approach=self.approach,
                                          add_dist=self.add_dist# jhl
                                          )
        ###kernel_size = 1===>first modFC layer!!only one layer!!input=embbed coords!!

        self.linears = nn.ModuleList()
        ##2xModFC for 2-8 Layers
        self.to_rgbs = nn.ModuleList()
        ##tRGB for 2-8 Layers
        self.log_size = int(CIPblocks.math.log(max(size), 2))
        ## 8 Layers

        self.n_intermediate = self.log_size - 6
        ## intermediate layer(7 layers except first layer)
        self.to_rgb_stride = 2
        ##how many ModFC between two tRGB==>in this case, 2 ModFC layers
        for i in range(0, self.log_size - 1):  ## for each layer in intermediate 7 Layers:
            out_channels = self.channels[i]
            self.linears.append(CIPblocks.StyledConv(in_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl
            self.linears.append(CIPblocks.StyledConv(out_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl
            self.to_rgbs.append(
                CIPblocks.ToRGB(out_channels, style_dim, upsample=False,
                                approach=self.approach,
                                add_dist=self.add_dist))  # jhl
            ###upsample turned off manually
            # print(out_channels)
            in_channels = out_channels
            ##2xModFC+tRGB for 2-8 Layers

        self.style_dim = style_dim
        ##dimension of style vector

        layers = [CIPblocks.PixelNorm()]
        ##layers for latent normalization

        for i in range(n_mlp):  ##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        ##mapping network that generate style w!!

        self.styleMatrix = nn.Parameter(torch.randn(35, 512))
        # self.styleMatrix.data.fill_(0.25)
        # self.alpha = nn.Parameter(torch.rand(1,512))
        # self.alpha.data.fill_(0.5)

    def forward(self,
                label,  ##[1,35,256,512]
                label_class_dict,
                coords,  ##[1,2,256,512]
                latent,  ##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                ):
        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        # print("input latent code:",latent)
        latent = torch.randn(label_class_dict.shape[0],35,512).to('cuda')
        ##input noirse z
        # print("received latent[0] :",latent.shape,latent)
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)
        ##style w [1,512]

        # latent = self.alpha*latent + (1-self.alpha)*self.styleMatrix
        ##combined style vector [35,512]

        x = self.lff(coords)
        x = torch.cat((x,self.lff(dist_map)),dim = 1)

        batch_size, _, h, w = coords.shape






        ##concatenation of Fourier Features and Coordinates Embeddings on channel dimension!!!
        ##[1,1024,256,512]

        rgb = 0

        x = self.conv1(x,latent,
                       label_class_dict=label_class_dict,
                       label=label,
                       class_style=self.styleMatrix,
                       dist_map=dist_map
                       )
        ##first ModFC layer
        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                x = self.linears[i * self.to_rgb_stride + j](x, latent,
                                                             label_class_dict=label_class_dict,
                                                             label=label,
                                                             class_style=self.styleMatrix,
                                                             dist_map=dist_map)

            rgb = self.to_rgbs[i](x,latent,rgb,
                                  label_class_dict=label_class_dict,
                                  label=label,
                                  class_style=self.styleMatrix,
                                  dist_map=dist_map)
            ####skip=rgb ==> rgb image accumulation!!

        if return_latents:
            return rgb, latent
        else:

            # print("rgb size:",rgb.size())
            # return self.tanh(rgb), None
            return self.tanh(rgb), None



class ImplicitGenerator_tanh_skip(nn.Module):

    def __init__(self, opt=None, size=(256, 512), hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, z=None, **kwargs):
        super(ImplicitGenerator_tanh_skip, self).__init__()

        self.opt = opt
        if opt.apply_MOD_CLADE:
            self.approach = 0
        elif opt.only_CLADE:
            self.approach = 1.6
        elif opt.Matrix_Computation:
            self.approach = 2
        else:
            self.approach = -1
        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = CIPblocks.LFF(int(hidden_size/2))
        self.emb = (CIPblocks.ConstantInput(hidden_size, size=size))

        self.channels = {
            0: 512,
            1: 512,  ##512
            2: 256,  ##512
            3: 256,  ##512
            4: 512 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])

        self.i2t = Img2Token(dim=in_channels)
        self.w_encoder_init = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
        self.b_encoder_init = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
        self.conv1 = CIPblocks.StyledConv(int(multiplier * hidden_size),  ##the real in_channel 1024
                                          in_channels,  ##actually is out_channel
                                          1,
                                          style_dim,
                                          demodulate=demodulate,
                                          activation=activation,
                                          approach=self.approach,
                                          add_dist=self.add_dist# jhl
                                          )

        ###kernel_size = 1===>first modFC layer!!only one layer!!input=embbed coords!!

        self.linears = nn.ModuleList()
        self.w_encoder = nn.ModuleList()
        self.b_encoder = nn.ModuleList()
        ##2xModFC for 2-8 Layers
        self.to_rgbs = nn.ModuleList()
        ##tRGB for 2-8 Layers
        self.log_size = int(CIPblocks.math.log(max(size), 2))
        ## 8 Layers

        self.n_intermediate = self.log_size - 6
        ## intermediate layer(7 layers except first layer)
        self.to_rgb_stride = 2

        self.linear512_256_w = nn.Linear(512,256)
        self.linear512_256_b = nn.Linear(512,256)



        ##how many ModFC between two tRGB==>in this case, 2 ModFC layers
        for i in range(0, self.log_size - 1):  ## for each layer in intermediate 7 Layers:
            out_channels = self.channels[i]

            self.w_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))
            self.b_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))

            self.linears.append(CIPblocks.StyledConv(in_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl

            self.w_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))
            self.b_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))

            self.linears.append(CIPblocks.StyledConv(out_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl

            self.to_rgbs.append(
                CIPblocks.ToRGB(out_channels, style_dim, upsample=False,
                                approach=self.approach,
                                add_dist=self.add_dist))  # jhl
            ###upsample turned off manually
            # print(out_channels)
            in_channels = out_channels
            ##2xModFC+tRGB for 2-8 Layers

        self.style_dim = style_dim
        ##dimension of style vector

        layers = [CIPblocks.PixelNorm()]
        ##layers for latent normalization

        for i in range(n_mlp):  ##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        ##mapping network that generate style w!!

        self.styleMatrix = nn.Parameter(torch.randn(35, 512))
        # self.styleMatrix.data.fill_(0.25)
        # self.alpha = nn.Parameter(torch.rand(1,512))
        # self.alpha.data.fill_(0.5)



    def forward(self,
                label,  ##[1,35,256,512]
                label_class_dict,
                coords,  ##[1,2,256,512]
                latent,  ##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                ):
        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        # print("input latent code:",latent)
        latent = latent[0]  ##[1,512]
        ##input noirse z
        # print("received latent[0] :",latent.shape,latent)
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)
        ##style w [1,512]

        clade_params = []
        tokens_0 = self.i2t(label)
        w_param = self.w_encoder_init(tokens_0)
        b_param = self.b_encoder_init(tokens_0)
        clade_params.append((w_param,b_param))
        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            if i >= 1:
                if not self.channels[i] == self.channels[i - 1]:
                    w_param = self.linear512_256_w(w_param)
                    b_param = self.linear512_256_b(b_param)
            for j in range(self.to_rgb_stride):  ##2xModFC
                w_param = self.w_encoder[i * self.to_rgb_stride + j](w_param)
                b_param = self.b_encoder[i * self.to_rgb_stride + j](b_param)
                clade_params.append((w_param,b_param))


        x = self.lff(coords)
        x = torch.cat((x,self.lff(dist_map)),dim = 1)
        ##Fourier Features:simple linear transformation with sin activation
        ##[N,512,256,512]
        # print(x)

        batch_size, _, h, w = coords.shape

        if self.training and h == self.size[0] and w == self.size[1]:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                # Given an input and a flow-field grid,
                # computes the output using input values and pixel locations from grid.
                # input(N,C,H_in,W_in),grid(N,2,H_out,W_out),out(N,C,H_out,W_out)
                self.emb.learnable_vectors.expand(batch_size, -1, -1, -1),
                # 调用emb class的self.input!!
                # -1 means not changing the size of that dimension!!!!
                (coords.permute(0, 2, 3, 1).contiguous()),
                padding_mode='border', mode='bilinear',
            )



        x = torch.cat([x, emb], 1)
        ##concatenation of Fourier Features and Coordinates Embeddings on channel dimension!!!
        ##[1,1024,256,512]

        rgb = 0

        x = self.conv1(x,latent,
                       label_class_dict=label_class_dict,
                       label=label,
                       class_style=self.styleMatrix,
                       dist_map=dist_map,
                       clade_param= clade_params[0]
                       )
        ##first ModFC layer



        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                x = self.linears[i * self.to_rgb_stride + j](x, latent,
                                                             label_class_dict=label_class_dict,
                                                             label=label,
                                                             class_style=self.styleMatrix,
                                                             dist_map=dist_map,
                                                             clade_param=clade_params[i * self.to_rgb_stride + j+1])

            rgb = self.to_rgbs[i](x,latent,rgb,
                                  label_class_dict=label_class_dict,
                                  label=label,
                                  class_style=self.styleMatrix,
                                  dist_map=dist_map)
            ####skip=rgb ==> rgb image accumulation!!

        if return_latents:
            return rgb, latent
        else:

            # print("rgb size:",rgb.size())
            # return self.tanh(rgb), None
            return self.tanh(rgb), None



class ImplicitGenerator_tanh_skip_512_U_net(nn.Module):

    def __init__(self, opt=None, size=(256, 512), hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, z=None, **kwargs):
        super(ImplicitGenerator_tanh_skip_512_U_net, self).__init__()

        self.opt = opt
        if opt.apply_MOD_CLADE:
            self.approach = 0
        elif opt.only_CLADE:
            self.approach = 1.6
        elif opt.Matrix_Computation:
            self.approach = 2
        else:
            self.approach = -1
        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = CIPblocks.LFF(int(hidden_size/2))
        self.emb = (CIPblocks.ConstantInput(hidden_size, size=size))

        self.channels = {
            0: 512,
            1: 512,  ##512
            2: 512,  ##512
            3: 512,  ##512
            4: 512 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])

        self.i2t = Img2Token(dim=in_channels)
        self.w_encoder_init = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
        self.b_encoder_init = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
        self.conv1 = CIPblocks.StyledConv(int(multiplier* hidden_size),  ##the real in_channel 1024
                                          in_channels,  ##actually is out_channel
                                          1,
                                          style_dim,
                                          demodulate=demodulate,
                                          activation=activation,
                                          approach=self.approach,
                                          add_dist=self.add_dist# jhl
                                          )

        ###kernel_size = 1===>first modFC layer!!only one layer!!input=embbed coords!!

        self.linears = nn.ModuleList()
        self.w_encoder = nn.ModuleList()
        self.b_encoder = nn.ModuleList()
        ##2xModFC for 2-8 Layers
        self.to_rgbs = nn.ModuleList()
        ##tRGB for 2-8 Layers
        self.log_size = int(CIPblocks.math.log(max(size), 2))
        ## 8 Layers

        self.n_intermediate = self.log_size - 6
        ## intermediate layer(7 layers except first layer)
        self.to_rgb_stride = 2


        ##how many ModFC between two tRGB==>in this case, 2 ModFC layers
        for i in range(0, self.n_intermediate):  ## for each layer in intermediate 7 Layers:
            out_channels = self.channels[i]

            self.w_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))
            self.b_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))

            self.linears.append(CIPblocks.StyledConv(in_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl

            self.w_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))
            self.b_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))

            self.linears.append(CIPblocks.StyledConv(out_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl

            self.to_rgbs.append(
                CIPblocks.ToRGB(out_channels, style_dim, upsample=False,
                                approach=self.approach,
                                add_dist=self.add_dist))  # jhl
            ###upsample turned off manually
            # print(out_channels)
            in_channels = out_channels
            ##2xModFC+tRGB for 2-8 Layers

        self.style_dim = style_dim
        ##dimension of style vector

        layers = [CIPblocks.PixelNorm()]
        ##layers for latent normalization

        for i in range(n_mlp):  ##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        ##mapping network that generate style w!!

        self.styleMatrix = nn.Parameter(torch.randn(35, 512))
        # self.styleMatrix.data.fill_(0.25)
        # self.alpha = nn.Parameter(torch.rand(1,512))
        # self.alpha.data.fill_(0.5)



    def forward(self,
                label,  ##[1,35,256,512]
                label_class_dict,
                coords,  ##[1,2,256,512]
                latent,  ##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                ):
        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        # print("input latent code:",latent)
        latent = latent[0]  ##[1,512]
        ##input noirse z
        # print("received latent[0] :",latent.shape,latent)
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)
        ##style w [1,512]


        clade_params = []

        tokens_0 = self.i2t(label)
        w_param = self.w_encoder_init(tokens_0)
        b_param = self.b_encoder_init(tokens_0)
        clade_params.append((w_param,b_param))
        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                if (i * self.to_rgb_stride + j) <= 4:
                    w_param = self.w_encoder[i * self.to_rgb_stride + j](w_param)
                    b_param = self.b_encoder[i * self.to_rgb_stride + j](b_param)
                    clade_params.append((w_param,b_param))
        length = len(clade_params)


        x = self.lff(coords)
        x = torch.cat((x,self.lff(dist_map)),dim = 1)
        ##Fourier Features:simple linear transformation with sin activation
        ##[N,512,256,512]
        # print(x)

        batch_size, _, h, w = coords.shape

        if self.training and h == self.size[0] and w == self.size[1]:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                # Given an input and a flow-field grid,
                # computes the output using input values and pixel locations from grid.
                # input(N,C,H_in,W_in),grid(N,2,H_out,W_out),out(N,C,H_out,W_out)
                self.emb.learnable_vectors.expand(batch_size, -1, -1, -1),
                # 调用emb class的self.input!!
                # -1 means not changing the size of that dimension!!!!
                (coords.permute(0, 2, 3, 1).contiguous()),
                padding_mode='border', mode='bilinear',
            )



        x = torch.cat([x, emb], 1)
        ##concatenation of Fourier Features and Coordinates Embeddings on channel dimension!!!
        ##[1,1024,256,512]

        rgb = 0

        x = self.conv1(x,latent,
                       label_class_dict=label_class_dict,
                       label=label,
                       class_style=self.styleMatrix,
                       dist_map=dist_map,
                       clade_param= clade_params[-1]
                       )
        ##first ModFC layer



        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                if (i * self.to_rgb_stride + j) <= 4:
                    x = self.linears[i * self.to_rgb_stride + j](x, latent,
                                                                 label_class_dict=label_class_dict,
                                                                 label=label,
                                                                 class_style=self.styleMatrix,
                                                                 dist_map=dist_map,
                                                                 clade_param=clade_params[length-(i * self.to_rgb_stride + j)-2])

            rgb = self.to_rgbs[i](x,latent,rgb,
                                  label_class_dict=label_class_dict,
                                  label=label,
                                  class_style=self.styleMatrix,
                                  dist_map=dist_map)
            ####skip=rgb ==> rgb image accumulation!!

        if return_latents:
            return rgb, latent
        else:

            # print("rgb size:",rgb.size())
            # return self.tanh(rgb), None
            return self.tanh(rgb), None



class ImplicitGenerator_tanh_skip_512_U_net_ff(nn.Module):

    def __init__(self, opt=None, size=(256, 512), hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, z=None, **kwargs):
        super(ImplicitGenerator_tanh_skip_512_U_net_ff, self).__init__()

        self.opt = opt
        if opt.apply_MOD_CLADE:
            self.approach = 0
        elif opt.only_CLADE:
            self.approach = 1.61
        elif opt.Matrix_Computation:
            self.approach = 2
        else:
            self.approach = -1
        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = CIPblocks.LFF(int(hidden_size/2))
        self.emb = (CIPblocks.ConstantInput(hidden_size, size=size))
        self.lff16 = CIPblocks.LFF(16)

        self.channels = {
            0: 512,
            1: 512,  ##512
            2: 512,  ##512
            3: 512,  ##512
            4: 512 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])

        self.i2t = Img2Token(dim=in_channels)
        self.w_encoder_init = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
        self.b_encoder_init = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
        self.conv1 = CIPblocks.StyledConv(int(multiplier* hidden_size),  ##the real in_channel 1024
                                          in_channels,  ##actually is out_channel
                                          1,
                                          style_dim,
                                          demodulate=demodulate,
                                          activation=activation,
                                          approach=self.approach,
                                          add_dist=self.add_dist# jhl
                                          )

        ###kernel_size = 1===>first modFC layer!!only one layer!!input=embbed coords!!

        self.linears = nn.ModuleList()
        self.w_encoder = nn.ModuleList()
        self.b_encoder = nn.ModuleList()
        ##2xModFC for 2-8 Layers
        self.to_rgbs = nn.ModuleList()
        ##tRGB for 2-8 Layers
        self.log_size = int(CIPblocks.math.log(max(size), 2))
        ## 8 Layers

        self.n_intermediate = self.log_size - 6
        ## intermediate layer(7 layers except first layer)
        self.to_rgb_stride = 2


        ##how many ModFC between two tRGB==>in this case, 2 ModFC layers
        for i in range(0, self.n_intermediate):  ## for each layer in intermediate 7 Layers:
            out_channels = self.channels[i]

            self.w_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))
            self.b_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))

            self.linears.append(CIPblocks.StyledConv(in_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl

            self.w_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))
            self.b_encoder.append(nn.TransformerEncoderLayer(d_model=out_channels, nhead=8))

            self.linears.append(CIPblocks.StyledConv(out_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl

            self.to_rgbs.append(
                CIPblocks.ToRGB(out_channels, style_dim, upsample=False,
                                approach=self.approach,
                                add_dist=self.add_dist))  # jhl
            ###upsample turned off manually
            # print(out_channels)
            in_channels = out_channels
            ##2xModFC+tRGB for 2-8 Layers

        self.style_dim = style_dim
        ##dimension of style vector

        layers = [CIPblocks.PixelNorm()]
        ##layers for latent normalization

        for i in range(n_mlp):  ##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        ##mapping network that generate style w!!

        self.styleMatrix = nn.Parameter(torch.randn(35, 512))
        # self.styleMatrix.data.fill_(0.25)
        # self.alpha = nn.Parameter(torch.rand(1,512))
        # self.alpha.data.fill_(0.5)



    def forward(self,
                label,  ##[1,35,256,512]
                label_class_dict,
                coords,  ##[1,2,256,512]
                latent,  ##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                ):
        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        # print("input latent code:",latent)
        latent = latent[0]  ##[1,512]
        ##input noirse z
        # print("received latent[0] :",latent.shape,latent)
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)
        ##style w [1,512]


        clade_params = []
        input = torch.cat([label,self.lff16(coords),self.lff16(dist_map)],dim=1)
        tokens_0 = self.i2t(input)
        w_param = self.w_encoder_init(tokens_0)
        b_param = self.b_encoder_init(tokens_0)
        clade_params.append((w_param,b_param))
        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                if (i * self.to_rgb_stride + j) <= 4:
                    w_param = self.w_encoder[i * self.to_rgb_stride + j](w_param)
                    b_param = self.b_encoder[i * self.to_rgb_stride + j](b_param)
                    clade_params.append((w_param,b_param))
        length = len(clade_params)


        x = self.lff(coords)
        x = torch.cat((x,self.lff(dist_map)),dim = 1)
        ##Fourier Features:simple linear transformation with sin activation
        ##[N,512,256,512]
        # print(x)

        batch_size, _, h, w = coords.shape

        if self.training and h == self.size[0] and w == self.size[1]:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                # Given an input and a flow-field grid,
                # computes the output using input values and pixel locations from grid.
                # input(N,C,H_in,W_in),grid(N,2,H_out,W_out),out(N,C,H_out,W_out)
                self.emb.learnable_vectors.expand(batch_size, -1, -1, -1),
                # 调用emb class的self.input!!
                # -1 means not changing the size of that dimension!!!!
                (coords.permute(0, 2, 3, 1).contiguous()),
                padding_mode='border', mode='bilinear',
            )



        x = torch.cat([x, emb], 1)
        ##concatenation of Fourier Features and Coordinates Embeddings on channel dimension!!!
        ##[1,1024,256,512]

        rgb = 0

        x = self.conv1(x,latent,
                       label_class_dict=label_class_dict,
                       label=label,
                       class_style=self.styleMatrix,
                       dist_map=dist_map,
                       clade_param= clade_params[-1]
                       )
        ##first ModFC layer



        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                if (i * self.to_rgb_stride + j) <= 4:
                    x = self.linears[i * self.to_rgb_stride + j](x, latent,
                                                                 label_class_dict=label_class_dict,
                                                                 label=label,
                                                                 class_style=self.styleMatrix,
                                                                 dist_map=dist_map,
                                                                 clade_param=clade_params[length-(i * self.to_rgb_stride + j)-2])

            rgb = self.to_rgbs[i](x,latent,rgb,
                                  label_class_dict=label_class_dict,
                                  label=label,
                                  class_style=self.styleMatrix,
                                  dist_map=dist_map)
            ####skip=rgb ==> rgb image accumulation!!

        if return_latents:
            return rgb, latent
        else:

            # print("rgb size:",rgb.size())
            # return self.tanh(rgb), None
            return self.tanh(rgb), None



class ImplicitGenerator_Conv_U_net(nn.Module):

    def __init__(self, opt=None, size=(256, 512), hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, z=None, **kwargs):
        super(ImplicitGenerator_Conv_U_net, self).__init__()

        self.kernel_size = 3
        ks = self.kernel_size

        self.opt = opt
        if opt.apply_MOD_CLADE:
            self.approach = 0
        elif opt.only_CLADE:
            self.approach = 1
        elif opt.Matrix_Computation:
            self.approach = 2
        else:
            self.approach = -1
        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = CIPblocks.LFF(int(hidden_size/2))
        self.lff2 = CIPblocks.LFF(16)
        self.linears = nn.ModuleList()
        self.conv_encoder = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        # self.emb = (CIPblocks.ConstantInput(hidden_size, size=size))

        self.channels = {
            0: 512,
            1: 512,  ##512
            2: 512,  ##512
            3: 512,  ##512
            4: 512 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        in_channels = int(self.channels[0])


        self.conv1 = CIPblocks.StyledConv(int(hidden_size),  ##the real in_channel 1024
                                          in_channels,  ##actually is out_channel
                                          1,
                                          style_dim,
                                          demodulate=demodulate,
                                          activation=activation,
                                          approach=self.approach,
                                          add_dist=self.add_dist# jhl
                                          )

        ###kernel_size = 1===>first modFC layer!!only one layer!!input=embbed coords!!


        ##tRGB for 2-8 Layers
        self.log_size = int(CIPblocks.math.log(max(size), 2))
        ## 8 Layers

        self.n_intermediate = self.log_size - 6
        ## intermediate layer(7 layers except first layer)
        self.to_rgb_stride = 2


        conv_in = 67
        conv_out = 128
        ##how many ModFC between two tRGB==>in this case, 2 ModFC layers
        for i in range(0, self.n_intermediate):  ## for each layer in intermediate 7 Layers:
            out_channels = self.channels[i]

            self.conv_encoder.append(CIPblocks.conv_bn_relu(in_channels=conv_in,out_channels=conv_out,
                                               kernel_size=ks,padding=ks//2))

            self.linears.append(CIPblocks.StyledConv(in_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl

            self.conv_encoder.append(CIPblocks.conv_bn_relu(in_channels=conv_out,out_channels=conv_out,
                                               kernel_size=ks,padding=ks//2))

            self.linears.append(CIPblocks.StyledConv(out_channels, out_channels, 1, style_dim,
                                                     demodulate=demodulate, activation=activation,
                                                     approach=self.approach,
                                                     add_dist=self.add_dist))  # jhl
            self.pooling.append(nn.MaxPool2d(kernel_size=2,padding=0))


            self.to_rgbs.append(
                CIPblocks.ToRGB(out_channels, style_dim, upsample=False,
                                approach=self.approach,
                                add_dist=self.add_dist))  # jhl
            conv_in=conv_out
            conv_out=2*conv_out
            ###upsample turned off manually
            # print(out_channels)
            in_channels = out_channels
            ##2xModFC+tRGB for 2-8 Layers
        self.conv_encoder.append(CIPblocks.conv_bn_relu(in_channels=conv_in, out_channels=conv_out,
                                                        kernel_size=ks, padding=ks // 2))

        self.style_dim = style_dim
        ##dimension of style vector

        layers = [CIPblocks.PixelNorm()]
        ##layers for latent normalization

        for i in range(n_mlp):  ##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
        ##mapping network that generate style w!!

        self.styleMatrix = nn.Parameter(torch.randn(35, 512))
        # self.styleMatrix.data.fill_(0.25)
        # self.alpha = nn.Parameter(torch.rand(1,512))
        # self.alpha.data.fill_(0.5)



    def forward(self,
                label,  ##[1,35,256,512]
                label_class_dict,
                coords,  ##[1,2,256,512]
                latent,  ##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                ):
        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        # print("input latent code:",latent)
        latent = latent[0]  ##[1,512]
        ##input noirse z
        # print("received latent[0] :",latent.shape,latent)
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)
        ##style w [1,512]


        actv = []

        param = torch.cat((label,self.lff2(coords),self.lff2(dist_map)),dim=1)

        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                # if (i * self.to_rgb_stride + j) <= 4:
                param = self.conv_encoder[i * self.to_rgb_stride + j](param)

                actv.append(param)
            param = self.pooling[i](param)
        param = self.conv_encoder[-1](param)
        actv.append(param)
        length = len(actv)


        x = self.lff(coords)
        x = torch.cat((x,self.lff(dist_map)),dim = 1)
        ##Fourier Features:simple linear transformation with sin activation
        ##[N,512,256,512]
        # print(x)

        batch_size, _, h, w = coords.shape

        # if self.training and h == self.size[0] and w == self.size[1]:
        #     emb = self.emb(x)
        # else:
        #     emb = F.grid_sample(
        #         # Given an input and a flow-field grid,
        #         # computes the output using input values and pixel locations from grid.
        #         # input(N,C,H_in,W_in),grid(N,2,H_out,W_out),out(N,C,H_out,W_out)
        #         self.emb.learnable_vectors.expand(batch_size, -1, -1, -1),
        #         # 调用emb class的self.input!!
        #         # -1 means not changing the size of that dimension!!!!
        #         (coords.permute(0, 2, 3, 1).contiguous()),
        #         padding_mode='border', mode='bilinear',
        #     )



        # x = torch.cat([x, emb], 1)
        ##concatenation of Fourier Features and Coordinates Embeddings on channel dimension!!!
        ##[1,1024,256,512]

        rgb = 0

        x = self.conv1(x,latent,
                       label_class_dict=label_class_dict,
                       label=label,
                       class_style=self.styleMatrix,
                       dist_map=dist_map,
                       clade_param= param[-1]
                       )
        ##first ModFC layer



        for i in range(self.n_intermediate):  ##2-8 ModFC layers
            for j in range(self.to_rgb_stride):  ##2xModFC
                # if (i * self.to_rgb_stride + j) <= 4
                p = param[length-(i * self.to_rgb_stride + j)-2]
                if p.shape[2] <= 64:
                    p = F.interpolate(p,size=(128,256),mode='nearest')
                    # p =
                x = self.linears[i * self.to_rgb_stride + j](x, latent,
                                                             label_class_dict=label_class_dict,
                                                             label=label,
                                                             class_style=self.styleMatrix,
                                                             dist_map=dist_map,
                                                             clade_param=param[length-(i * self.to_rgb_stride + j)-2])

            rgb = self.to_rgbs[i](x,latent,rgb,
                                  label_class_dict=label_class_dict,
                                  label=label,
                                  class_style=self.styleMatrix,
                                  dist_map=dist_map)
            ####skip=rgb ==> rgb image accumulation!!

        if return_latents:
            return rgb, latent
        else:

            # print("rgb size:",rgb.size())
            # return self.tanh(rgb), None
            return self.tanh(rgb), None














class ImplicitGenerator_multi_scale_U(nn.Module):##34
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        # layers = [CIPblocks.PixelNorm()]
        #
        # for i in range(n_mlp):##mapping network for style w(in total 8 layers)
        #     layers.append(
        #         CIPblocks.EqualLinear(
        #             style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
        #         )
        #     )
        # self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))


    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        # latent = latent[0]
        # if truncation < 1:
        #     latent = truncation_latent + truncation * (latent - truncation_latent)
        # if not input_is_latent:
        #     latent = self.style(latent)

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)


        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')

        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None






class ImplicitGenerator_multi_scale_U_bilinear(nn.Module):##343
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_bilinear, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_bilinear(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)


        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None








class ImplicitGenerator_multi_scale_U_bilinear_avepool(nn.Module):##3431
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_bilinear_avepool, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder_avepool(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_bilinear(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)
        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16, label_32, label_64, label_128, label]
        latent = [latent, labels]


        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None









class ImplicitGenerator_multi_scale_U_bilinear_sle(nn.Module):##346
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_bilinear_sle, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,128,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_SLE(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)


        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None










class ImplicitGenerator_multi_scale_U_trans_conv(nn.Module):##345
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_trans_conv, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)


        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None











class ImplicitGenerator_multi_scale_U_trans_conv_separate_style(nn.Module):##345
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_trans_conv_separate_style, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_multi_class_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm_multi_class(128,35)]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(CIPblocks.EqualConv1d(in_channel=128,out_channel=128,kernel_size=1,groups=35))
            layers.append(nn.ReLU())
        self.style = nn.Sequential(*layers)

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape
        latent = CIPblocks.mixing_noise(batch_size, 35*128, 0, self.device)


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)
        latent = latent.view(latent.shape[0],35,128)
        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [latent,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None






class ImplicitGenerator_multi_scale_U_trans_conv_separate_style_no_emb_noise_input(nn.Module):##3454
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_trans_conv_separate_style_no_emb_noise_input, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_multi_class_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm_multi_class(128,35)]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(CIPblocks.EqualConv1d(in_channel=128,out_channel=128,kernel_size=1,groups=35))
            layers.append(nn.ReLU())
        self.style = nn.Sequential(*layers)

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape
        latent = CIPblocks.mixing_noise(batch_size, 35*128, 0, self.device)


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)
        latent = latent.view(latent.shape[0],35,128)
        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [latent,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,torch.randn(ff16.shape).to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None






class ImplicitGenerator_multiscaleU_transconv_35style_noembnoiseinput_noisylabel(nn.Module):##34541
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_35style_noembnoiseinput_noisylabel, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_multi_class_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm_multi_class(128,35)]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(CIPblocks.EqualConv1d(in_channel=128,out_channel=128,kernel_size=1,groups=35))
            layers.append(nn.ReLU())
        self.style = nn.Sequential(*layers)

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape
        latent = CIPblocks.mixing_noise(batch_size, 35*128, 0, self.device)


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)
        latent = latent.view(latent.shape[0],35,128)
        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [latent,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,torch.randn(ff16.shape).to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None



















class ImplicitGenerator_multiscaleU_transconv_ganformer_noembnoiseinput_noisylabel(nn.Module):##411
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_ganformer_noembnoiseinput_noisylabel, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder_ordered(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_transpose2d_ordered(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        # self.style_dim = style_dim
        # layers = [CIPblocks.PixelNorm_multi_class(128,35)]
        # for i in range(n_mlp):##mapping network for style w(in total 8 layers)
        #     layers.append(CIPblocks.EqualConv1d(in_channel=128,out_channel=128,kernel_size=1,groups=35))
        #     layers.append(nn.ReLU())
        # self.style = nn.Sequential(*layers)


################ params for ganformer ####################################################


#         self.z_dim = 32
#         self.c_dim = 0
#         self.w_dim = 32
#         self.k = 17
#         self.img_resolution = 256
#         self.img_channels = 3
#         self.component_dropout = 0
#         self.num_ws = 18## total number of output features from encoder
#         self.input_shape = [None, self.k, self.z_dim]
#         self.cond_shape = [None, self.c_dim]
#
#         self.pos = Ganformerget_embeddings(self.k - 1, self.w_dim)
# ################ mapping network for ganformer ###########################################
#         mapping_kwargs = {'num_layers': 8,
#                           'layer_dim': None,
#                           'act': 'lrelu',
#                           'lrmul': 0.01,
#                           'w_avg_beta': 0.995,
#                           'resnet': True,
#                           'ltnt2ltnt': True,
#                           'transformer': True,
#                           'num_heads': 1,
#                           'attention_dropout': 0.12,
#                           'ltnt_gate': False,
#                           'use_pos': True,
#                           'normalize_global': True}
#         self.mapping = GanformerMappingNetwork(z_dim = 32, c_dim = 0, w_dim = 32, k = 17,
#             num_broadcast = self.num_ws, **mapping_kwargs)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        synthesis_kwargs = {'crop_ratio': None,
                            'channel_base': 32768,
                            'channel_max': 512,
                            'architecture': 'resnet',
                            'resample_kernel': [1, 3, 3, 1],
                            'local_noise': True,
                            'act': 'lrelu',
                            'ltnt_stem': False,
                            'style': True,
                            'transformer': True,
                            'start_res': 0,
                            'end_res': 8,
                            'num_heads': 1,
                            'attention_dropout': 0.12,
                            'ltnt_gate': False,
                            'img_gate': False,
                            'integration': 'mul', 'norm': 'layer',
                            'kmeans': True,
                            'kmeans_iters': 1,
                            'iterative': False,
                            'use_pos': True,
                            'pos_dim': None,
                            'pos_type': 'sinus',
                            'pos_init': 'uniform',
                            'pos_directions_num': 2}
        _kwargs = {'nothing': None}
        self.bipartite_computer = bipartite_attention_computer(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            synthesis_kwargs=synthesis_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )













    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None,
                ws = None,
                ):
        batch_size,_,H,W = label.shape
        # latent = CIPblocks.mixing_noise(batch_size, 35*128, 0, self.device)

################ noise mapping for ganformer###########################################
        # z = torch.randn([batch_size, self.k, self.z_dim], device=self.device)
        # truncation_psi = 0.7
        # c = 0
        # truncation_cutoff = None
        #
        #
        # _input = z if z is not None else ws
        # mask = Ganformerrandom_dp_binary([_input.shape[0], self.k - 1], self.component_dropout, self.training, _input.device)
        # if ws is None:
        #     ws = self.mapping(z, c, pos = self.pos, mask = mask, truncation_psi = truncation_psi, truncation_cutoff = truncation_cutoff)
        # Ganformerassert_shape(ws, [None, self.k, self.num_ws, self.w_dim])
        #
        #
        #
        #
        # ret = ()
        # if return_img or return_att:
        #     img, att_maps = self.synthesis(ws, pos = self.pos, mask = mask, **synthesis_kwargs)
        #     if return_img:  ret += (img, )
        #     if return_att:  ret += (att_maps, )
        #
        # if return_ws:  ret += (ws, )
        #
        # if return_tensor:
        #     ret = ret[0]

###########################################################################################








        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]


        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,torch.randn(ff16.shape).to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)


        for i in range(len(encoder_outputs)):
            encoder_outputs[i].reverse()
        encoder_outputs.reverse()

        bipartite_kwargs = {
                            'encoder_outputs':encoder_outputs,
                            }
        z = torch.randn([batch_size, *self.bipartite_computer.input_shape[1:]], device=self.device)  # Sample latent vector
        input_for_decoder = self.bipartite_computer(z, truncation_psi=0.7,**bipartite_kwargs)






        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = input_for_decoder, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = input_for_decoder, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None













class ImplicitGenerator_multiscaleU_transconv_bipartiteencoder_noembnoiseinput(nn.Module):##412
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_bipartiteencoder_noembnoiseinput, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        synthesis_kwargs = {'crop_ratio': None,
                            'channel_base': 32768,
                            'channel_max': 512,
                            'architecture': 'resnet',
                            'resample_kernel': [1, 3, 3, 1],
                            'local_noise': True,
                            'act': 'lrelu',
                            'ltnt_stem': False,
                            'style': True,
                            'transformer': True,
                            'start_res': 0,
                            'end_res': 8,
                            'num_heads': 1,
                            'attention_dropout': 0.12,
                            'ltnt_gate': False,
                            'img_gate': False,
                            'integration': 'mul', 'norm': 'layer',
                            'kmeans': True,
                            'kmeans_iters': 1,
                            'iterative': False,
                            'use_pos': True,
                            'pos_dim': None,
                            'pos_type': 'sinus',
                            'pos_init': 'uniform',
                            'pos_directions_num': 2}
        _kwargs = {'nothing': None}
        self.encoder = BipartiteEncoder(
                                        z_dim=32,                      # Input latent (Z) dimensionality
                                        c_dim=0,                      # Conditioning label (C) dimensionality
                                        w_dim=32,                      # Intermediate latent (W) dimensionality
                                        k=17,                          # Number of latent vector components z_1,...,z_k
                                        img_resolution=256,             # Output resolution
                                        img_channels=3,               # Number of output color channels
                                        component_dropout   = 0.0,  # Dropout over the latent components, 0 = disable
                                        mapping_kwargs      = mapping_kwargs,
                                        synthesis_kwargs    = synthesis_kwargs,   # Arguments for SynthesisNetwork
                                        **_kwargs                   # Ignore unrecognized keyword args
                                          )
        self.decoder = CIPblocks.conv1x1_decoder_for_ganformerencoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        # self.style_dim = style_dim
        # layers = [CIPblocks.PixelNorm_multi_class(128,35)]
        # for i in range(n_mlp):##mapping network for style w(in total 8 layers)
        #     layers.append(CIPblocks.EqualConv1d(in_channel=128,out_channel=128,kernel_size=1,groups=35))
        #     layers.append(nn.ReLU())
        # self.style = nn.Sequential(*layers)



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None,
                ws = None,
                ):
        batch_size,_,H,W = label.shape
        # latent = CIPblocks.mixing_noise(batch_size, 35*128, 0, self.device)


        z = torch.randn([batch_size, *self.encoder.input_shape[1:]], device=self.device)  # Sample latent vector



        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]


        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]






        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,torch.randn(ff16.shape).to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        bipartite_kwargs = {
                            'nothing':None,
                            }
        encoder_outputs = self.encoder(encoder_input,z,truncation_psi=0.7,**bipartite_kwargs)


        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None












class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder(nn.Module):##413
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None






class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade(nn.Module):##4131
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_1spade(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None




class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade_catFeature(nn.Module):##41311
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade_catFeature, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_1spade_catFeature(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword argsf
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None


class ImplicitGenerator_bipDEC_1spd_catFeat_noskip(nn.Module):##413113
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipDEC_1spd_catFeat_noskip, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_1spd_catFeat_noSkip(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword argsf
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),
                                   self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),
                                   self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None



class ImplicitGenerator_noENC_bipDEC_1spade(nn.Module):##413111
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_noENC_bipDEC_1spade, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        # in_channel_en = 67
        # encoder_resolutions = [256, 128, 64, 32, 16]
        # encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        # self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
        #                                       channels_nums=encoder_channels,
        #                                       in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_1spade_noENC_catFeature(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword argsf
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        # encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        # _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(None,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None






class ImplicitGenerator_bipDC_1spd_catFeat_skip(nn.Module):##413112
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipDC_1spd_catFeat_skip, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avgpool_for_bipDC(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'skip',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_1spade_catFeat_skip(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword argsf
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None










class ImplicitGenerator_bipDEC_1spade_catlabel(nn.Module):##41312
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipDEC_1spade_catlabel, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_1spade_catLabel(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword argsf
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None








class ImplicitGenerator_bipDEC_1spade_catFeature_3D(nn.Module):##41313
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipDEC_1spade_catFeature_3D, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_1spade_catFeature_3D(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword argsf
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None








class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat(nn.Module):##4132
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_cat(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None







class ImplicitGenerator_bipartiteDEcoder_catlabel_skipSPD_3Dnoise(nn.Module):##41321
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipartiteDEcoder_catlabel_skipSPD_3Dnoise, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_catlabel_skipSPD_3Dnoise(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None






class ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb(nn.Module):##41322
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_catlabel_skipSPD_3Dnoise(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),
                                   self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),
                                   self.lff_encoder(dist_map256)],
                                   dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None




class ImplicitGenerator_noEC_bipDEC_catFeat_skipSPD_3Dnoise_noisylb(nn.Module):##413221
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_noEC_bipDEC_catFeat_skipSPD_3Dnoise_noisylb, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        # self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
        #                                       channels_nums=encoder_channels,
        #                                       in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'skip',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipDC_catFeat_skipSPD_3Dnoise_noEC(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256],
        }


        # label = torch.cat([label,torch.randn(size=[batch_size,32,label.shape[2],label.shape[3]])],dim=1)
        label = torch.cat([label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),
                                   self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),
                                   self.lff_encoder(dist_map256)],
                                   dim = 1)


        # _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(None,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None







class ImplicitGenerator_bipDEC_shallow_skipSPD_3Dnoise_noisylb(nn.Module):##413222
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipDEC_shallow_skipSPD_3Dnoise_noisylb, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 2,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_shallow_skipSPD_3Dnoise(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=25,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),
                                   self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),
                                   self.lff_encoder(dist_map256)],
                                   dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None







class ImplicitGenerator_bipDEC_shallow2_skipSPD_3Dnoise_noisylb(nn.Module):##413223
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipDEC_shallow2_skipSPD_3Dnoise_noisylb, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_shallow2_skipSPD_3Dnoise(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),
                                   self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),
                                   self.lff_encoder(dist_map256)],
                                   dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        label = encoder_input[:,:35,:,:]

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None







class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD(nn.Module):##4134
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_catFeature_skipSPD(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None


class ImplicitGenerator_bipartiteDEcoder_catLbFeat_skipSPD(nn.Module):##41341
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_bipartiteDEcoder_catLbFeat_skipSPD, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_catLbFeat_skipSPD(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        # ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None






class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_3iter33styles(nn.Module):##4135
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_3iter33styles, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_catFeature_skipSPD(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=33,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None





class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_noDistmap(nn.Module):##4136
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_noDistmap, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_catFeature_skipSPD(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(32)

        self.connector_lff_16 = CIPblocks.LFF(512)
        self.connector_lff_32 = CIPblocks.LFF(256)
        self.connector_lff_64 = CIPblocks.LFF(128)
        self.connector_lff_128 = CIPblocks.LFF(64)
        self.connector_lff_256 = CIPblocks.LFF(32)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        # dist_map16 = dict['dist_16_to_128'][16]
        # dist_map32 = dict['dist_16_to_128'][32]
        # dist_map64 = dict['dist_16_to_128'][64]
        # dist_map128 = dict['dist_16_to_128'][128]
        # dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        # ff_dist16 = self.lff_16(dist_map16)
        # ff_dist32 = self.lff_32(dist_map32)
        # ff_dist64 = self.lff_64(dist_map64)
        # ff_dist128 = self.lff_128(dist_map128)
        # ff_dist256 = self.lff_256(dist_map256)

        # connector_16 = torch.cat(
        #     (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
        #     dim=1)
        # connector_32 = torch.cat(
        #     (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
        #     dim=1)
        # connector_64 = torch.cat(
        #     (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
        #     dim=1)
        # connector_128 = torch.cat(
        #     (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
        #     dim=1)
        # connector_256 = torch.cat(
        #     (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
        #     dim=1)

        connector_16 =self.connector_lff_16(self.coords16).expand(batch_size, -1, -1, -1)
        connector_32 =self.connector_lff_32(self.coords32).expand(batch_size, -1, -1, -1)
        connector_64 =self.connector_lff_64(self.coords64).expand(batch_size, -1, -1, -1)
        connector_128 =self.connector_lff_128(self.coords128).expand(batch_size, -1, -1, -1)
        connector_256 =self.connector_lff_256(self.coords256).expand(batch_size, -1, -1, -1)


        # ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        # ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        # ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        # ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        # ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        # ffs = [ff16,ff32,ff64,ff128,ff256]


        emb = {
            'res16':  [ff_coords_16, connector_16],
            'res32':  [ff_coords_32, connector_32],
            'res64':  [ff_coords_64, connector_64],
            'res128': [ff_coords_128, connector_128],
            'res256': [ff_coords_256, connector_256]
        }





        # encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        encoder_input =torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1)],dim=1)

        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None







class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_no_spade_iter(nn.Module):##4133
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_no_spade_iter, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 3,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_cat_nospade(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )


        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None












class ImplicitGenerator_multiscaleU_bipDEC_contrastive(nn.Module):##511
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_bipDEC_contrastive, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_contrastive(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim


        self.projection_head = CIPblocks.ProjectionHead(32, proj_dim=32, proj='convmlp',)
    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )
        embbed_feature = self.projection_head(output_from_decoder[-1][-1])

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), embbed_feature






class ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_contrastive(nn.Module):##512
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_contrastive, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size


        in_channel_en = 67
        encoder_resolutions = [256, 128, 64, 32, 16]
        encoder_channels = [32, 64, 128, 256, 512]##若加逗号，输入函数后会变成turple！！！

        self.encoder = CIPblocks.Conv_encoder_avepool_for_bipartite_decoder(block_resolutions=encoder_resolutions,
                                              channels_nums=encoder_channels,
                                              in_channel=in_channel_en)

        mapping_kwargs = {'num_layers': 8,
                          'layer_dim': None,
                          'act': 'lrelu',
                          'lrmul': 0.01,
                          'w_avg_beta': 0.995,
                          'resnet': True,
                          'ltnt2ltnt': True,
                          'transformer': True,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'use_pos': True,
                          'normalize_global': True}
        decoder_kwargs = {'crop_ratio': None,
                          'channel_base': 32768,
                          'channel_max': 512,
                          'architecture': 'resnet',
                          'resample_kernel': [1, 3, 3, 1],
                          'local_noise': True,
                          'act': 'lrelu',
                          'ltnt_stem': False,
                          'style': True,
                          'transformer': True,
                          'start_res': 0,
                          'end_res': 8,
                          'num_heads': 1,
                          'attention_dropout': 0.12,
                          'ltnt_gate': False,
                          'img_gate': False,
                          'integration': 'mul',
                          'norm': 'layer',
                          'kmeans': True,
                          'kmeans_iters': 1,
                          'iterative': False,
                          'use_pos': True,
                          'pos_dim': None,
                          'pos_type': 'sinus',
                          'pos_init': 'uniform',
                          'pos_directions_num': 2}
        _kwargs = {'nothing': None}

        self.bipartite_decoder = BipartiteDecoder_cat(
            z_dim=32,  # Input latent (Z) dimensionality
            c_dim=0,  # Conditioning label (C) dimensionality
            w_dim=32,  # Intermediate latent (W) dimensionality
            k=17,  # Number of latent vector components z_1,...,z_k
            img_resolution=256,  # Output resolution
            img_channels=3,  # Number of output color channels
            component_dropout=0.0,  # Dropout over the latent components, 0 = disable
            mapping_kwargs=mapping_kwargs,
            decoder_kwargs=decoder_kwargs,  # Arguments for SynthesisNetwork
            **_kwargs  # Ignore unrecognized keyword args
        )



        self.lff_16 = CIPblocks.LFF(256)
        self.lff_32 = CIPblocks.LFF(128)
        self.lff_64 = CIPblocks.LFF(64)
        self.lff_128 = CIPblocks.LFF(32)
        self.lff_256 = CIPblocks.LFF(16)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_16 = CIPblocks.LFF(256)
        self.connector_lff_32 = CIPblocks.LFF(128)
        self.connector_lff_64 = CIPblocks.LFF(64)
        self.connector_lff_128 = CIPblocks.LFF(32)
        self.connector_lff_256 = CIPblocks.LFF(16)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim


        self.projection_head = CIPblocks.ProjectionHead(32, proj_dim=32, proj='convmlp',)


    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_16 = torch.cat(
            (self.connector_lff_16(self.coords16).expand(batch_size,-1,-1,-1),self.connector_lff_16(dist_map16)),
            dim=1)
        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),
            dim=1)
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),
            dim=1)
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),
            dim=1)
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),
            dim=1)


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = {
            'res16':  [ff16, connector_16],
            'res32':  [ff32, connector_32],
            'res64':  [ff64, connector_64],
            'res128': [ff128, connector_128],
            'res256': [ff256, connector_256]
        }


        encoder_input = torch.cat([label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)],dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        z = torch.randn([batch_size, *self.bipartite_decoder.input_shape[1:]], device=self.device)

        output_from_decoder,rgb = self.bipartite_decoder(encoder_outputs,emb,label,z, truncation_psi = 0.7 )

        embbed_feature = self.projection_head(output_from_decoder[-1][-1])

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), embbed_feature









class ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput(nn.Module):##3455
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder_resblock(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,torch.randn(ff16.shape).to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None










class ImplicitGenerator_multiscaleU_transconv_nomodulation_noemblatentinput(nn.Module):##34552
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multiscaleU_transconv_nomodulation_noemblatentinput, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = 1024
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim


        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    self.style_dim, self.style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        z = torch.randn(batch_size, 1024, dtype=torch.float32, device=self.device)
        z = self.style(z)
        z = z.view(z.size(0), self.style_dim, 1, 1)
        z = z.expand(z.size(0), self.style_dim, 16, 32)






        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,z.to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None









class ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput_noisylabel(nn.Module):##34551
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput_noisylabel, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder_resblock_tree(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,torch.randn(ff16.shape).to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label+torch.normal(0.0,0.1*torch.ones(label.shape)).to(self.device),self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None










class ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_mappednoiseinput(nn.Module):##345
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_mappednoiseinput, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67



        mapping_layers = [CIPblocks.PixelNorm()]
        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            mapping_layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.noise_mapping = nn.Sequential(*mapping_layers)



        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim



    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [None,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        decoder_input = torch.cat((ff16,self.noise_mapping(torch.randn(ff16.shape)).to(self.device)),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None











class ImplicitGenerator_multi_scale_U_trans_conv_separate_style_convdownsampling(nn.Module):##345
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_trans_conv_separate_style_convdownsampling, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_multi_class_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder_convdownsampling(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm_multi_class(128,35)]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(CIPblocks.EqualConv1d(in_channel=128,out_channel=128,kernel_size=1,groups=35))
            layers.append(nn.ReLU())
        self.style = nn.Sequential(*layers)

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape
        latent = CIPblocks.mixing_noise(batch_size, 35*128, 0, self.device)


        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)
        latent = latent.view(latent.shape[0],35,128)
        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [latent,labels]

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None















class ImplicitGenerator_multi_scale_U_trans_conv_styleGan(nn.Module):##347
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_trans_conv_styleGan, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'styleGan_like'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)
        label_16 = F.interpolate(label, (16, 32), mode='nearest')
        label_32 = F.interpolate(label, (32, 64), mode='nearest')
        label_64 = F.interpolate(label, (64, 128), mode='nearest')
        label_128 = F.interpolate(label, (128, 256), mode='nearest')
        labels = [label_16,label_32,label_64,label_128,label]
        latent = [latent,labels]


        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None











class ImplicitGenerator_multi_scale_U_transpose_avepool(nn.Module):##3451
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_transpose_avepool, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder_avepool(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_trans_con2d(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)


        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None












class ImplicitGenerator_multi_scale_U_nearest_modulation(nn.Module):##343
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_nearest_modulation,self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist
        self.tanh = nn.Tanh()

        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 67

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)

        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))

    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]

        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)


        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1),self.connector_lff_32(dist_map32)),dim=1
                                   )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1),self.connector_lff_64(dist_map64)),dim=1
                                   )
        connector_128 = torch.cat(
            (self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1),self.connector_lff_128(dist_map128)),dim=1
                                   )
        connector_256 = torch.cat(
            (self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1),self.connector_lff_256(dist_map256)),dim=1
                                   )

        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ff16 = torch.cat([ff_coords_16, ff_dist16 ], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32 ], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64 ], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128 ], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256 ], 1)
        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)

        # label64 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label128 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        # label256 = F.interpolate(label,  ##!!  interpolate can also downscale!!!
        #                       size=x.size()[2:],  ##take the H and W
        #                       mode='nearest')
        encoder_input = torch.cat((label,self.lff_encoder(self.coords256).expand(batch_size,-1,-1,-1),self.lff_encoder(dist_map256)),dim = 1)
        _, encoder_outputs = self.encoder(encoder_input)

        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs, latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs, latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None
















class ImplicitGenerator_multi_scale_U_no_dist(nn.Module):##32
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_no_dist, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist

        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()


        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 35

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_bilinear(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)



        self.lff_16 = CIPblocks.LFF(1024)
        self.lff_32 = CIPblocks.LFF(512)
        self.lff_64 = CIPblocks.LFF(256)
        self.lff_128 = CIPblocks.LFF(128)
        self.lff_256 = CIPblocks.LFF(64)


        self.connector_lff_32 = CIPblocks.LFF(1024)
        self.connector_lff_64 = CIPblocks.LFF(512)
        self.connector_lff_128 = CIPblocks.LFF(256)
        self.connector_lff_256 = CIPblocks.LFF(128)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))


    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,_ = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)

        ff16 = self.lff_16(self.coords16).expand(batch_size,-1,-1,-1)
        ff32 = self.lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        ff64 = self.lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        ff128 = self.lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        ff256 = self.lff_256(self.coords256).expand(batch_size,-1,-1,-1)


        connector_32 =self.connector_lff_32(self.coords32).expand(batch_size,-1,-1,-1)
        connector_64 =self.connector_lff_64(self.coords64).expand(batch_size,-1,-1,-1)
        connector_128 =self.connector_lff_128(self.coords128).expand(batch_size,-1,-1,-1)
        connector_256 =self.connector_lff_256(self.coords256).expand(batch_size,-1,-1,-1)
        connectors = [0,connector_32,connector_64,connector_128,connector_256]


        ffs = [ff16,ff32,ff64,ff128,ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16,emb),dim = 1)


        encoder_input =label
        _, encoder_outputs = self.encoder(encoder_input)


        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs,latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs,latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None










class ImplicitGenerator_multi_scale_U_only_decod_dist(nn.Module):##32
    def __init__(self, opt=None, **kwargs):
        super(ImplicitGenerator_multi_scale_U_only_decod_dist, self).__init__()
        self.opt = opt
        self.device = 'cpu' if opt.gpu_ids == '-1' else 'cuda'
        n_mlp = opt.n_mlp
        style_dim = opt.style_dim
        lr_mlp = opt.lr_mlp
        activation = opt.activation
        hidden_size = opt.hidden_size
        channel_multiplier = opt.channel_multiplier
        demodulate = False
        decoder_param = {}
        decoder_param['style_dim'] = style_dim
        decoder_param['demodulate'] = demodulate
        decoder_param['approach'] = 'SPADE_like_modulation'
        decoder_param['activation'] = activation
        decoder_param['add_dist'] = opt.add_dist




        # if opt.apply_MOD_CLADE:
        #     self.approach = 0
        # elif opt.only_CLADE:
        #     self.approach = 1.2
        # elif opt.Matrix_Computation:
        #     self.approach = 2
        # else:
        #     self.approach = -1

        self.add_dist = opt.add_dist

        self.tanh = nn.Tanh()


        self.final_channel = opt.final_channel
        self.input_size = opt.input_size
        num_scales = int(math.log2(256) - math.log2(self.input_size[0]) + 1)
        self.scales = []##[16,32,64,128,256]
        self.channels = []##[1024,512,256,64]
        for i1 in range(0,num_scales):
            self.scales.append(self.input_size[0]*2**i1)
            self.channels.append(self.final_channel*2**(num_scales-i1-1))
        in_channel_en = 35

        self.encoder = CIPblocks.Conv_encoder(scales=self.scales,
                                              out_channels=self.channels,
                                              in_channel=in_channel_en)
        self.decoder = CIPblocks.conv1x1_decoder(scales=self.scales,
                                                 out_channels=self.channels,
                                                 decoder_param=decoder_param)
        self.to_rgb = CIPblocks.to_rgb_block_bilinear(scales=self.scales,
                                        out_channels=self.channels,
                                        decoder_param=decoder_param)



        self.lff_16 = CIPblocks.LFF(512)
        self.lff_32 = CIPblocks.LFF(256)
        self.lff_64 = CIPblocks.LFF(128)
        self.lff_128 = CIPblocks.LFF(64)
        self.lff_256 = CIPblocks.LFF(32)

        self.lff_encoder = CIPblocks.LFF(16)

        self.connector_lff_32 = CIPblocks.LFF(512)
        self.connector_lff_64 = CIPblocks.LFF(256)
        self.connector_lff_128 = CIPblocks.LFF(128)
        self.connector_lff_256 = CIPblocks.LFF(64)

        self.coords16 = tt.convert_to_coord_format(1, 16, 32, integer_values=False,device=self.device)
        self.coords32 = tt.convert_to_coord_format(1, 32, 64, integer_values=False,device=self.device)
        self.coords64 = tt.convert_to_coord_format(1, 64, 128, integer_values=False,device=self.device)
        self.coords128 = tt.convert_to_coord_format(1, 128, 256, integer_values=False,device=self.device)
        self.coords256 = tt.convert_to_coord_format(1, 256, 512, integer_values=False,device=self.device)

        self.style_dim = style_dim

        layers = [CIPblocks.PixelNorm()]

        for i in range(n_mlp):##mapping network for style w(in total 8 layers)
            layers.append(
                CIPblocks.EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.styleMatrix = nn.Parameter(torch.randn(35,512))

        self.emb = (CIPblocks.ConstantInput(self.channels[0], size=self.input_size))


    def forward(self,
                label,##[1,35,256,512]
                label_class_dict,
                coords,##[1,2,256,512]
                latent,##1D list[Tensor(1,512)]
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                edges=None,
                dict = None
                ):
        batch_size,_,H,W = label.shape

        label_class_dict,dist_map = label_class_dict[:,0,:,:],label_class_dict[:,1:,:,:]
        latent = latent[0]
        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)
        if not input_is_latent:
            latent = self.style(latent)

        ff_coords_16 = self.lff_16(self.coords16).expand(batch_size, -1, -1, -1)
        ff_coords_32 = self.lff_32(self.coords32).expand(batch_size, -1, -1, -1)
        ff_coords_64 = self.lff_64(self.coords64).expand(batch_size, -1, -1, -1)
        ff_coords_128 = self.lff_128(self.coords128).expand(batch_size, -1, -1, -1)
        ff_coords_256 = self.lff_256(self.coords256).expand(batch_size, -1, -1, -1)

        dist_map16 = dict['dist_16_to_128'][16]
        dist_map32 = dict['dist_16_to_128'][32]
        dist_map64 = dict['dist_16_to_128'][64]
        dist_map128 = dict['dist_16_to_128'][128]
        dist_map256 = dist_map

        ff_dist16 = self.lff_16(dist_map16)
        ff_dist32 = self.lff_32(dist_map32)
        ff_dist64 = self.lff_64(dist_map64)
        ff_dist128 = self.lff_128(dist_map128)
        ff_dist256 = self.lff_256(dist_map256)

        connector_32 = torch.cat(
            (self.connector_lff_32(self.coords32).expand(batch_size, -1, -1, -1), self.connector_lff_32(dist_map32)),
            dim=1
        )
        connector_64 = torch.cat(
            (self.connector_lff_64(self.coords64).expand(batch_size, -1, -1, -1), self.connector_lff_64(dist_map64)),
            dim=1
        )
        connector_128 = torch.cat(
            (
            self.connector_lff_128(self.coords128).expand(batch_size, -1, -1, -1), self.connector_lff_128(dist_map128)),
            dim=1
        )
        connector_256 = torch.cat(
            (
            self.connector_lff_256(self.coords256).expand(batch_size, -1, -1, -1), self.connector_lff_256(dist_map256)),
            dim=1
        )

        connectors = [0, connector_32, connector_64, connector_128, connector_256]

        ff16 = torch.cat([ff_coords_16, ff_dist16], 1)
        ff32 = torch.cat([ff_coords_32, ff_dist32], 1)
        ff64 = torch.cat([ff_coords_64, ff_dist64], 1)
        ff128 = torch.cat([ff_coords_128, ff_dist128], 1)
        ff256 = torch.cat([ff_coords_256, ff_dist256], 1)
        ffs = [ff16, ff32, ff64, ff128, ff256]
        emb = self.emb(ff16)
        decoder_input = torch.cat((ff16, emb), dim=1)


        encoder_input =label
        _, encoder_outputs = self.encoder(encoder_input)


        _, decoder_outputs = self.decoder(decoder_input, ffs, connectors,outputs_from_encoder = encoder_outputs,latent=latent)
        rgb_input = []
        for output in decoder_outputs:
            rgb_input.append(output[-1])
        rgb = self.to_rgb(rgb_input,outputs_from_encoder = encoder_outputs,latent=latent)

        if return_latents:
            return rgb, latent
        else:
            return self.tanh(rgb), None





























































class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        self.add_edges = 1 if opt.add_edges else 0
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin+self.add_edges, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle+self.add_edges, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin+self.add_edges, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin+self.add_edges, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle+self.add_edges, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin+self.add_edges, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg, edges = None):


        if self.learned_shortcut:
            if self.add_edges:
                edges = F.interpolate(edges, size=x.shape[-2:])
                x = torch.cat([x, edges], dim=1)
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
            if self.add_edges:
                edges = F.interpolate(edges, size=x.shape[-2:])
                x = torch.cat([x, edges], dim=1)

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        if self.add_edges :
            dx = torch.cat([dx,edges],dim = 1)
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

class ResnetBlock_with_IWT_SPADE_HWT(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.IWT_SPADE_HWT(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.IWT_SPADE_HWT(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.IWT_SPADE_HWT(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

class ResBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.add_edges = 1 if opt.add_edges else 0
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin+self.add_edges, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle+self.add_edges, fout, kernel_size=3, padding=1))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin+self.add_edges, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle+self.add_edges, spade_conditional_input_dims)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg, edges = None):

        if self.add_edges:
            edges = F.interpolate(edges, size=x.shape[-2:])
            x = torch.cat([x, edges], dim=1)
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        if self.add_edges :
            dx = torch.cat([dx,edges],dim = 1)
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = dx
        return out

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel,opt, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        sp_norm = norms.get_spectral_norm(opt)

        if upsample:
            self.upsample = nn.Upsample(scale_factor = 2)

        self.conv = sp_norm(nn.Conv2d(in_channel, 3, 1, 1,padding_mode='reflect'))

    def forward(self, input, skip=None):
        out = self.conv(F.leaky_relu(input,2e1))

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class wavelet_generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.init_W = self.init_W // 2
        self.init_H = self.init_H // 2
        #self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.conv_img = ToRGB_wavelet(self.channels[-1])
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(ToRGB_wavelet(in_channel=self.channels[i+1]))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4 * 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4 * 16 * ch, 3, padding=1)

        self.iwt = InverseHaarTransform(3)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
        else :
            x = self.conv_img(x)

        x = self.iwt(x)
        x = F.tanh(x)

        return x

class ToRGB_wavelet(nn.Module):
    def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.iwt = InverseHaarTransform(3)
            self.upsample = Upsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = nn.Conv2d(in_channel, 3 * 4,1,1)

    def forward(self, input, skip=None):
        out = self.conv(input)

        if skip is not None:
            skip = self.iwt(skip)
            skip = self.upsample(skip)
            skip = self.dwt(skip)

            out = out + skip

        return out

class WaveletUpsample(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        if factor != 2 :
            print('wavelet upsampling is not implemented for factors different than 2')

        self.factor = factor
        self.w = torch.zeros(self.factor, self.factor)
        self.w[0, 0] = 1



    def forward(self, input):
        output = F.conv_transpose2d(input, self.w.expand(input.size(1), 1, self.factor, self.factor),
                                  stride=self.factor, groups=input.size(1))
        output[...,0:input.size(-2),0:input.size(-1)]=2*input
        return output


class WaveletUpsample2(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        if factor != 2 :
            print('wavelet upsampling is not implemented for factors different than 2')

        self.factor = factor
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')




    def forward(self, input):
        output = self.upsample(input)
        output[...,0:input.size(-2),0:input.size(-1)]=2*input
        return output


class WaveletUpsampleChannels(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        if factor != 2 :
            print('wavelet upsampling is not implemented for factors different than 2')

        self.factor = factor
        self.w = torch.zeros(self.factor, self.factor).cuda()
        self.w[0, 0] = 1

        self.iwt = InverseHaarTransform(3)



    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)

        ll_out = self.iwt(input)
        lh_out = F.conv_transpose2d(lh, self.w.expand(lh.size(1), 1, self.factor, self.factor),
                                  stride=self.factor, groups=lh.size(1))
        hl_out = F.conv_transpose2d(hl, self.w.expand(hl.size(1), 1, self.factor, self.factor),
                                    stride=self.factor, groups=hl.size(1))
        hh_out = F.conv_transpose2d(hh, self.w.expand(hh.size(1), 1, self.factor, self.factor),
                                    stride=self.factor, groups=hh.size(1))
        output=torch.cat((ll_out,lh_out,hl_out,hh_out),dim=1)

        return output

class ReductiveWaveletUpsampleChannels(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        if factor != 2 :
            print('wavelet upsampling is not implemented for factors different than 2')

        self.factor = factor

        self.iwt = InverseHaarTransform(3)



    def forward(self, input):

        ll_out = self.iwt(input)

        output=ll_out

        return output

class IWT_Upsample_HWT(nn.Module):
    def __init__(self, factor=2,mode='nearest'):
        super().__init__()
        if factor != 2 :
            print('wavelet upsampling is not implemented for factors different than 2')

        self.factor = factor

        self.iwt = InverseHaarTransform(3)
        self.up = nn.Upsample(scale_factor=factor,mode=mode)
        self.hwt = HaarTransform(3)



    def forward(self, input):

        output = self.iwt(input)
        output = self.up(output)
        output = self.hwt(output)

        return output


class wavelet_generator_multiple_levels(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.init_W = self.init_W // 2
        self.init_H = self.init_H // 2
        #self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.conv_img = ToRGB_wavelet(self.channels[-1])
        self.up = WaveletUpsampleChannels(factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(ToRGB_wavelet(in_channel=self.channels[i+1]))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4 * 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4 * 16 * ch, 3, padding=1)

        self.iwt = InverseHaarTransform(3)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
        else :
            x = self.conv_img(x)

        x = self.iwt(x)
        x = F.tanh(x)

        return x

class wavelet_generator_multiple_levels_no_tanh(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.init_W = self.init_W // 2
        self.init_H = self.init_H // 2
        #self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.conv_img = ToRGB_wavelet(self.channels[-1])
        self.up = WaveletUpsampleChannels(factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(ToRGB_wavelet(in_channel=self.channels[i+1]))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4 * 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4 * 16 * ch, 3, padding=1)

        self.iwt = InverseHaarTransform(3)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
        else :
            x = self.conv_img(x)

        x = self.iwt(x)

        return x

class IWT_spade_upsample_WT_generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.init_W = self.init_W // 2
        self.init_H = self.init_H // 2
        #self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.conv_img = ToRGB_wavelet(self.channels[-1])
        self.up = IWT_Upsample_HWT(factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_IWT_SPADE_HWT(self.channels[i], self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(ToRGB_wavelet(in_channel=self.channels[i+1]))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4 * 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4 * 16 * ch, 3, padding=1)

        self.iwt = InverseHaarTransform(3)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
        else :
            x = self.conv_img(x)

        x = self.iwt(x)
        x = F.tanh(x)

        return x

class wavelet_generator_multiple_levels_reductive_upsample(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.init_W = self.init_W // 2
        self.init_H = self.init_H // 2
        #self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.conv_img = ToRGB_wavelet(self.channels[-1])
        self.up = ReductiveWaveletUpsampleChannels(factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i]//4, self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(ToRGB_wavelet(in_channel=self.channels[i+1]))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4 * 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4 * 16 * ch, 3, padding=1)

        self.iwt = InverseHaarTransform(3)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
        else :
            x = self.conv_img(x)

        x = self.iwt(x)

        return x




class IWT_spade_upsample_WT_reductive_upsample_generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.init_W = self.init_W // 2
        self.init_H = self.init_H // 2
        #self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.conv_img = ToRGB_wavelet(self.channels[-1])
        self.up = ReductiveWaveletUpsampleChannels(factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_IWT_SPADE_HWT(self.channels[i]//4, self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(ToRGB_wavelet(in_channel=self.channels[i+1]))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4 * 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4 * 16 * ch, 3, padding=1)

        self.iwt = InverseHaarTransform(3)


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
        else :
            x = self.conv_img(x)

        x = self.iwt(x)
        x = F.tanh(x)

        return x

class progGrow_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.add_edges = 1 if opt.add_edges else 0
        #self.conv_img =
        # (self.channels[-1])
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(progGrow_ToRGB(in_channel=self.channels[i+1],opt = opt))
        """        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim+self.add_edges, 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc+self.add_edges, 16 * ch, 3, padding=1)"""

        self.constant_input = ConstantInput(self.channels[0],(self.init_W, self.init_H))

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        if self.add_edges :
            x = F.interpolate(torch.cat((seg,edges),dim = 1), size=(self.init_W, self.init_H))
        else :
            x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.constant_input(seg)
        #x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg,edges)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x,seg)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,seg,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
            #x = F.tanh(x)
        else :
            x = self.conv_img(F.leaky_relu(x, 2e-1))
            x = F.tanh(x)

        return x



class progGrow_ToRGB(nn.Module):
    def __init__(self, in_channel,opt, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        sp_norm = norms.get_spectral_norm(opt)

        if upsample:
            self.upsample = nn.Upsample(scale_factor = 2,mode='bilinear')

        self.conv = nn.Conv2d(in_channel, 3, 1, 1)

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm = norms.SPADE(opt, in_channel, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input,seg, skip=None):
        out = self.conv(input)

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConstantInput(nn.Module):
    def __init__(self, channel, size=(8,4)):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class ResidualWaveletGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]

        self.init_W, self.init_H = self.compute_latent_vector_size(opt)

        self.conv_img = ToRGB_wavelet(in_channel=self.channels[-1],upsample = False)
        self.iwt = InverseHaarTransform(3)

        self.up = nn.Upsample(scale_factor=2)
        self.up_residual = IWT_Upsample_HWT(factor=2,mode='bilinear')
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(WaveletBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))

        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4*16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4*16 * ch, 3, padding=1)

#        self.constant_input = ConstantInput(self.channels[0],(self.init_W, self.init_H))

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h//2, w//2

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)


        x = F.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x_s,x = self.body[i](x, seg,edges)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
                x_s = self.up_residual(x_s)
                x = x+x_s


        x = self.conv_img(x+x_s)
        x = self.iwt(x)
        x = F.tanh(x)

        return x

class WaveletBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg, edges = None):


        if self.learned_shortcut:
            #x_s = self.conv_s(x)
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        return x_s,dx


class ResidualWaveletGenerator_1(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]

        self.init_W, self.init_H = self.compute_latent_vector_size(opt)

        self.conv_img = ToRGB_wavelet(in_channel=self.channels[-1],upsample = False)
        self.iwt = InverseHaarTransform(3)

        self.up = nn.Upsample(scale_factor=2)
        self.up_residual = IWT_Upsample_HWT(factor=2,mode='bilinear')
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(WaveletBlock_with_IWT_SPADE_HWT(self.channels[i], self.channels[i+1], opt))

        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4*16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4*16 * ch, 3, padding=1)

#        self.constant_input = ConstantInput(self.channels[0],(self.init_W, self.init_H))

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h//2, w//2

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)


        x = F.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x_s,x = self.body[i](x, seg,edges)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
                x_s = self.up_residual(x_s)
                x = x+x_s


        x = self.conv_img(x+x_s)
        x = self.iwt(x)
        x = F.tanh(x)

        return x

    def forward_determinstic(self, input, noise_vector):
        seg = input
        edges = None
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = noise_vector.to(dev)
            seg = torch.cat((z, seg), dim = 1)


        x = F.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x_s,x = self.body[i](x, seg,edges)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
                x_s = self.up_residual(x_s)
                x = x+x_s


        x = self.conv_img(x+x_s)
        x = self.iwt(x)
        x = F.tanh(x)

        return x


class WaveletBlock_with_IWT_SPADE_HWT(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.IWT_SPADE_HWT(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.IWT_SPADE_HWT(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.IWT_SPADE_HWT(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg, edges = None):


        if self.learned_shortcut:
            x_s = self.conv_s(x)
            #x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        return x_s,dx


class WaveletBlock_with_SPADE_residual_too(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg, edges = None):


        if self.learned_shortcut:
            #x_s = self.conv_s(x)
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        return x_s,dx


class ResidualWaveletGenerator_2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]

        self.init_W, self.init_H = self.compute_latent_vector_size(opt)

        self.conv_img = ToRGB_wavelet(in_channel=self.channels[-1],upsample = False)
        self.iwt = InverseHaarTransform(3)

        self.up = nn.Upsample(scale_factor=2)
        self.up_residual = IWT_Upsample_HWT(factor=2,mode='bilinear')
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(WaveletBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))

        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4*16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4*16 * ch, 3, padding=1)

#        self.constant_input = ConstantInput(self.channels[0],(self.init_W, self.init_H))

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h//2, w//2

    def forward(self, input, z=None,edges = None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)


        x = F.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x_s,x = self.body[i](x, seg,edges)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
                x_s = self.up_residual(x_s)
                x = x+x_s


        x = self.conv_img(x+x_s)
        x = self.iwt(x)

        return x

