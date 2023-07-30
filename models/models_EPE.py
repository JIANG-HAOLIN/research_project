from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import torch
from torch.nn import init
import models.losses as losses
from models.CannyFilter import CannyFilter
# from torchstat import stat


from models.EPEdiscriminator import  epe_discriminators as EPE_D
from models.EPEdiscriminator import  vgg16
from models.EPEdiscriminator.discriminator_losses import  LSLoss

from pretrained_models.msegsemantic.mseg_semantic.tool.universal_demo_batched import pretrained_mseg_model
from pretrained_models.msegsemantic.mseg_semantic.tool.universal_demo_batched_finetune import pretrained_mseg_model_35_ft
from models.EPEdiscriminator.compute_result import mapping_batched_labelmap_35_24,map_194_to_24

from torch import nn, autograd, optim



## ganformer begin
# device_models = 'cuda'

def tee_loss(x, y):
    return x+y, y.detach()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def real_penalty(loss, real_img):
    ''' Compute penalty on real images. '''
    b = real_img.shape[0]
    grad_out = autograd.grad(outputs=loss, inputs=[real_img], create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
    reg_loss = torch.cat([g.pow(2).reshape(b, -1).sum(dim=1, keepdim=True) for g in grad_out if g is not None], 1).mean()
    return reg_loss








class EPE_model(nn.Module):
    def __init__(self, opt):
        super(EPE_model, self).__init__()
        self.opt = opt
        self.device_models = 'cpu' if self.opt.gpu_ids == '-1' else 'cuda'
        #--- generator and discriminator ---

        if opt.netG == 2:
            self.netG = generators.ImplicitGenerator_tanh(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 26:
            self.netG = generators.ImplicitGenerator_tanh_no_emb(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 27:
            self.netG = generators.ImplicitGenerator_tanh_skip(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 28:
            self.netG = generators.ImplicitGenerator_tanh_skip_512_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 29:
            self.netG = generators.ImplicitGenerator_Conv_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 34:
            self.netG = generators.ImplicitGenerator_multi_scale_U(opt=opt,dict=dict)
        if opt.netG == 341:
            self.netG = generators.ImplicitGenerator_multi_scale_U_no_dist(opt=opt,dict=dict)
        if opt.netG == 342:
            self.netG = generators.ImplicitGenerator_multi_scale_U_only_decod_dist(opt=opt,dict=dict)
        if opt.netG == 343:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear(opt=opt,dict=dict)
        if opt.netG == 3431:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_avepool(opt=opt,dict=dict)
        if opt.netG == 344:
            self.netG = generators.ImplicitGenerator_multi_scale_U_nearest_modulation(opt=opt,dict=dict)
        if opt.netG == 345:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv(opt=opt,dict=dict)
        if opt.netG == 3451:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transpose_avepool(opt=opt,dict=dict)
        if opt.netG == 3452:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style(opt=opt,dict=dict)
        if opt.netG == 3453:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_convdownsampling(opt=opt,dict=dict)
        if opt.netG == 3454:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_no_emb_noise_input(opt=opt,dict=dict)
        if opt.netG == 34541:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_35style_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 3455:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput(opt=opt,dict=dict)
        if opt.netG == 34552:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomodulation_noemblatentinput(opt=opt,dict=dict)

        if opt.netG == 34551:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 346:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_sle(opt=opt,dict=dict)
        if opt.netG == 347:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_styleGan(opt=opt,dict=dict)
        if opt.netG == 411:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_ganformer_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 412:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_bipartiteencoder_noembnoiseinput(opt=opt,dict=dict)
        if opt.netG == 413:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder(opt=opt,dict=dict)
        if opt.netG == 4131:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade(opt=opt,dict=dict)
        if opt.netG == 41311:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade_catFeature(opt=opt,dict=dict)
        if opt.netG == 413111:
            self.netG = generators.ImplicitGenerator_noENC_bipDEC_1spade(opt=opt,dict=dict)
        if opt.netG == 413112:
            self.netG = generators.ImplicitGenerator_bipDC_1spd_catFeat_skip(opt=opt,dict=dict)
        if opt.netG == 41312:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catlabel(opt=opt,dict=dict)
        if opt.netG == 41313:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catFeature_3D(opt=opt,dict=dict)
        if opt.netG == 4132:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat(opt=opt,dict=dict)
        if opt.netG == 41321:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catlabel_skipSPD_3Dnoise(opt=opt,dict=dict)
        if opt.netG == 41322:
            self.netG = generators.ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413221:
            self.netG = generators.ImplicitGenerator_noEC_bipDEC_catFeat_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413222:
            self.netG = generators.ImplicitGenerator_bipDEC_shallow_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 4133:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_no_spade_iter(opt=opt,dict=dict)
        if opt.netG == 4134:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD(opt=opt,dict=dict)
        if opt.netG == 41341:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catLbFeat_skipSPD(opt=opt,dict=dict)
        if opt.netG == 4135:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_3iter33styles(opt=opt,dict=dict)
        if opt.netG == 4136:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_noDistmap(opt=opt,dict=dict)
        if opt.phase == "train":
            if opt.netD == 'EPE_origin':
                print('starting to construct discriminator')
                self.netD = EPE_D.PerceptualProjectionDiscEnsemble().to(self.device_models)
                print('discriminator constructed')
                self.vgg = vgg16.VGG16().to(self.device_models)
                print('pretrained vgg inited')
                self.seg_net_pretrained = pretrained_mseg_model(device=self.device_models)
                print('pretrained mseg inited')
                self.run_discs = [True] * len(self.netD)
                self.gan_loss = LSLoss()
                self.reg_weight = 0.03

        self.print_parameter_count()
        self.init_networks_EPE()
        #--- EMA of generator weights ---
        with torch.no_grad():
           self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
            # pass
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()

        self.coords=None
        self.latent=None

    def forward(self, image, label,label_class_dict, mode, losses_computer,
                converted=None, latent = None,z=None,edges = None,
                dict = None):
        # Branching is applied to be compatible with DataParallel
        self.coords=converted

        self.latent=latent
        # print("input latent code:",self.latent)
        if mode == "losses_G":
            loss_G = 0
            label_map = label_class_dict[:,:1,:,:]
            labelmap_24,label_24 = mapping_batched_labelmap_35_24(label_map,device=self.device_models)
            fake,_ = self.netG(
                            label=label,
                            label_class_dict=label_class_dict,
                            coords=converted,
                            latent=latent,
                            return_latents=False,
                            truncation=1,
                            truncation_latent=None,
                            input_is_latent=False,
                            dict = dict
                            )


            loss_G_vgg = None

            robust_labels_logits = self.seg_net_pretrained.execute(input=fake, label=label)
            robust_labels_24 = map_194_to_24(robust_labels_logits)
            loss_G_adv = losses_computer.loss(robust_labels_24, label_24, for_real=True)
            loss_G += loss_G_adv
            realism_maps = self.netD.forward(vgg=self.vgg, img=fake, robust_labels=robust_labels_logits,
                                                fix_input=False, run_discs=self.run_discs)
            loss_G_realism = 0
            for i, rm in enumerate(realism_maps):
                loss_G_realism, _ = tee_loss(loss_G_realism, self.gan_loss.forward_gen(input=rm).mean())
            loss_G += loss_G_realism
            loss_G.backward()
            return loss_G, [loss_G_adv, loss_G_vgg],[loss_G_realism.detach()]




        if mode == "losses_D":
            with torch.no_grad():
                # fake = self.netG(label)
                rep_fake, _ = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                    dict = dict
                )
                robust_labels_logits = self.seg_net_pretrained.execute(input=rep_fake, label=label)
                robust_labels_logits_real = self.seg_net_pretrained.execute(input=image, label=label)
            loss_D_lm = None

            rec_fake = rep_fake.detach()
            rec_fake.requires_grad_()

            # forward fake images
            realism_maps = self.netD.forward(vgg=self.vgg, img=rec_fake, robust_labels=robust_labels_logits,
                                                                fix_input=True, run_discs=self.run_discs)

            loss_D_fake = 0
            pred_labels = {}  # for adaptive backprop
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_fake,_ = tee_loss(loss_D_fake, self.gan_loss.forward_fake(input=rm).mean())
                pass
            del rm
            del realism_maps

            loss_D_fake.backward()

            image.requires_grad_()

            # forward real images
            realism_maps = self.netD.forward(
                            vgg=self.vgg, img=image, robust_labels=robust_labels_logits_real, robust_img=image,
                            fix_input=(self.reg_weight <= 0), run_discs=self.run_discs)

            loss_D_real = 0
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_real += self.gan_loss.forward_real(input=rm).mean()

            del rm
            del realism_maps

            # compute gradient penalty on real images
            if self.reg_weight > 0:
                loss_D_real.backward(retain_graph=True)
                reg_loss, _ = tee_loss(0, real_penalty(loss_D_real, image))
                (self.reg_weight * reg_loss).backward()
            else:
                loss_D_real.backward()

            return loss_D_fake.detach()+loss_D_real.detach(), [loss_D_fake, loss_D_real, loss_D_lm],





        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
                else:
                    fake = self.netEMA(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)

    def init_networks_EPE(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False\
                    and ("DiscEnsemble" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)


    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges




class EPE_model_unet_ft(nn.Module):
    def __init__(self, opt):
        super(EPE_model_unet_ft, self).__init__()
        self.opt = opt
        self.device_models = 'cpu' if self.opt.gpu_ids == '-1' else 'cuda'
        #--- generator and discriminator ---

        if opt.netG == 2:
            self.netG = generators.ImplicitGenerator_tanh(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 26:
            self.netG = generators.ImplicitGenerator_tanh_no_emb(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 27:
            self.netG = generators.ImplicitGenerator_tanh_skip(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 28:
            self.netG = generators.ImplicitGenerator_tanh_skip_512_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 29:
            self.netG = generators.ImplicitGenerator_Conv_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 34:
            self.netG = generators.ImplicitGenerator_multi_scale_U(opt=opt,dict=dict)
        if opt.netG == 341:
            self.netG = generators.ImplicitGenerator_multi_scale_U_no_dist(opt=opt,dict=dict)
        if opt.netG == 342:
            self.netG = generators.ImplicitGenerator_multi_scale_U_only_decod_dist(opt=opt,dict=dict)
        if opt.netG == 343:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear(opt=opt,dict=dict)
        if opt.netG == 3431:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_avepool(opt=opt,dict=dict)
        if opt.netG == 344:
            self.netG = generators.ImplicitGenerator_multi_scale_U_nearest_modulation(opt=opt,dict=dict)
        if opt.netG == 345:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv(opt=opt,dict=dict)
        if opt.netG == 3451:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transpose_avepool(opt=opt,dict=dict)
        if opt.netG == 3452:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style(opt=opt,dict=dict)
        if opt.netG == 3453:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_convdownsampling(opt=opt,dict=dict)
        if opt.netG == 3454:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_no_emb_noise_input(opt=opt,dict=dict)
        if opt.netG == 34541:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_35style_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 3455:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput(opt=opt,dict=dict)
        if opt.netG == 34552:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomodulation_noemblatentinput(opt=opt,dict=dict)

        if opt.netG == 34551:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 346:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_sle(opt=opt,dict=dict)
        if opt.netG == 347:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_styleGan(opt=opt,dict=dict)
        if opt.netG == 411:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_ganformer_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 412:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_bipartiteencoder_noembnoiseinput(opt=opt,dict=dict)
        if opt.netG == 413:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder(opt=opt,dict=dict)
        if opt.netG == 4131:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade(opt=opt,dict=dict)
        if opt.netG == 41311:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade_catFeature(opt=opt,dict=dict)
        if opt.netG == 413111:
            self.netG = generators.ImplicitGenerator_noENC_bipDEC_1spade(opt=opt,dict=dict)
        if opt.netG == 413112:
            self.netG = generators.ImplicitGenerator_bipDC_1spd_catFeat_skip(opt=opt,dict=dict)
        if opt.netG == 41312:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catlabel(opt=opt,dict=dict)
        if opt.netG == 41313:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catFeature_3D(opt=opt,dict=dict)
        if opt.netG == 4132:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat(opt=opt,dict=dict)
        if opt.netG == 41321:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catlabel_skipSPD_3Dnoise(opt=opt,dict=dict)
        if opt.netG == 41322:
            self.netG = generators.ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413221:
            self.netG = generators.ImplicitGenerator_noEC_bipDEC_catFeat_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413222:
            self.netG = generators.ImplicitGenerator_bipDEC_shallow_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413227:
            self.netG = generators.ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb_simplex(opt=opt,dict=dict)
        if opt.netG == 4133:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_no_spade_iter(opt=opt,dict=dict)
        if opt.netG == 4134:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD(opt=opt,dict=dict)
        if opt.netG == 41341:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catLbFeat_skipSPD(opt=opt,dict=dict)
        if opt.netG == 4135:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_3iter33styles(opt=opt,dict=dict)
        if opt.netG == 4136:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_noDistmap(opt=opt,dict=dict)
        if opt.phase == "train":
            print('starting to construct discriminator')
            self.netD = EPE_D.PerceptualProjectionDiscEnsemble().to(self.device_models)
            print('discriminator constructed')
            self.vgg = vgg16.VGG16().to(self.device_models)
            print('pretrained vgg inited')
            self.seg = discriminators.OASIS_Discriminator(opt).to(self.device_models)
            self.seg.layer_up_last = nn.Conv2d(64, 35, 1, 1, 0)
            self.seg.load_state_dict(torch.load(opt.seg_path, map_location=self.device_models))

            for param in self.seg.parameters():
                param.requires_grad=False

            print('pretrained Unet seg inited')
            self.run_discs = [True] * len(self.netD)
            self.gan_loss = LSLoss()
            self.reg_weight = 0.03

        self.print_parameter_count()
        self.init_networks_EPE()
        #--- EMA of generator weights ---
        with torch.no_grad():
           self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
            # pass
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()

        self.coords=None
        self.latent=None

    def forward(self, image, label,label_class_dict, mode, losses_computer,
                converted=None, latent = None,z=None,edges = None,
                dict = None):
        # Branching is applied to be compatible with DataParallel
        self.coords=converted

        self.latent=latent
        # print("input latent code:",self.latent)
        if mode == "losses_G":
            loss_G = 0
            label_map = label_class_dict[:,:1,:,:]
            fake,_ = self.netG(
                            label=label,
                            label_class_dict=label_class_dict,
                            coords=converted,
                            latent=latent,
                            return_latents=False,
                            truncation=1,
                            truncation_latent=None,
                            input_is_latent=False,
                            dict = dict
                            )


            loss_G_vgg = None

            robust_labels_logits = self.seg(input=fake)
            loss_G_adv = losses_computer.loss(robust_labels_logits, label)
            loss_G += loss_G_adv.mean()
            realism_maps = self.netD.forward(vgg=self.vgg, img=fake, robust_labels=robust_labels_logits,
                                                fix_input=False, run_discs=self.run_discs)
            loss_G_realism = 0
            for i, rm in enumerate(realism_maps):
                loss_G_realism, _ = tee_loss(loss_G_realism, self.gan_loss.forward_gen(input=rm).mean())
            loss_G += loss_G_realism
            loss_G.backward()
            return loss_G, [loss_G_adv, loss_G_vgg],[loss_G_realism.detach()]




        if mode == "losses_D":
            with torch.no_grad():
                # fake = self.netG(label)
                rep_fake, _ = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                    dict = dict
                )
                robust_labels_logits = self.seg(input=rep_fake)
            robust_labels_logits_real = self.seg(input=image)
            loss_D_lm = None

            rec_fake = rep_fake.detach()
            rec_fake.requires_grad_()

            # loss_S = 0
            # loss_S_real = losses_computer.loss(robust_labels_logits_real, label)
            # loss_S += loss_S_real
            # loss_S = loss_S.mean()
            # loss_S.backward()

            # forward fake images
            realism_maps = self.netD.forward(vgg=self.vgg, img=rec_fake, robust_labels=robust_labels_logits,
                                                                fix_input=True, run_discs=self.run_discs)

            loss_D_fake = 0
            pred_labels = {}  # for adaptive backprop
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_fake,_ = tee_loss(loss_D_fake, self.gan_loss.forward_fake(input=rm).mean())
                pass
            del rm
            del realism_maps

            loss_D_fake.backward()

            image.requires_grad_()

            # forward real images
            realism_maps = self.netD.forward(
                            vgg=self.vgg, img=image, robust_labels=robust_labels_logits_real, robust_img=image,
                            fix_input=(self.reg_weight <= 0), run_discs=self.run_discs)

            loss_D_real = 0
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_real += self.gan_loss.forward_real(input=rm).mean()

            del rm
            del realism_maps

            # compute gradient penalty on real images
            if self.reg_weight > 0:
                loss_D_real.backward(retain_graph=True)
                reg_loss, _ = tee_loss(0, real_penalty(loss_D_real, image))
                (self.reg_weight * reg_loss).backward()
            else:
                loss_D_real.backward()

            return loss_D_fake.detach()+loss_D_real.detach(), [loss_D_fake, loss_D_real, loss_D_lm],





        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
                else:
                    fake = self.netEMA(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)

    def init_networks_EPE(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False\
                    and ("DiscEnsemble" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)


    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges


class EPE_model_mseg_ft(nn.Module):
    def __init__(self, opt):
        super(EPE_model_mseg_ft, self).__init__()
        self.opt = opt
        self.device_models = 'cpu' if self.opt.gpu_ids == '-1' else 'cuda'
        #--- generator and discriminator ---

        if opt.netG == 2:
            self.netG = generators.ImplicitGenerator_tanh(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 26:
            self.netG = generators.ImplicitGenerator_tanh_no_emb(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 27:
            self.netG = generators.ImplicitGenerator_tanh_skip(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 28:
            self.netG = generators.ImplicitGenerator_tanh_skip_512_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 29:
            self.netG = generators.ImplicitGenerator_Conv_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 34:
            self.netG = generators.ImplicitGenerator_multi_scale_U(opt=opt,dict=dict)
        if opt.netG == 341:
            self.netG = generators.ImplicitGenerator_multi_scale_U_no_dist(opt=opt,dict=dict)
        if opt.netG == 342:
            self.netG = generators.ImplicitGenerator_multi_scale_U_only_decod_dist(opt=opt,dict=dict)
        if opt.netG == 343:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear(opt=opt,dict=dict)
        if opt.netG == 3431:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_avepool(opt=opt,dict=dict)
        if opt.netG == 344:
            self.netG = generators.ImplicitGenerator_multi_scale_U_nearest_modulation(opt=opt,dict=dict)
        if opt.netG == 345:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv(opt=opt,dict=dict)
        if opt.netG == 3451:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transpose_avepool(opt=opt,dict=dict)
        if opt.netG == 3452:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style(opt=opt,dict=dict)
        if opt.netG == 3453:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_convdownsampling(opt=opt,dict=dict)
        if opt.netG == 3454:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_no_emb_noise_input(opt=opt,dict=dict)
        if opt.netG == 34541:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_35style_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 3455:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput(opt=opt,dict=dict)
        if opt.netG == 34552:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomodulation_noemblatentinput(opt=opt,dict=dict)

        if opt.netG == 34551:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 346:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_sle(opt=opt,dict=dict)
        if opt.netG == 347:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_styleGan(opt=opt,dict=dict)
        if opt.netG == 411:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_ganformer_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 412:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_bipartiteencoder_noembnoiseinput(opt=opt,dict=dict)
        if opt.netG == 413:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder(opt=opt,dict=dict)
        if opt.netG == 4131:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade(opt=opt,dict=dict)
        if opt.netG == 41311:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade_catFeature(opt=opt,dict=dict)
        if opt.netG == 413111:
            self.netG = generators.ImplicitGenerator_noENC_bipDEC_1spade(opt=opt,dict=dict)
        if opt.netG == 413112:
            self.netG = generators.ImplicitGenerator_bipDC_1spd_catFeat_skip(opt=opt,dict=dict)
        if opt.netG == 41312:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catlabel(opt=opt,dict=dict)
        if opt.netG == 41313:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catFeature_3D(opt=opt,dict=dict)
        if opt.netG == 4132:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat(opt=opt,dict=dict)
        if opt.netG == 41321:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catlabel_skipSPD_3Dnoise(opt=opt,dict=dict)
        if opt.netG == 41322:
            self.netG = generators.ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413221:
            self.netG = generators.ImplicitGenerator_noEC_bipDEC_catFeat_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413222:
            self.netG = generators.ImplicitGenerator_bipDEC_shallow_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413227:
            self.netG = generators.ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb_simplex(opt=opt,dict=dict)
        if opt.netG == 4133:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_no_spade_iter(opt=opt,dict=dict)
        if opt.netG == 4134:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD(opt=opt,dict=dict)
        if opt.netG == 41341:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catLbFeat_skipSPD(opt=opt,dict=dict)
        if opt.netG == 4135:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_3iter33styles(opt=opt,dict=dict)
        if opt.netG == 4136:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_noDistmap(opt=opt,dict=dict)
        if opt.phase == "train":
            print('starting to construct discriminator')
            self.netD = EPE_D.PerceptualProjectionDiscEnsemble().to(self.device_models)
            print('discriminator constructed')
            self.vgg = vgg16.VGG16().to(self.device_models)
            print('pretrained vgg inited')
            self.seg_net_pretrained = pretrained_mseg_model_35_ft(device=self.device_models,pretrained_path=opt.seg_path)

            for param in self.seg_net_pretrained.model.parameters():
                param.requires_grad = False

            print('pretrained mseg inited')
            self.run_discs = [True] * len(self.netD)
            self.gan_loss = LSLoss()
            self.reg_weight = 0.03

        self.print_parameter_count()
        self.init_networks_EPE()
        #--- EMA of generator weights ---
        with torch.no_grad():
           self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
            # pass
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()

        self.coords=None
        self.latent=None

    def forward(self, image, label,label_class_dict, mode, losses_computer,
                converted=None, latent = None,z=None,edges = None,
                dict = None):
        # Branching is applied to be compatible with DataParallel
        self.coords=converted

        self.latent=latent
        # print("input latent code:",self.latent)
        if mode == "losses_G":
            loss_G = 0
            label_map = label_class_dict[:,:1,:,:]
            fake,_ = self.netG(
                            label=label,
                            label_class_dict=label_class_dict,
                            coords=converted,
                            latent=latent,
                            return_latents=False,
                            truncation=1,
                            truncation_latent=None,
                            input_is_latent=False,
                            dict = dict
                            )


            loss_G_vgg = None

            robust_labels_logits = self.seg_net_pretrained.execute(input=fake, label=label)
            loss_G_adv = losses_computer.loss(robust_labels_logits, label)
            loss_G += loss_G_adv.mean()
            realism_maps = self.netD.forward(vgg=self.vgg, img=fake, robust_labels=robust_labels_logits,
                                                fix_input=False, run_discs=self.run_discs)
            loss_G_realism = 0
            for i, rm in enumerate(realism_maps):
                loss_G_realism, _ = tee_loss(loss_G_realism, self.gan_loss.forward_gen(input=rm).mean())
            loss_G += loss_G_realism
            loss_G.backward()
            return loss_G, [loss_G_adv, loss_G_vgg],[loss_G_realism.detach()]




        if mode == "losses_D":
            with torch.no_grad():
                # fake = self.netG(label)
                rep_fake, _ = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                    dict = dict
                )
                robust_labels_logits = self.seg_net_pretrained.execute(input=rep_fake, label=label)
            robust_labels_logits_real = self.seg_net_pretrained.execute(input=image, label=label)
            loss_D_lm = None

            rec_fake = rep_fake.detach()
            rec_fake.requires_grad_()

            # loss_S = 0
            # loss_S_real = losses_computer.loss(robust_labels_logits_real, label)
            # loss_S += loss_S_real
            # loss_S = loss_S.mean()
            # loss_S.backward()

            # forward fake images
            realism_maps = self.netD.forward(vgg=self.vgg, img=rec_fake, robust_labels=robust_labels_logits,
                                                                fix_input=True, run_discs=self.run_discs)

            loss_D_fake = 0
            pred_labels = {}  # for adaptive backprop
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_fake,_ = tee_loss(loss_D_fake, self.gan_loss.forward_fake(input=rm).mean())
                pass
            del rm
            del realism_maps

            loss_D_fake.backward()

            image.requires_grad_()

            # forward real images
            realism_maps = self.netD.forward(
                            vgg=self.vgg, img=image, robust_labels=robust_labels_logits_real, robust_img=image,
                            fix_input=(self.reg_weight <= 0), run_discs=self.run_discs)

            loss_D_real = 0
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_real += self.gan_loss.forward_real(input=rm).mean()

            del rm
            del realism_maps

            # compute gradient penalty on real images
            if self.reg_weight > 0:
                loss_D_real.backward(retain_graph=True)
                reg_loss, _ = tee_loss(0, real_penalty(loss_D_real, image))
                (self.reg_weight * reg_loss).backward()
            else:
                loss_D_real.backward()

            return loss_D_fake.detach()+loss_D_real.detach(), [loss_D_fake, loss_D_real, loss_D_lm],





        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
                else:
                    fake = self.netEMA(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)

    def init_networks_EPE(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False\
                    and ("DiscEnsemble" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)


    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges




class EPE_model_unetn1_adv(nn.Module):
    def __init__(self, opt):
        super(EPE_model_unetn1_adv, self).__init__()
        self.opt = opt
        self.device_models = 'cpu' if self.opt.gpu_ids == '-1' else 'cuda'
        #--- generator and discriminator ---

        if opt.netG == 2:
            self.netG = generators.ImplicitGenerator_tanh(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 26:
            self.netG = generators.ImplicitGenerator_tanh_no_emb(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 27:
            self.netG = generators.ImplicitGenerator_tanh_skip(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 28:
            self.netG = generators.ImplicitGenerator_tanh_skip_512_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 29:
            self.netG = generators.ImplicitGenerator_Conv_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 34:
            self.netG = generators.ImplicitGenerator_multi_scale_U(opt=opt,dict=dict)
        if opt.netG == 341:
            self.netG = generators.ImplicitGenerator_multi_scale_U_no_dist(opt=opt,dict=dict)
        if opt.netG == 342:
            self.netG = generators.ImplicitGenerator_multi_scale_U_only_decod_dist(opt=opt,dict=dict)
        if opt.netG == 343:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear(opt=opt,dict=dict)
        if opt.netG == 3431:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_avepool(opt=opt,dict=dict)
        if opt.netG == 344:
            self.netG = generators.ImplicitGenerator_multi_scale_U_nearest_modulation(opt=opt,dict=dict)
        if opt.netG == 345:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv(opt=opt,dict=dict)
        if opt.netG == 3451:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transpose_avepool(opt=opt,dict=dict)
        if opt.netG == 3452:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style(opt=opt,dict=dict)
        if opt.netG == 3453:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_convdownsampling(opt=opt,dict=dict)
        if opt.netG == 3454:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_no_emb_noise_input(opt=opt,dict=dict)
        if opt.netG == 34541:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_35style_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 3455:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput(opt=opt,dict=dict)
        if opt.netG == 34552:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomodulation_noemblatentinput(opt=opt,dict=dict)

        if opt.netG == 34551:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 346:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_sle(opt=opt,dict=dict)
        if opt.netG == 347:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_styleGan(opt=opt,dict=dict)
        if opt.netG == 411:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_ganformer_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 412:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_bipartiteencoder_noembnoiseinput(opt=opt,dict=dict)
        if opt.netG == 413:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder(opt=opt,dict=dict)
        if opt.netG == 4131:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade(opt=opt,dict=dict)
        if opt.netG == 41311:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade_catFeature(opt=opt,dict=dict)
        if opt.netG == 413111:
            self.netG = generators.ImplicitGenerator_noENC_bipDEC_1spade(opt=opt,dict=dict)
        if opt.netG == 413112:
            self.netG = generators.ImplicitGenerator_bipDC_1spd_catFeat_skip(opt=opt,dict=dict)
        if opt.netG == 41312:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catlabel(opt=opt,dict=dict)
        if opt.netG == 41313:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catFeature_3D(opt=opt,dict=dict)
        if opt.netG == 4132:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat(opt=opt,dict=dict)
        if opt.netG == 41321:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catlabel_skipSPD_3Dnoise(opt=opt,dict=dict)
        if opt.netG == 41322:
            self.netG = generators.ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413221:
            self.netG = generators.ImplicitGenerator_noEC_bipDEC_catFeat_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413222:
            self.netG = generators.ImplicitGenerator_bipDEC_shallow_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 4133:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_no_spade_iter(opt=opt,dict=dict)
        if opt.netG == 4134:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD(opt=opt,dict=dict)
        if opt.netG == 41341:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catLbFeat_skipSPD(opt=opt,dict=dict)
        if opt.netG == 4135:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_3iter33styles(opt=opt,dict=dict)
        if opt.netG == 4136:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_noDistmap(opt=opt,dict=dict)
        if opt.phase == "train":
            print('starting to construct discriminator')
            self.netD = EPE_D.PerceptualProjectionDiscEnsemble().to(self.device_models)
            print('discriminator constructed')
            self.vgg = vgg16.VGG16().to(self.device_models)
            print('pretrained vgg inited')
            self.seg = discriminators.OASIS_Discriminator(opt).to(self.device_models)
            self.seg.load_state_dict(torch.load(opt.seg_path, map_location=self.device_models))

            for param in self.seg.parameters():
                param.requires_grad=True

            print('pretrained n+1 Unet seg inited')
            self.run_discs = [True] * len(self.netD)
            self.gan_loss = LSLoss()
            self.reg_weight = 0.03

        self.print_parameter_count()
        self.init_networks_EPE()
        #--- EMA of generator weights ---
        with torch.no_grad():
           self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
            # pass
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()

        self.coords=None
        self.latent=None

    def forward(self, image, label,label_class_dict, mode, losses_computer,
                converted=None, latent = None,z=None,edges = None,
                dict = None):
        # Branching is applied to be compatible with DataParallel
        self.coords=converted

        self.latent=latent
        # print("input latent code:",self.latent)
        if mode == "losses_G":
            loss_G = 0
            label_map = label_class_dict[:,:1,:,:]
            fake,_ = self.netG(
                            label=label,
                            label_class_dict=label_class_dict,
                            coords=converted,
                            latent=latent,
                            return_latents=False,
                            truncation=1,
                            truncation_latent=None,
                            input_is_latent=False,
                            dict = dict
                            )

            loss_G_vgg = None

            robust_labels_logits = self.seg(input=fake)
            loss_G_adv = losses_computer.loss(robust_labels_logits, label, for_real=True)
            loss_G += loss_G_adv.mean()
            realism_maps = self.netD.forward(vgg=self.vgg, img=fake, robust_labels=robust_labels_logits,
                                                fix_input=False, run_discs=self.run_discs)
            loss_G_realism = 0
            for i, rm in enumerate(realism_maps):
                loss_G_realism, _ = tee_loss(loss_G_realism, self.gan_loss.forward_gen(input=rm).mean())
            loss_G += loss_G_realism
            loss_G.backward()
            return loss_G, [loss_G_adv, loss_G_vgg],[loss_G_realism.detach()]




        if mode == "losses_D":
            with torch.no_grad():
                # fake = self.netG(label)
                rep_fake, _ = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                    dict = dict
                )
                robust_labels_logits = self.seg(input=rep_fake)
            robust_labels_logits_real = self.seg(input=image)
            loss_D_lm = None

            rec_fake = rep_fake.detach()
            rec_fake.requires_grad_()

            loss_seg = 0
            output_seg_fake = self.seg(rec_fake)
            loss_seg_fake = losses_computer.loss(output_seg_fake, label, for_real=False)
            loss_seg += loss_seg_fake
            output_seg_real = self.seg(image)
            loss_seg_real = losses_computer.loss(output_seg_real, label, for_real=True)
            loss_seg += loss_seg_real
            loss_seg.backward()

            # forward fake images
            realism_maps = self.netD.forward(vgg=self.vgg, img=rec_fake, robust_labels=robust_labels_logits,
                                                                fix_input=True, run_discs=self.run_discs)

            loss_D_fake = 0
            pred_labels = {}  # for adaptive backprop
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_fake,_ = tee_loss(loss_D_fake, self.gan_loss.forward_fake(input=rm).mean())
                pass
            del rm
            del realism_maps

            loss_D_fake.backward()

            image.requires_grad_()

            # forward real images
            realism_maps = self.netD.forward(
                            vgg=self.vgg, img=image, robust_labels=robust_labels_logits_real, robust_img=image,
                            fix_input=(self.reg_weight <= 0), run_discs=self.run_discs)

            loss_D_real = 0
            for i, rm in enumerate(realism_maps):
                if rm is None:
                    continue
                loss_D_real += self.gan_loss.forward_real(input=rm).mean()

            del rm
            del realism_maps

            # compute gradient penalty on real images
            if self.reg_weight > 0:
                loss_D_real.backward(retain_graph=True)
                reg_loss, _ = tee_loss(0, real_penalty(loss_D_real, image))
                (self.reg_weight * reg_loss).backward()
            else:
                loss_D_real.backward()

            return loss_D_fake.detach()+loss_D_real.detach(), [loss_D_fake, loss_D_real, loss_D_lm],





        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
                else:
                    fake = self.netEMA(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)

    def init_networks_EPE(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False\
                    and ("DiscEnsemble" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)


    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges






def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model = torch.nn.DataParallel(model)
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    instance_map = data['instance']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    ####!!原本的label是一张图像!!通过scatter_函数转换成one-shot coding!!!
    return data['image'], input_semantics, label_map, instance_map


def generate_labelmix(label, fake_image, real_image,device_models):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,)).to(device_models)
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map

def preprocess_input_with_dist(opt,data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
        data['distance'] = data['distance'].cuda()
    label_map = data['label']
    instance_map = data['instance']
    dist_map = data['distance']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    ####!!原本的label是一张图像!!通过scatter_函数转换成one-shot coding!!!
    return data['image'], input_semantics, label_map, instance_map, dist_map


class seg_losses_computer():  ## for discriminator
    def __init__(self, opt):
        self.opt = opt

    def get_class_balancing(self, opt, input, label):
        if not opt.no_balancing_inloss:
            class_occurence = torch.sum(label, dim=(0, 2, 3))
            if opt.contain_dontcare_label:
                class_occurence[0] = 0
            num_of_classes = (class_occurence > 0).sum()
            coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
            integers = torch.argmax(label, dim=1, keepdim=True)
            if opt.contain_dontcare_label:
                coefficients[0] = 0
            weight_map = coefficients[integers]
        else:
            weight_map = torch.ones_like(input[:, :, :, :])
        return weight_map

    def loss(self, input, label):
        # --- balancing classes ---
        weight_map = self.get_class_balancing(self.opt, input, label)
        # --- n loss ---
        target = label
        loss = torch.nn.functional.cross_entropy(input, target, reduction='none')
        loss = torch.mean(loss * weight_map[:, 0, :, :])

        return loss




class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        self.device_models = 'cpu' if self.opt.gpu_ids == '-1' else 'cuda'
        #--- generator and discriminator ---

        if opt.netG == 2:
            self.netG = generators.ImplicitGenerator_tanh(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 26:
            self.netG = generators.ImplicitGenerator_tanh_no_emb(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 27:
            self.netG = generators.ImplicitGenerator_tanh_skip(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 28:
            self.netG = generators.ImplicitGenerator_tanh_skip_512_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 29:
            self.netG = generators.ImplicitGenerator_Conv_U_net(opt=opt, size=(256, 512), hidden_size=512,
                                                                style_dim=512, n_mlp=8,
                                                                activation=None, channel_multiplier=2)
        if opt.netG == 34:
            self.netG = generators.ImplicitGenerator_multi_scale_U(opt=opt,dict=dict)
        if opt.netG == 341:
            self.netG = generators.ImplicitGenerator_multi_scale_U_no_dist(opt=opt,dict=dict)
        if opt.netG == 342:
            self.netG = generators.ImplicitGenerator_multi_scale_U_only_decod_dist(opt=opt,dict=dict)
        if opt.netG == 343:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear(opt=opt,dict=dict)
        if opt.netG == 3431:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_avepool(opt=opt,dict=dict)
        if opt.netG == 344:
            self.netG = generators.ImplicitGenerator_multi_scale_U_nearest_modulation(opt=opt,dict=dict)
        if opt.netG == 345:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv(opt=opt,dict=dict)
        if opt.netG == 3451:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transpose_avepool(opt=opt,dict=dict)
        if opt.netG == 3452:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style(opt=opt,dict=dict)
        if opt.netG == 3453:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_convdownsampling(opt=opt,dict=dict)
        if opt.netG == 3454:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_separate_style_no_emb_noise_input(opt=opt,dict=dict)
        if opt.netG == 34541:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_35style_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 3455:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput(opt=opt,dict=dict)
        if opt.netG == 34552:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomodulation_noemblatentinput(opt=opt,dict=dict)

        if opt.netG == 34551:
            self.netG = generators.ImplicitGenerator_multi_scale_U_transconv_nomodulation_noemb_noiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 346:
            self.netG = generators.ImplicitGenerator_multi_scale_U_bilinear_sle(opt=opt,dict=dict)
        if opt.netG == 347:
            self.netG = generators.ImplicitGenerator_multi_scale_U_trans_conv_styleGan(opt=opt,dict=dict)
        if opt.netG == 411:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_ganformer_noembnoiseinput_noisylabel(opt=opt,dict=dict)
        if opt.netG == 412:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_bipartiteencoder_noembnoiseinput(opt=opt,dict=dict)
        if opt.netG == 413:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder(opt=opt,dict=dict)
        if opt.netG == 4131:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade(opt=opt,dict=dict)
        if opt.netG == 41311:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_1spade_catFeature(opt=opt,dict=dict)
        if opt.netG == 413111:
            self.netG = generators.ImplicitGenerator_noENC_bipDEC_1spade(opt=opt,dict=dict)
        if opt.netG == 413112:
            self.netG = generators.ImplicitGenerator_bipDC_1spd_catFeat_skip(opt=opt,dict=dict)
        if opt.netG == 41312:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catlabel(opt=opt,dict=dict)
        if opt.netG == 41313:
            self.netG = generators.ImplicitGenerator_bipDEC_1spade_catFeature_3D(opt=opt,dict=dict)
        if opt.netG == 4132:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat(opt=opt,dict=dict)
        if opt.netG == 41321:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catlabel_skipSPD_3Dnoise(opt=opt,dict=dict)
        if opt.netG == 41322:
            self.netG = generators.ImplicitGenerator_bipDEC_catlabel_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413221:
            self.netG = generators.ImplicitGenerator_noEC_bipDEC_catFeat_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 413222:
            self.netG = generators.ImplicitGenerator_bipDEC_shallow_skipSPD_3Dnoise_noisylb(opt=opt,dict=dict)
        if opt.netG == 4133:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_cat_no_spade_iter(opt=opt,dict=dict)
        if opt.netG == 4134:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD(opt=opt,dict=dict)
        if opt.netG == 41341:
            self.netG = generators.ImplicitGenerator_bipartiteDEcoder_catLbFeat_skipSPD(opt=opt,dict=dict)
        if opt.netG == 4135:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_3iter33styles(opt=opt,dict=dict)
        if opt.netG == 4136:
            self.netG = generators.ImplicitGenerator_multiscaleU_transconv_nomod_bipartiteDEcoder_catFeature_skipSPD_noDistmap(opt=opt,dict=dict)
        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with torch.no_grad():
           self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
            # pass
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()
        #--- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
        self.coords=None
        self.latent=None

    def forward(self, image, label,label_class_dict, mode, losses_computer,
                converted=None, latent = None,z=None,edges = None,
                dict = None):
        # Branching is applied to be compatible with DataParallel
        self.coords=converted

        self.latent=latent
        # print("input latent code:",self.latent)
        if mode == "losses_G":
            loss_G = 0
            fake,_ = self.netG(
                            label=label,
                            label_class_dict=label_class_dict,
                            coords=converted,
                            latent=latent,
                            return_latents=False,
                            truncation=1,
                            truncation_latent=None,
                            input_is_latent=False,
                            dict = dict
                            )
            # print("fake:",fake.shape,fake)
            output_D = self.netD(fake)
            loss_G_adv = losses_computer.loss(output_D, label, for_real=True)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg],[None]

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                # fake = self.netG(label)
                fake, _ = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                    dict = dict
                )
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image)
            loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image,self.device_models)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
                                                                                output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
                else:
                    fake = self.netEMA(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator"\
                    and ("ImplicitGenerator" in type(net).__name__) == False:  ########jhl
                net.apply(init_weights)

    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges
