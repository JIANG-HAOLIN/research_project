from EPEdiscriminator import discriminator_losses as D_loss
from EPEdiscriminator import  epe_discriminators as EPE_D
# from EPEdiscriminator import network_factory as NF
from EPEdiscriminator import  vgg16
from utils.drn_segment import DRNSeg
from torch import autograd
from pretrained_models.msegsemantic.mseg_semantic.tool.universal_demo_batched import mseg_compute_predicted,pretrained_mseg_model
from EPEdiscriminator.compute_result import compute_result_24
import torch
import os
import cv2
import numpy as np
from EPEdiscriminator.discriminator_losses import LSLoss




device = 'cpu'



def imread_rgb(img_fpath: str) -> np.ndarray:
    """
    Returns:
        RGB 3 channel nd-array with shape (H, W, 3)
    """
    bgr_img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img)
    return rgb_img


dir_path = os.path.abspath(os.path.join(__file__, "..", ".."))
img_path = os.path.join(dir_path, 'pretrained_models/msegsemantic/content/city2-480-360SS.png')
label_path = os.path.join(dir_path, 'pretrained_models/msegsemantic/content/city2label.png')



rgb_img = imread_rgb(img_path)


label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # # GRAY 1 channel ndarray with shape H * W
# label = cv2.resize(label,[1024,512],interpolation=cv2.INTER_LINEAR)
label = label.astype(np.int64)
# labelmap,label = mapping_labelmap(label)

original_image = rgb_img
input = torch.from_numpy(rgb_img).unsqueeze(0).permute(0, 3, 1, 2)/255
input = input.expand(2, 3, input.shape[2], input.shape[3])




# def tee_loss(x, y):
#     return x + y, y.detach()
#
#
# def accuracy(pred):
# 	return (pred > 0.5).float().mean()
#
# def real_penalty(loss, real_img):
# 	''' Compute penalty on real images. '''
# 	b = real_img.shape[0]
# 	grad_out = autograd.grad(outputs=loss, inputs=[real_img], create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
# 	logger.debug(f'real_penalty: g:{grad_out[0].shape}')
# 	reg_loss = torch.cat([g.pow(2).reshape(b, -1).sum(dim=1, keepdim=True) for g in grad_out if g is not None], 1).mean()
# 	return reg_loss
#
# def _run_generator(self, batch_fake, batch_real, batch_id):
#     rec_fake = self.network.generator(batch_fake)
#
#     realism_maps = self.network.discriminator.forward(
#         vgg=self.vgg, img=rec_fake, robust_labels=batch_fake.robust_labels,
#         fix_input=False, run_discs=True)
#
#     loss = 0
#     log_info = {}
#     for i, rm in enumerate(realism_maps):
#         loss, log_info[f'gs{i}'] = tee_loss(loss, self.gan_loss.forward_gen(rm[0, :, :, :].unsqueeze(0)).mean())
#         pass
#
#     loss, log_info['vgg'] = tee_loss(loss, self.vgg_weight * self.vgg_loss.forward_fake(batch_fake.img, rec_fake)[0])
#     loss.backward()
#
#     return log_info, \
#            {'rec_fake': rec_fake.detach(), 'fake': batch_fake.img.detach(), 'real': batch_real.img.detach()}
#
#
# def _forward_generator_fake(self, batch_fake):
#     """ Run the generator without any loss computation. """
#
#     rec_fake = self.network.generator(batch_fake)
#     return {'rec_fake': rec_fake.detach(), 'fake': batch_fake.img.detach()}
#
#
# def _run_discriminator(self, batch_fake, batch_real, batch_id: int):
#     log_scalar = {}
#     log_img = {}
#
#     # sample probability of running certain discriminator
#     if self.adaptive_backprop is not None:
#         run_discs = self.adaptive_backprop.sample()
#     else:
#         run_discs = [True] * len(self.network.discriminator)
#         pass
#
#     if not any(run_discs):
#         return log_scalar, log_img
#
#     with torch.no_grad():
#         rep_fake = self.network.generator(batch_fake)
#         pass
#
#     log_img['fake'] = batch_fake.img.detach()
#     log_img['rec_fake'] = rep_fake.detach()
#
#     rec_fake = rep_fake.detach()
#     rec_fake.requires_grad_()
#
#     # forward fake images
#     realism_maps = self.network.discriminator.forward( \
#         vgg=self.vgg, img=rec_fake, robust_labels=batch_fake.robust_labels,
#         fix_input=True, run_discs=run_discs)
#
#     loss = 0
#     pred_labels = {}  # for adaptive backprop
#     for i, rm in enumerate(realism_maps):
#         if rm is None:
#             continue
#
#         if self._log.isEnabledFor(logging.DEBUG):
#             log_img[f'realism_fake_{i}'] = rm.detach()
#             pass
#
#         # for getting probability of back
#         if self.check_fake_for_backprop:
#             pred_labels[i] = [(rm.detach() < 0.5).float().reshape(1, -1)]
#             pass
#         log_scalar[f'rdf{i}'] = accuracy(rm.detach())  # percentage of fake predicted as real
#         loss, log_scalar[f'ds{i}'] = tee_loss(loss, self.gan_loss.forward_fake(rm).mean())
#         pass
#     del rm
#     del realism_maps
#
#     loss.backward()
#
#     log_img['real'] = batch_real.img.detach()
#     batch_real.img.requires_grad_()
#
#     # forward real images
#     realism_maps = self.network.discriminator.forward(
#         vgg=self.vgg, img=batch_real.img, robust_labels=batch_real.robust_labels, robust_img=batch_real.img,
#         fix_input=(self.reg_weight <= 0), run_discs=run_discs)
#
#     loss = 0
#     for i, rm in enumerate(realism_maps):
#         if rm is None:
#             continue
#
#         if self._log.isEnabledFor(logging.DEBUG):
#             log_img[f'realism_real_{i}'] = rm.detach()
#             pass
#
#         if i in pred_labels:
#             # predicted correctly, here real as real
#             pred_labels[i].append((rm.detach() > 0.5).float().reshape(1, -1))
#         else:
#             pred_labels[i] = [(rm.detach() > 0.5).float().reshape(1, -1)]
#             pass
#
#         log_scalar[f'rdr{i}'] = accuracy(rm.detach())  # percentage of real predicted as real
#         loss += self.gan_loss.forward_real(rm).mean()
#         pass
#     del rm
#     del realism_maps
#
#     # compute gradient penalty on real images
#     if self.reg_weight > 0:
#         loss.backward(retain_graph=True)
#         self._log.debug(f'Computing penalty on real: {loss} from i:{batch_real.img.shape}.')
#         reg_loss, log_scalar['reg'] = tee_loss(0, real_penalty(loss, batch_real.img))
#         (self.reg_weight * reg_loss).backward()
#     else:
#         loss.backward()
#         pass
#     pass
#
#     # update disc probabilities
#     if self.adaptive_backprop is not None:
#         self.adaptive_backprop.update(pred_labels)
#         pass
#
#     return log_scalar, log_img
discriminator = EPE_D.PerceptualProjectionDiscEnsemble()
vgg = vgg16.VGG16().to(device)
rec_fake = input
run_discs = [True] * len(discriminator)
# seg_net = DRNSeg('drn_d_105',19,pretrained=False)
# seg_net.load_state_dict(torch.load('../pretrained_models/drn-d-105_ms_cityscapes.pth'))
# robust_label = seg_net(rec_fake)

pretrained_mseg_model = pretrained_mseg_model(device='cpu')
robust_labels = pretrained_mseg_model.execute(input=input,label=label)
# robust_labels = torch.randn([1,19,256,512])
compute_result_24(img_path, robust_labels, label, original_image, min_resolution=1080)
realism_maps = discriminator.forward(vgg=vgg, img=rec_fake, robust_labels=robust_labels,
                                        fix_input=True, run_discs=run_discs)


log_scalar = {}
log_img = {}

# # sample probability of running certain discriminator
# if self.adaptive_backprop is not None:
#     run_discs = self.adaptive_backprop.sample()
# else:
#     run_discs = [True] * len(self.network.discriminator)
#     pass


# with torch.no_grad():
#     rep_fake = self.Generator(batch_fake)
#     pass
#
# rec_fake = rep_fake.detach()
# rec_fake.requires_grad_()
#
# # forward fake images
# realism_maps = self.network.discriminator.forward( \
#     vgg=self.vgg, img=rec_fake, robust_labels=batch_fake.robust_labels,
#     fix_input=True, run_discs=run_discs)
#
# loss = 0
# pred_labels = {}  # for adaptive backprop
# for i, rm in enumerate(realism_maps):
#     if rm is None:
#         continue
#
#     if self._log.isEnabledFor(logging.DEBUG):
#         log_img[f'realism_fake_{i}'] = rm.detach()
#         pass
#
#     # for getting probability of back
#     if self.check_fake_for_backprop:
#         pred_labels[i] = [(rm.detach() < 0.5).float().reshape(1, -1)]
#         pass
#     log_scalar[f'rdf{i}'] = accuracy(rm.detach())  # percentage of fake predicted as real
#     loss, log_scalar[f'ds{i}'] = tee_loss(loss, self.gan_loss.forward_fake(rm).mean())
#     pass
# del rm
# del realism_maps
#
# loss.backward()
#
# log_img['real'] = batch_real.img.detach()
# batch_real.img.requires_grad_()
#
# # forward real images
# realism_maps = self.network.discriminator.forward( \
#     vgg=self.vgg, img=batch_real.img, robust_labels=batch_real.robust_labels, robust_img=batch_real.img,
#     fix_input=(self.reg_weight <= 0), run_discs=run_discs)
#
# loss = 0
# for i, rm in enumerate(realism_maps):
#     if rm is None:
#         continue
#
#     if self._log.isEnabledFor(logging.DEBUG):
#         log_img[f'realism_real_{i}'] = rm.detach()
#         pass
#
#     if i in pred_labels:
#         # predicted correctly, here real as real
#         pred_labels[i].append((rm.detach() > 0.5).float().reshape(1, -1))
#     else:
#         pred_labels[i] = [(rm.detach() > 0.5).float().reshape(1, -1)]
#         pass
#
#     log_scalar[f'rdr{i}'] = accuracy(rm.detach())  # percentage of real predicted as real
#     loss += self.gan_loss.forward_real(rm).mean()
#     pass
# del rm
# del realism_maps
#
# # compute gradient penalty on real images
# if self.reg_weight > 0:
#     loss.backward(retain_graph=True)
#     self._log.debug(f'Computing penalty on real: {loss} from i:{batch_real.img.shape}.')
#     reg_loss, log_scalar['reg'] = tee_loss(0, real_penalty(loss, batch_real.img))
#     (self.reg_weight * reg_loss).backward()
# else:
#     loss.backward()
#     pass
# pass
#
# # update disc probabilities
# if self.adaptive_backprop is not None:
#     self.adaptive_backprop.update(pred_labels)
#     pass
#
# return log_scalar, log_img


print("done")