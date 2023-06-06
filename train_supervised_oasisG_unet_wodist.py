import torch
import torch.nn.functional as F
import models.losses as losses
from models.models_EPE import seg_losses_computer
import models.models_EPE as models_EPE
import dataloaders.dataloaders as dataloaders
import utils.utils_EPE as utils
from utils.fid_scores import fid_pytorch_vanillaG
from utils.miou_scores import miou_pytorch_vanillaG
import matplotlib.backends

##############


import config

import os
# from models.noise import mixing_noise
import torch
import models.tensor_transforms as tt
import numpy as np

from models.blocks import make_dist_train_val_cityscapes_datasets as distance_map
from models.blocks import make_dist_train_val_cityscapes_datasets_multichannel
from models.blocks import mixing_noise






#--- read options ---#
opt = config.read_arguments(train=True)

#
# opt.dataroot = '/Users/hlj/Documents/NoSync.nosync/FA/cityscapes'
# opt.gpu_ids = '-1'
# opt.netG = 1
# opt.batch_size = 2
# opt.checkpoints_dir='./checkpoints/test_cpu'
# opt.netD = 'EPE_origin'
# opt.name = 'oasis_cityscapes'


device = "cpu" if opt.gpu_ids == '-1' else 'cuda'
print("nb of gpus: ", torch.cuda.device_count())
#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = seg_losses_computer(opt)
dataloader,dataloader_supervised, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.vanillaG_image_saver(opt)
im_saver_all_in_one = utils.vanillaG_image_saver_all_in_one(opt)
fid_computer = fid_pytorch_vanillaG(opt, dataloader_val)
miou_computer = miou_pytorch_vanillaG(opt,dataloader_val)

#--- create models ---#
opt.seg_path = os.path.abspath(os.path.join(__file__, "..", "pretrained_models", "best_EMA_Unet.pth"))
model = models_EPE.EPE_model_unet_ft_vanillaG(opt)
model = models_EPE.put_on_multi_gpus(model, opt)



#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
optimizerS = torch.optim.Adam(model.module.seg.parameters(), lr=0.0001   , betas=(opt.beta1, opt.beta2))


def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item



#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader_supervised))
if opt.model_supervision != 0 :
    supervised_iter = loopy_iter(dataloader_supervised)


batch_size = opt.batch_size#@jhl
# label_class_extractor= torch.arange(1,36,1).view(1,35,1,1).cuda(0)

for epoch in range(start_epoch, opt.num_epochs):
    print('epoch %d' %epoch)
    for i, data_i in enumerate(dataloader_supervised):
        # torch.cuda.empty_cache()
        print('batch %d' %i)
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader_supervised) + i
        image, label, label_map, instance_map = models_EPE.preprocess_input(opt, data_i)





        # --- generator update ---#

        noise = mixing_noise(batch_size, 512, 0, device)
        coords = tt.convert_to_coord_format(batch_size, 256, 512,device=device, integer_values=False)
        # input_img = torch.randn([batch_size, 3, 256, 512])
        input_img = image
        real_stack = torch.cat([input_img, coords], 1)
        real_img, converted = real_stack[:, :3], real_stack[:, 3:]
        model.module.netG.zero_grad()

        loss_G, losses_G_list, loss_G_realism = model(image=image,
                                                      label= label,
                                                      label_class_dict=None,
                                                      mode= "losses_G",
                                                      losses_computer= losses_computer,
                                                      converted=converted,
                                                      latent=noise,
                                                      dict=None)
        optimizerG.step()



        # --- discriminator update ---#
        model.module.netD.zero_grad()##????????
        # model.netD.zero_grad()
        # loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = model(image=image,
                                      label= label,
                                      label_class_dict=None,
                                      mode= "losses_D",
                                      losses_computer= losses_computer,
                                      converted=converted,
                                      latent=noise,
                                      dict=None)
        optimizerD.step()
        optimizerS.step()

        #--- stats update ---#
        if not opt.no_EMA:
            # utils.update_EMA(model, cur_iter, dataloader_supervised, opt)
            utils.update_EMA(model, cur_iter, dataloader_supervised, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model,
                                     image,
                                     label,
                                     cur_iter,
                                     label_class_dict=None,
                                     converted=converted,
                                     latent=noise,
                                     dict = None)
            im_saver_all_in_one.visualize_batch(model,
                                                image,
                                                label,
                                                cur_iter,
                                                label_class_dict=None,
                                                converted=converted,
                                                latent=noise,
                                                dict=None)
            timer(epoch, cur_iter)
        #if cur_iter % opt.freq_save_ckpt == 0:
        #    utils.save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
            _ = miou_computer.update(model,cur_iter)
        visualizer_losses(cur_iter, losses_G_list+losses_D_list+loss_G_realism)

#--- after training ---#
utils.update_EMA(model, cur_iter, dataloader_supervised, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")
