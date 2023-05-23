import torch
import torch.nn as nn
import torch.nn.functional as F
import models.lib_loss_contrastive.losses_non_contrast as losses
import models.models_contrastive_nomem as models
import dataloaders.dataloaders as dataloaders
import utils_contrastive.utils as utils
from utils_contrastive.fid_scores import fid_pytorch
from utils_contrastive.miou_scores import miou_pytorch
import matplotlib.backends

##############


import config_contrastive as config


# from models.noise import mixing_noise
import torch
import models.tensor_transforms as tt
import numpy as np

from models.blocks import make_dist_train_val_cityscapes_datasets as distance_map
from models.blocks import make_dist_train_val_cityscapes_datasets_multichannel
from models.blocks import mixing_noise






#--- read options ---#
opt = config.read_arguments(train=True)



# opt.Matrix_Computation = True
# opt.apply_MOD_CLADE = True
# opt.only_CLADE = True
# opt.add_dist = True
#
# opt.dataroot = '/Users/hlj/Documents/NoSync.nosync/FA/cityscapes'
# opt.gpu_ids = '-1'
# opt.netG = 512
# opt.batch_size = 2
# opt.checkpoints_dir='./checkpoints/test_cpu'


device = "cpu" if opt.gpu_ids == '-1' else 'cuda'
print("nb of gpus: ", torch.cuda.device_count())
#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloader,dataloader_supervised, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)
im_saver_all_in_one = utils.image_saver_all_in_one(opt)
fid_computer = fid_pytorch(opt, dataloader_val)
miou_computer = miou_pytorch(opt,dataloader_val)

#--- create models ---#
# model = models.OASIS_model(opt)
# model = models.put_on_multi_gpus(model, opt)






model = models.OASIS_model(opt)

# model.half()

model = models.put_on_multi_gpus(model, opt)

# model = Generator(size=256, hidden_size=512, style_dim=512, n_mlp=8,
#                       activation=None, channel_multiplier=2,
#                       ).to(device)



# loss_G, losses_G_list = model(image, label,"losses_G", losses_computer,converted=converted,latent=noise)


#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))##????????
# optimizerG = torch.optim.Adam(model.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))##????????



# checkpoint = torch.load('./checkpoints_MOD+CLADE/oasis_cityscapes/')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.eval()
# # - or -
# model.train()


def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item



# height=256
# width =512
# in_channel = 1024
# out_channel = 512
# batch = 1
# scale = 1

# input=torch.randn(batch,in_channel,height,width).cuda(0)
# weight= torch.randn(batch,out_channel,in_channel,height,width).cuda(0)
# output= torch.einsum('bihw,boihw->bohw',input,weight).cuda(0)
# print(input.size(),weight.size(),output.size())








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
        image, label, label_map, instance_map = models.preprocess_input(opt, data_i)

        # label_class_dict = torch.sum((label*label_class_extractor),dim=1,keepdim=True)##don't use this

        # label_class_dict = torch.argmax(label, 1)##output original labelmap  # [n, h, w]
        # equal = torch.sum(label_map - label_class_dict, dim=(0, 1, 2, 3))

        #label_class_dict = label_map.squeeze(1)##just use original labelmap directly

        dist_map = distance_map(label_map.squeeze(1).to('cpu'), norm='norm').to(device)
        # dist_map = make_dist_train_val_cityscapes_datasets_multichannel(label.to('cpu'), norm='norm').to(device)
        dist_map_16 = distance_map(F.interpolate(label_map.float(), (16, 32), mode='nearest').squeeze(1).to('cpu'),
                                   norm='norm').to(device)
        dist_map_32 = distance_map(F.interpolate(label_map.float(), (32, 64), mode='nearest').squeeze(1).to('cpu'),
                                   norm='norm').to(device)
        dist_map_64 = distance_map(F.interpolate(label_map.float(), (64, 128), mode='nearest').squeeze(1).to('cpu'),
                                   norm='norm').to(device)
        dist_map_128 = distance_map(F.interpolate(label_map.float(), (128, 256), mode='nearest').squeeze(1).to('cpu'),
                                    norm='norm').to(device)
        dist_16_to_128 = {
            16: dist_map_16,
            32: dist_map_32,
            64: dist_map_64,
            128: dist_map_128,
        }
        dict = {
            'dist_16_to_128': dist_16_to_128
        }


        label_class_dict = torch.cat((label_map,dist_map),dim=1)



        # --- generator update ---#

        noise = mixing_noise(batch_size, 512, 0, device)
        coords = tt.convert_to_coord_format(batch_size, 256, 512,device=device, integer_values=False)
        # input_img = torch.randn([batch_size, 3, 256, 512])
        input_img = image
        real_stack = torch.cat([input_img, coords], 1)
        real_img, converted = real_stack[:, :3], real_stack[:, 3:]


        # image.half()
        # label.half()
        # label_class_dict.half()
        # converted.half()
        # noise[0].half()


        model.module.netG.zero_grad()
        # model.netG.zero_grad()
        # loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)##????????
        loss_G, losses_G_list, loss_contra_list = model(image=image,
                                      label= label,
                                      label_class_dict=label_class_dict,
                                      mode= "losses_G",
                                      losses_computer= losses_computer,
                                      converted=converted,
                                      latent=noise,
                                      dict=dict)
        loss_G.backward()
        optimizerG.step()



        # --- discriminator update ---#
        model.module.netD.zero_grad()##????????
        # model.netD.zero_grad()
        # loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = model(image=image,
                                      label= label,
                                      label_class_dict=label_class_dict,
                                      mode= "losses_D",
                                      losses_computer= losses_computer,
                                      converted=converted,
                                      latent=noise,
                                      dict=dict)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()

        #--- stats update ---#
        if not opt.no_EMA:
            # utils.update_EMA(model, cur_iter, dataloader_supervised, opt)
            utils.update_EMA(model, cur_iter, dataloader_supervised, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model,
                                     image,
                                     label,
                                     cur_iter,
                                     label_class_dict=label_class_dict,
                                     converted=converted,
                                     latent=noise,
                                     dict = dict)
            im_saver_all_in_one.visualize_batch(model,
                                                image,
                                                label,
                                                cur_iter,
                                                label_class_dict=label_class_dict,
                                                converted=converted,
                                                latent=noise,
                                                dict=dict)
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
        visualizer_losses(cur_iter, losses_G_list+losses_D_list+loss_contra_list)

#--- after training ---#
utils.update_EMA(model, cur_iter, dataloader_supervised, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")


def _dequeue_and_enqueue(self, keys, labels,
                         segment_queue, segment_queue_ptr,
                         pixel_queue, pixel_queue_ptr):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]

    labels = labels[:, ::self.network_stride, ::self.network_stride]

    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero()

            # segment enqueue and dequeue
            feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
            ptr = int(segment_queue_ptr[lb])
            segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
            segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % self.memory_size

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, self.pixel_update_freq)
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)
            ptr = int(pixel_queue_ptr[lb])

            if ptr + K >= self.memory_size:
                pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                pixel_queue_ptr[lb] = 0
            else:
                pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + K) % self.memory_size



