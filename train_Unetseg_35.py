from models.sync_batchnorm import DataParallelWithCallback
import copy
from torch.nn import init

import models.norms as norms

from models.discriminator import residual_block_D
from torch import nn, autograd, optim

from pathlib import Path
import matplotlib.pyplot as plt
from utils import utils

from torch.autograd import Variable

from collections import namedtuple


import config

import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
from tqdm import tqdm





#--- read options ---#
opt = config.read_arguments(train=True)




miou_scores_device = "cpu" if opt.gpu_ids == '-1' else 'cuda'

# --------------------------------------------------------------------------#
# This code is an adapted version of https://github.com/mseitzer/pytorch-fid
# --------------------------------------------------------------------------#


def update_EMA(model, cur_iter, dataloader, opt, force_run_stats=False):
    # update weights based on new generator weights
    with torch.no_grad():
        for key in model.module.netEMA.state_dict():
            model.module.netEMA.state_dict()[key].data.copy_(
                model.module.netEMA.state_dict()[key].data * opt.EMA_decay +
                model.module.netD.state_dict()[key].data   * (1 - opt.EMA_decay)
            )
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0 or
                           cur_iter % opt.freq_save_ckpt == 0 or
                           cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        with torch.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                image, label, label_map, instance_map  = preprocess_input(opt, data_i)


                pred = model.module.netEMA(image)
                num_upd += 1
                if num_upd > 50:
                    break


def save_networks(opt, cur_iter, model, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        torch.save(model.module.netD.state_dict(), path+'/%s_D.pth' % ("latest"))
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%s_EMA.pth' % ("latest"))
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        torch.save(model.module.netD.state_dict(), path+'/%s_D.pth' % ("best"))
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%s_EMA.pth' % ("best"))
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        torch.save(model.module.netD.state_dict(), path+'/%d_D.pth' % (cur_iter))
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%d_EMA.pth' % (cur_iter))


class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics,for_supervision = False):

        opt.load_size =  512 if for_metrics else 512
        opt.crop_size =  512 if for_metrics else 512
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0


        self.opt = opt
        self.for_metrics = for_metrics
        self.for_supervision = False
        self.images, self.labels, self.paths = self.list_images()

        if opt.mixed_images and not for_metrics :
            self.mixed_index=np.random.permutation(len(self))
        else :
            self.mixed_index=np.arange(len(self))

        if for_supervision :

            if opt.model_supervision == 0 :
                return
            elif opt.model_supervision == 1 :
                self.supervised_indecies = np.array(np.random.choice(len(self),opt.supervised_num),dtype=int)
            elif opt.model_supervision == 2 :
                self.supervised_indecies = np.arange(len(self),dtype = int)
            images = []
            labels = []

            for index in self.supervised_indecies :
                images.append(self.images[index])
                labels.append(self.labels[index])

            self.images = images
            self.labels = labels

            self.mixed_index = np.arange(len(self))

            classes_counts = np.zeros((34),dtype=int)
            supervised_classes_in_images = []
            counts_in_images = []
            self.weights = []
            self.for_supervision = False
            self.to_tensor = TR.ToTensor()

        self.normalize = TR.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])


    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[self.mixed_index[idx]])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        instance = Image.open(os.path.join(self.paths[1], self.labels[idx].replace('labelIds','instanceIds')))
        # label = np.array(label).max()
        image, label, instance = self.transforms(image, label,instance)
        label = label * 255##??
        if self.for_supervision :
            return {"image": image, "label": label,'instance':instance, "name": self.images[self.mixed_index[idx]],"weight" :self.weights[self.mixed_index[idx]]}
        else :
            return {"image": image, "label": label,'instance':instance, "name": self.images[self.mixed_index[idx]]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        self.mode  = mode
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        path_lab = os.path.join(self.opt.dataroot, "gtFine", mode)
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("labelIds") != -1:
                    labels.append(os.path.join(city_folder, item))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace("_gtFine_labelIds.png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label, instance):
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        instance = TR.functional.resize(instance, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
                instance = TR.functional.hflip(instance)

        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        instance = TR.functional.to_tensor(instance)
        # normalize
        if self.mode == "val":
            image = self.normalize(image)
        else:image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label, instance




def get_dataloaders(opt):
    dataset_name   = "CityscapesDataset"

    dataset_train = CityscapesDataset(opt, for_metrics=False)
    dataset_supervised = CityscapesDataset(opt,for_metrics = False ,for_supervision = True)
    dataset_val   = CityscapesDataset(opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = 1, shuffle = False, drop_last=False)
    dataloader_supervised = torch.utils.data.DataLoader(dataset_supervised, batch_size = opt.batch_size, shuffle = True, drop_last=True)
    return dataloader_train,dataloader_supervised, dataloader_val


def test(opt,eval_data_loader, model, num_classes, has_gt=True):
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    for iter, data_i in enumerate(eval_data_loader):

        image, label_map, instance_map = preprocess_input_19(opt, data_i)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()[0]
        # a = np.unique(pred)
        if has_gt:
            label = label_map.cpu().numpy()[0, 0, :, :]
            # b = np.unique(label)
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            '''            print('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)),end="\r")'''
        print('Eval: [{0}/{1}]'.format(iter, len(eval_data_loader)), end="\r")
    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        print(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)











class miou_pytorch():
    def __init__(self, opt, dataloader_val):
        self.opt = opt
        self.val_dataloader = dataloader_val
        self.best_miou = 0
        self.path_to_save = os.path.join(self.opt.checkpoints_dir, self.opt.name, "MIOU")
        Path(self.path_to_save).mkdir(parents=True, exist_ok=True)

    def compute_miou(self, netEMA, model, current_iter = 'latest'):
        netEMA.eval()
        with torch.no_grad():
            answer = test(opt,self.val_dataloader, netEMA, 35)

        model.module.netD.train()
        netEMA.train()
        return answer


    def update(self, model, cur_iter):
        print("--- Iter %s: computing MIOU ---" % (cur_iter))
        cur_miou = self.compute_miou( model.module.netEMA,model,cur_iter)
        self.update_logs(cur_miou, cur_iter)
        print("--- MIOU at Iter %s: " % cur_iter, "{:.2f}".format(cur_miou))
        if cur_miou > self.best_miou:
            self.best_miou = cur_miou
            is_best = True
        else:
            is_best = False
        return is_best

    def update_logs(self, cur_miou, epoch):
        try :
            np_file = np.load(self.path_to_save + "/miou_log.npy")
            first = list(np_file[0, :])
            sercon = list(np_file[1, :])
            first.append(epoch)
            sercon.append(cur_miou)
            np_file = [first, sercon]
        except:
            np_file = [[epoch], [cur_miou]]

        np.save(self.path_to_save + "/miou_log.npy", np_file)

        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.title(f'best miou:{self.best_miou}')

        plt.savefig(self.path_to_save + "/plot_miou", dpi=600)
        plt.close()


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()












class losses_computer():## for discriminator
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
        #--- balancing classes ---
        weight_map = self.get_class_balancing(self.opt, input, label)
        #--- n loss ---
        target = label
        loss = torch.nn.functional.cross_entropy(input, target, reduction='none')
        loss = torch.mean(loss * weight_map[:, 0, :, :])

        return loss





class OASIS_Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        output_channel = opt.semantic_nc
        self.channels = [3, 128, 128, 256, 256, 512, 512]
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(opt.num_res_blocks):
            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
        # decoder part
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
        for i in range(1, opt.num_res_blocks-1):
            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], opt, 1))
        self.body_up.append(residual_block_D(2*self.channels[1], 64, opt, 1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, input):
        x = input
        #encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        #decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](torch.cat((encoder_res[-i-1], x), dim=1))
        ans = self.layer_up_last(x)
        return ans


class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        self.device_models = 'cpu' if self.opt.gpu_ids == '-1' else 'cuda'
        #--- generator and discriminator ---
        if opt.phase == "train":
            self.netD = OASIS_Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with torch.no_grad():
           self.netEMA = copy.deepcopy(self.netD) if not opt.no_EMA else None
            # pass
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()
        #--- perceptual loss ---#

    def forward(self, image, label,label_class_dict, mode, losses_computer,
                converted=None, latent = None,z=None,edges = None,
                dict = None):
        # Branching is applied to be compatible with DataParallel

        # print("input latent code:",self.latent)
        if mode == "losses_G":
            return None, [None, None]

        if mode == "losses_D":
            loss_D = 0
            output_D_real = self.netD(image)
            loss_D_real = losses_computer.loss(output_D_real, label)
            loss_D += loss_D_real
            return loss_D, [None, loss_D_real, None]

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        networks = [self.netD]
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
            networks = [self.netD]
        else:
            networks = [self.netD]
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


def preprocess_input_19(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    instance_map = data['instance']
    bs, _, h, w = label_map.size()
    ####!!原本的label是一张图像!!通过scatter_函数转换成one-shot coding!!!
    return data['image'], label_map, instance_map





class losses_saver():
    def __init__(self, opt):
        if opt.model_supervision == 2:
            self.name_list = ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"]
        elif opt.model_supervision ==1:
            self.name_list = ["sup_G_Du",
                              "sup_G_D",
                              "sup_VGG",
                              "sup_G_feat_match",
                              "sup_D_fake",
                              "sup_D_real",
                              "sup_D_LM",
                              "sup_Du_fake",
                              "sup_Du_real",
                              "un_G_D",
                              "un_VGG",
                              "un_G_Du",
                              "un_edge",
                              "un_Du_fake",
                              "un_Du_real",
                              "un_Du_regularize",
                              "sup_Du_regularize"]
        else:
            self.name_list = ["Generator", "Vgg", "GAN","edge",'featMatch',"D_fake", "D_real", "LabelMix","Du_fake","Du_real","Du_regularize"]
        self.opt = opt
        self.freq_smooth_loss = 100
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        print(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path+"/losses.npy", allow_pickle = True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss-1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i]/self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss-1:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig,ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve])))*self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)),  dpi=600)
            plt.close(fig)

        fig,ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)





if __name__ =="__main__":

    # opt.Matrix_Computation = True
    # opt.apply_MOD_CLADE = True
    # opt.only_CLADE = True
    # opt.add_dist = True
    #
    # opt.netG = 413222
    # opt.batch_size = 2
    # #
    # opt.gpu_ids = '-1'
    # opt.checkpoints_dir='./checkpoints/test_cpu'
    # opt.dataroot = '/Users/hlj/Documents/NoSync.nosync/FA/cityscapes'


    # opt.gpu_ids = '0'
    # opt.dataroot='/data/public/cityscapes'
    # opt.checkpoints_dir='./checkpoints/test_interactive'


    device = "cpu" if opt.gpu_ids == '-1' else 'cuda'
    print("nb of gpus: ", torch.cuda.device_count())
    #--- create utils ---#
    timer = utils.timer(opt)
    visualizer_losses = losses_saver(opt)
    losses_computer = losses_computer(opt)
    dataloader,dataloader_supervised, dataloader_val = get_dataloaders(opt)
    miou_computer = miou_pytorch(opt,dataloader_val)

    #--- create models ---#
    model = OASIS_model(opt)
    model = put_on_multi_gpus(model, opt)



    #--- create optimizers ---#
    optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=0.005, betas=(opt.beta1, opt.beta2))##????????



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
            image, label, label_map, instance_map = preprocess_input(opt, data_i)

            # --- discriminator update ---#
            model.module.netD.zero_grad()
            loss_D, losses_D_list = model(image=image,
                                          label= label,
                                          label_class_dict=None,
                                          mode= "losses_D",
                                          losses_computer= losses_computer,
                                          converted=None,
                                          latent=None,
                                          dict=None)
            loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
            loss_D.backward()
            optimizerD.step()

            #--- stats update ---#
            if not opt.no_EMA:
                update_EMA(model, cur_iter, dataloader_supervised, opt)
            if cur_iter % 100 == 0:
                timer(epoch, cur_iter)
            if cur_iter % opt.freq_save_latest == 0:
                save_networks(opt, cur_iter, model, latest=True)
            if cur_iter % 600 == 0 and cur_iter > 0:
                is_best = miou_computer.update(model, cur_iter)
                if is_best:
                    save_networks(opt, cur_iter, model, best=True)
            losses_G_list = [None,None]
            visualizer_losses(cur_iter, losses_G_list+losses_D_list)

    #--- after training ---#
    if not opt.no_EMA:
        update_EMA(model, cur_iter, dataloader_supervised, opt, force_run_stats=True)
    save_networks(opt, cur_iter, model)
    save_networks(opt, cur_iter, model, latest=True)
    is_best = miou_computer.update(model, cur_iter)
    if is_best:
        utils.save_networks(opt, cur_iter, model, best=True)

    print("The training has successfully finished")
