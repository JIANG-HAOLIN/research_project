from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.lib_loss_contrastive.loss_helper import FSAuxCELoss, FSRMILoss, FSCELoss, FSCELOVASZLoss,FSAuxRMILoss
# from lib.utils.tools.logger import Logger as Log


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelContrastLoss, self).__init__()

        # self.configer = configer
        # self.temperature = self.configer.get('contrast', 'temperature')
        self.temperature = 0.1

        # self.base_temperature = self.configer.get('contrast', 'base_temperature')
        self.base_temperature = 0.07


        self.ignore_label = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        # self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_samples = 2048

        # self.max_views = self.configer.get('contrast', 'max_views')
        self.max_views = 100
        # self.label_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]




    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float()
        y_ = torch.zeros((class_num * cache_size, 1)).float()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, seg_q, pix_q):
        batch_loss = 0
        for i in range(self.batch_size):
            seg_x = seg_q['feature'][i]
            seg_y = seg_q['label'][i]
            pix_x = pix_q['feature'][i]
            pix_y = pix_q['label'][i]
            ### X_anchor : [batch_size, spatial_size, feature_size]
            # anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]


            mask = torch.eq(pix_y, seg_y.T).float()



        ###################################### outer product, same as attention !! ################################
            anchor_dot_contrast = torch.div(torch.matmul(pix_x, seg_x.T),
                                            self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            neg_mask = 1 - mask


            neg_logits = torch.exp(logits) * neg_mask
            neg_logits = neg_logits.sum(1, keepdim=True)

            exp_logits = torch.exp(logits)

            log_prob = logits - torch.log(exp_logits + neg_logits)

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            batch_loss += loss.mean()

        return batch_loss/self.batch_size

    def forward(self, seg_q = None, pix_q = None,batch_size=None,feature_dim=None):
        # labels = labels.unsqueeze(1).float().clone()
        # labels = torch.nn.functional.interpolate(labels,
        #                                          (feats.shape[2], feats.shape[3]), mode='nearest')
        # labels = labels.squeeze(1).long()
        # assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
        #
        # batch_size = feats.shape[0]
        #
        # labels = labels.contiguous().view(batch_size, -1)
        # feats = feats.permute(0, 2, 3, 1)
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        #
        # feats_, labels_ = self._hard_anchor_sampling(feats, labels)
        self.batch_size = batch_size
        self.feature_dim = feature_dim

        loss = self._contrastive(seg_q, pix_q)
        return loss


class ContrastCELoss_nomem(nn.Module, ABC):
    def __init__(self, configer=None,device=None):
        super(ContrastCELoss_nomem, self).__init__()

        self.configer = configer
        self.device = device
        ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        # Log.info('ignore_index: {}'.format(ignore_index))

        # self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.loss_weight = 1
        # self.use_rmi = self.configer.get('contrast', 'use_rmi')
        # self.use_lovasz = self.configer.get('contrast', 'use_lovasz')

        # if self.use_rmi:
        #     self.seg_criterion = FSRMILoss(configer=configer)
        # elif self.use_lovasz:
        #     self.seg_criterion = FSCELOVASZLoss(configer=configer)
        # else:
        #     self.seg_criterion = FSCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)


        # self.memory_size = 5000
        self.feature_dim = 32
        self.network_stride = 1
        self.num_class_sample = 50
        # "num_classes": 19
        self.important_class = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.label_list = np.linspace(start = 0,stop = 35,num=36)
        self.num_classes = len(self.label_list)


    def compute_class_prototype(self, emb, labels):

        seg_Q = {'feature':[],'label':[]}

        pix_Q = {'feature':[],'label':[]}

        batch_size = emb.shape[0]
        feat_dim = emb.shape[1]


        labels = labels[:, ::self.network_stride, ::self.network_stride]

        for bs in range(batch_size):
            seg_feat_queue = None
            seg_feat_label_queue = None
            pixel_feat_queue = None
            pixel_feat_label_queue = None
            this_feat = emb[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label).to(torch.int64)
            this_label_ids = [x for x in this_label_ids if x >= 0]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1)
                # feat = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                feat_labels = lb*torch.ones(size=[1,1],device=self.device)
                if seg_feat_queue == None:
                    seg_feat_queue = feat
                    seg_feat_label_queue = feat_labels
                else:
                    seg_feat_queue = torch.cat([seg_feat_queue,feat],dim=1)
                    seg_feat_label_queue = torch.cat([seg_feat_label_queue,feat_labels],dim=1)


                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = idxs[torch.randperm(num_pixel)]
                K = min(num_pixel, self.num_class_sample)
                feat = this_feat[:, perm[:K]].squeeze(-1)
                feat_labels = lb*torch.ones(size=[1,K],device=self.device)
                if pixel_feat_queue == None:
                    pixel_feat_queue = feat
                    pixel_feat_label_queue = feat_labels
                else:
                    pixel_feat_queue = torch.cat([pixel_feat_queue,feat],dim=1)
                    pixel_feat_label_queue = torch.cat([pixel_feat_label_queue,feat_labels],dim=1)
            seg_Q['feature'].append(seg_feat_queue.permute(1,0))
            seg_Q['label'].append(seg_feat_label_queue.permute(1,0))
            pix_Q['feature'].append(pixel_feat_queue.permute(1,0))
            pix_Q['label'].append(pixel_feat_label_queue.permute(1,0))


        return seg_Q,pix_Q

    # def _hard_anchor_sampling(self, emb, labelmap):
    #     batch_size, feat_dim = emb.shape[0], emb.shape[-1]
    #
    #     classes = []
    #     total_classes = 0
    #     for ii in range(batch_size):
    #         this_y = labelmap[ii]
    #         this_classes = torch.unique(this_y)
    #         this_classes = [x for x in this_classes if x != self.ignore_label]
    #         this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
    #
    #         classes.append(this_classes)
    #         total_classes += len(this_classes)
    #
    #     if total_classes == 0:
    #         return None, None
    #
    #
    #     n_view = self.max_samples // total_classes
    #     n_view = min(n_view, self.max_views)
    #
    #     X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float)
    #     y_ = torch.zeros(total_classes, dtype=torch.float)
    #
    #     X_ptr = 0
    #     for ii in range(batch_size):
    #         this_y_hat = labelmap[ii] ## True label
    #         # this_y = y[ii] ## predicted labeel
    #         this_classes = classes[ii]
    #
    #         for cls_id in this_classes:
    #             # hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    #             # easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    #             easy_indices = (this_y_hat == cls_id).nonzero()
    #
    #
    #             # num_hard = hard_indices.shape[0]
    #             num_easy = easy_indices.shape[0]
    #
    #             # if num_hard >= n_view / 2 and num_easy >= n_view / 2:
    #             #     num_hard_keep = n_view // 2
    #             #     num_easy_keep = n_view - num_hard_keep
    #             # elif num_hard >= n_view / 2:
    #             #     num_easy_keep = num_easy
    #             #     num_hard_keep = n_view - num_easy_keep
    #             # elif num_easy >= n_view / 2:
    #             #     num_hard_keep = num_hard
    #             #     num_easy_keep = n_view - num_hard_keep
    #
    #
    #             num_easy_keep = n_view
    #
    #
    #
    #             # perm = torch.randperm(num_hard)
    #             # ### return a randoem permutation of indices of hard anchors!!
    #             # hard_indices = hard_indices[perm[:num_hard_keep]]
    #             perm = torch.randperm(num_easy)
    #             if num_easy < num_easy_keep:
    #                 extra = torch.randint(low=0,high = num_easy,size = [n_view,])
    #                 perm = torch.cat([perm,extra],dim=0)
    #             ### return a randoem permutation of indices of non-hard anchors!!
    #             easy_indices = easy_indices[perm[:num_easy_keep]]
    #             # indices = torch.cat((hard_indices, easy_indices), dim=0)
    #             indices = easy_indices
    #
    #
    #             X_[X_ptr, :, :] = emb[ii, indices, :].squeeze(1)
    #             y_[X_ptr] = cls_id
    #             X_ptr += 1
    #
    #     return X_, y_



    def forward(self, outputs, label_map):
        ## preds:output of network: iincluding feature and predicted seg map
        ## target: real label map
        self.batch_size ,h, w = label_map.size(0),label_map.size(1), label_map.size(2)

        # assert "seg" in preds
        assert "embed" in outputs
        # seg = preds['seg']## output of network
        embedding = outputs['embed']
        self.feature_dim = embedding.shape[1]

        segment_queue, pixel_queue = self.compute_class_prototype(embedding,label_map)

        # if "segment_queue" in preds:
        # segment_queue = outputs['segment_queue']
        # else:
        #     segment_queue = None

        # if "pixel_queue" in preds:
        # pixel_queue = outputs['pixel_queue']
        # else:
        #     pixel_queue = None

        # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        # loss = self.seg_criterion(pred, target)

        # if segment_queue is not None and pixel_queue is not None:
        # queue = torch.cat((segment_queue, pixel_queue), dim=1)

        # _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(segment_queue,pixel_queue,self.batch_size,self.feature_dim)


        # if with_embed is True:
        #     return loss + self.loss_weight * loss_contrast
        return self.loss_weight * loss_contrast

        # return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training


# class ContrastAuxCELoss(nn.Module, ABC):
#     def __init__(self, configer=None):
#         super(ContrastAuxCELoss, self).__init__()
#
#         self.configer = configer
#
#         ignore_index = -1
#         if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
#             ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
#         # Log.info('ignore_index: {}'.format(ignore_index))
#
#         self.loss_weight = self.configer.get('contrast', 'loss_weight')
#         self.use_rmi = self.configer.get('contrast', 'use_rmi')
#
#         if self.use_rmi:
#             self.seg_criterion = FSAuxRMILoss(configer=configer)
#         else:
#             self.seg_criterion = FSAuxCELoss(configer=configer)
#
#         self.contrast_criterion = PixelContrastLoss(configer=configer)
#
#     def forward(self, preds, target):
#         h, w = target.size(1), target.size(2)
#
#         assert "seg" in preds
#         assert "seg_aux" in preds
#
#         seg = preds['seg']
#         seg_aux = preds['seg_aux']
#
#         embedding = preds['embedding'] if 'embedding' in preds else None
#
#         pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
#         pred_aux = F.interpolate(input=seg_aux, size=(h, w), mode='bilinear', align_corners=True)
#         loss = self.seg_criterion([pred_aux, pred], target)
#
#         if embedding is not None:
#             _, predict = torch.max(seg, 1)
#
#             loss_contrast = self.contrast_criterion(embedding, target, predict)
#             return loss + self.loss_weight * loss_contrast
#
#         return loss
