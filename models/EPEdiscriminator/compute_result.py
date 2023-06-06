import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms
import torchvision.transforms as tf
import os
import imageio
import mseg.utils.names_utils as names_utils
import mseg.utils.resize_util as resize_util
from mseg.utils.mask_utils_detectron2 import Visualizer
from pathlib import Path


import config
from train_Unetseg_35 import OASIS_model,put_on_multi_gpus
from models.discriminator import OASIS_Discriminator


labels = {
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
          'unlabeled'            :[  0 ,      24 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'ego vehicle'          :[  1 ,      24 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'rectification border' :[  2 ,      24 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'out of roi'           :[  3 ,      24 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'static'               :[  4 ,      24 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'dynamic'              :[  5 ,      24 , 'void'            , 0       , False        , True         , (111, 74,  0) ],
          'ground'               :[  6 ,      24 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ],
          'road'                 :[  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ],
          'sidewalk'             :[  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ],
          'parking'              :[  9 ,      24 , 'flat'            , 1       , False        , True         , (250,170,160) ],
          'rail track'           :[ 10 ,       21 , 'flat'            , 1       , False        , True         , (230,150,140) ],
          'building'             :[ 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ],
          'wall'                 :[ 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ],
          'fence'                :[ 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ],
          'guard rail'           :[ 14 ,       22 , 'construction'    , 2       , False        , True         , (180,165,180) ],
          'bridge'               :[ 15 ,       20 , 'construction'    , 2       , False        , True         , (150,100,100) ],
          'tunnel'               :[ 16 ,       19 , 'construction'    , 2       , False        , True         , (150,120, 90) ],
          'pole'                 :[ 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ],
          'polegroup'            :[ 18 ,      24 , 'object'          , 3       , False        , True         , (153,153,153) ],
          'traffic light'        :[ 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ],
          'traffic sign'         :[ 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ],
          'vegetation'           :[ 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ],
          'terrain'              :[ 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ],
          'sky'                  :[ 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ],
          'person'               :[ 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ],
          'rider'                :[ 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ],
          'car'                  :[ 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ],
          'truck'                :[ 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ],
          'bus'                  :[ 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ],
          'caravan'              :[ 29 ,      24 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ],
          'trailer'              :[ 30 ,       23 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ],
          'train'                :[ 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ],
          'motorcycle'           :[ 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ],
          'bicycle'              :[ 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ],
          'license plate'        :[ -1 ,      24 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ],
}


labels19 = {
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
          'unlabeled'            :[  0 ,      20 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'ego vehicle'          :[  1 ,      20 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'rectification border' :[  2 ,      20 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'out of roi'           :[  3 ,      20 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'static'               :[  4 ,      20 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
          'dynamic'              :[  5 ,      20 , 'void'            , 0       , False        , True         , (111, 74,  0) ],
          'ground'               :[  6 ,      20 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ],
          'road'                 :[  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ],
          'sidewalk'             :[  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ],
          'parking'              :[  9 ,      20 , 'flat'            , 1       , False        , True         , (250,170,160) ],
          'rail track'           :[ 10 ,      20 , 'flat'            , 1       , False        , True         , (230,150,140) ],
          'building'             :[ 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ],
          'wall'                 :[ 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ],
          'fence'                :[ 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ],
          'guard rail'           :[ 14 ,      20 , 'construction'    , 2       , False        , True         , (180,165,180) ],
          'bridge'               :[ 15 ,      20 , 'construction'    , 2       , False        , True         , (150,100,100) ],
          'tunnel'               :[ 16 ,      20 , 'construction'    , 2       , False        , True         , (150,120, 90) ],
          'pole'                 :[ 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ],
          'polegroup'            :[ 18 ,      20 , 'object'          , 3       , False        , True         , (153,153,153) ],
          'traffic light'        :[ 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ],
          'traffic sign'         :[ 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ],
          'vegetation'           :[ 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ],
          'terrain'              :[ 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ],
          'sky'                  :[ 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ],
          'person'               :[ 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ],
          'rider'                :[ 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ],
          'car'                  :[ 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ],
          'truck'                :[ 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ],
          'bus'                  :[ 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ],
          'caravan'              :[ 29 ,      20 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ],
          'trailer'              :[ 30 ,      20 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ],
          'train'                :[ 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ],
          'motorcycle'           :[ 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ],
          'bicycle'              :[ 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ],
          'license plate'        :[ -1 ,      -1, 'vehicle'         , 7       , False        , True         , (  0,  0,142) ],
}




def compute_mapping_matrix_24():
      class_mapping = torch.zeros(size=(25, 1))
      class_mapping[labels['tunnel'][1]]=31
      class_mapping[labels['bridge'][1]]=32
      class_mapping[labels['building'][1]]=35
      class_mapping[labels['rail track'][1]]=97
      class_mapping[labels['road'][1]]=98
      class_mapping[labels['sidewalk'][1]]=100
      class_mapping[labels['terrain'][1]]=102
      class_mapping[labels['person'][1]]=125
      class_mapping[labels['rider'][1]]=127
      class_mapping[labels['traffic sign'][1]]=135
      class_mapping[labels['traffic light'][1]]=136
      class_mapping[labels['sky'][1]]=142
      class_mapping[labels['pole'][1]]=143
      class_mapping[labels['fence'][1]]=144
      class_mapping[labels['guard rail'][1]]=146
      class_mapping[labels['vegetation'][1]]=174
      class_mapping[labels['bicycle'][1]]=175
      class_mapping[labels['car'][1]]=176
      class_mapping[labels['motorcycle'][1]]=178
      class_mapping[labels['bus'][1]]=180
      class_mapping[labels['train'][1]]=181
      class_mapping[labels['truck'][1]]=182
      class_mapping[labels['trailer'][1]]=183
      class_mapping[labels['wall'][1]]=191

      class_mapping.int()
      zeros = torch.zeros((25,194))
      # print(class_mapping)
      class_mapping = class_mapping.long()
      class_mapping_matrix = zeros.scatter_(1, class_mapping, 1)
      unlabeled = torch.ones([194,])
      indices = [31,32,35,97,98,100,102,125,127,135,136,142,143,144,146,174,175,176,178,180,181,182,183,191]
      unlabeled[indices]=0
      # print(unlabeled)
      class_mapping_matrix[24,:]=unlabeled
      # print(torch.sum(class_mapping_matrix))
      # print(torch.argmax(class_mapping_matrix,dim = 1))
      # print(class_mapping_matrix)
      return class_mapping_matrix


def mapping_labelmap(input_label):
      batch_size = 2
      label_copy = np.empty(input_label.shape, dtype=np.uint8)
      for name,label_list in labels.items():
            label_copy[input_label == label_list[0]] = label_list[1]
      input_label = torch.tensor(label_copy).long().unsqueeze(0).expand(batch_size,1,input_label.shape[0],input_label.shape[1])

      class_mapping_matrix = torch.zeros(size = [batch_size,25,input_label.shape[2],input_label.shape[3]]).scatter_(1, input_label, 1)

      return input_label.squeeze(1),class_mapping_matrix


def mapping_batched_labelmap_35_24(input_label,device):
    label_copy = torch.zeros(input_label.shape, device=device)
    for name, label_list in labels.items():
        label_copy[input_label == label_list[0]] = label_list[1]
    input_label = label_copy.long()

    class_mapping_matrix = torch.zeros(size=[input_label.shape[0], 25, input_label.shape[2], input_label.shape[3]],device=device).scatter_(1,
                                                                                                                   input_label,
                                                                                                                   1)

    return input_label.squeeze(1), class_mapping_matrix

# compute_mapping_matrix()

def compute_miou(pred,labelmap,num_classes, has_gt=True):
    hist = np.zeros((num_classes, num_classes))

    if has_gt:

        hist += fast_hist(pred.flatten(), labelmap.flatten(), num_classes)
        '''            print('===> mAP {mAP:.3f}'.format(
            mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)),end="\r")'''

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


def compute_iou(pred_mask, true_mask, num_classes):
    iou = np.zeros(num_classes)

    for class_id in range(num_classes):
        if ((class_id>=0) and (class_id<num_classes)) :
            pred_class = pred_mask == class_id
            true_class = true_mask == class_id

            intersection = np.logical_and(pred_class, true_class)
            union = np.logical_or(pred_class, true_class)

            iou[class_id] = np.sum(intersection) / np.sum(union)

    return iou


def map_194_to_24(logits):
    mapping_matrix = compute_mapping_matrix_24()
    logits = torch.matmul(mapping_matrix.to(logits.device), logits.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1)
    return logits.permute(0,3,1,2)





def compute_result_24(image_path=None,logits=None,label=None,original_image=None,min_resolution=1080):


    id_to_class_name_map = {info[1]: classname for classname,info in labels.items()}
    id_to_class_name_map[24]='unlabeled'

    in_fname_stem = Path(image_path).stem
    output_gray_fpath = os.path.join('/Users/hlj/Documents/NoSync.nosync/FA/Eskandar_CODE/Code/mseg-semantic/content/result',
                                     in_fname_stem + '_gray')
    output_demo_fpath = os.path.join('/Users/hlj/Documents/NoSync.nosync/FA/Eskandar_CODE/Code/mseg-semantic/content/result',
                                     in_fname_stem + '_overlaid_classes')



    predictions = torch.argmax(logits, dim=1)
    batch_size = logits.shape[0]
    # dump results to disk
    predictions = predictions.data.cpu().numpy()
    gray_batch = np.uint8(predictions)

    labelmap, label = mapping_labelmap(label)
    mapping_matrix = compute_mapping_matrix_24()
    logits = torch.matmul(mapping_matrix, logits.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1)
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.data.cpu().numpy()
    labelmap = labelmap.cpu().detach().numpy()
    gray_batch2 = np.uint8(predictions)
    gray_batch3 = np.uint8(labelmap)
    iou = compute_iou(pred_mask=predictions, true_mask=labelmap, num_classes=24)
    iou = np.nanmean(iou)
    print(iou)

    for i in range(batch_size):
        # rgb_img = np.transpose(input[i,:,:,:].cpu().detach().numpy(),(1,2,0))
        rgb_img = original_image

        # pred_label_img = gray_batch[i, :, :]
        pred_label_img2 = gray_batch2[i, :, :]
        pred_label_img3 = gray_batch3[i, :, :]
        if np.amin(rgb_img.shape[:2]) < min_resolution:
            rgb_img = resize_util.resize_img_by_short_side(rgb_img, min_resolution, "rgb")
            # pred_label_img = resize_util.resize_img_by_short_side(pred_label_img, min_resolution, "label")
            pred_label_img2 = resize_util.resize_img_by_short_side(pred_label_img2, min_resolution, "label")
            pred_label_img3 = resize_util.resize_img_by_short_side(pred_label_img3, min_resolution, "label")

        metadata = None
        frame_visualizer = Visualizer(rgb_img, metadata)
        frame_visualizer2 = Visualizer(rgb_img, metadata)
        frame_visualizer3 = Visualizer(rgb_img, metadata)
        # overlaid_img = frame_visualizer.overlay_instances(
        #     label_map=pred_label_img, id_to_class_name_map=id_to_class_name_map
        # )
        overlaid_img2 = frame_visualizer2.overlay_instances(
            label_map=pred_label_img2, id_to_class_name_map=id_to_class_name_map
        )
        overlaid_img3 = frame_visualizer3.overlay_instances(
            label_map=pred_label_img3, id_to_class_name_map=id_to_class_name_map
        )
        # imageio.imwrite(output_demo_fpath + f'{i}' + '.jpg', overlaid_img)
        imageio.imwrite(output_demo_fpath + f'{i}' + 'predicitions24' + '.jpg', overlaid_img2)
        imageio.imwrite(output_demo_fpath + f'{i}' + 'labelmap24' + '.jpg', overlaid_img3)

        # imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img)
        imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img2)
        imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img3)


def compute_result_19(image_path=None,logits=None,label=None,original_image=None,min_resolution=1080):


    id_to_class_name_map = {info[1]: classname for classname,info in labels.items()}
    id_to_class_name_map[24]='unlabeled'

    in_fname_stem = Path(image_path).stem
    output_gray_fpath = os.path.join('/Users/hlj/Documents/NoSync.nosync/FA/Eskandar_CODE/Code/mseg-semantic/content/result',
                                     in_fname_stem + '_gray')
    output_demo_fpath = os.path.join('/Users/hlj/Documents/NoSync.nosync/FA/Eskandar_CODE/Code/mseg-semantic/content/result',
                                     in_fname_stem + '_overlaid_classes')



    predictions = torch.argmax(logits, dim=1)
    batch_size = logits.shape[0]
    # dump results to disk
    predictions = predictions.data.cpu().numpy()
    gray_batch = np.uint8(predictions)

    labelmap, label = mapping_labelmap_19(label)

    labelmap = labelmap.cpu().detach().numpy()
    gray_batch2 = np.uint8(predictions)
    gray_batch3 = np.uint8(labelmap)
    miou = compute_iou(pred_mask=predictions, true_mask=labelmap, num_classes=19)
    print(miou)

    for i in range(batch_size):
        # rgb_img = np.transpose(input[i,:,:,:].cpu().detach().numpy(),(1,2,0))
        rgb_img = original_image

        # pred_label_img = gray_batch[i, :, :]
        pred_label_img2 = gray_batch2[i, :, :]
        pred_label_img3 = gray_batch3[i, :, :]
        if np.amin(rgb_img.shape[:2]) < min_resolution:
            rgb_img = resize_util.resize_img_by_short_side(rgb_img, min_resolution, "rgb")
            # pred_label_img = resize_util.resize_img_by_short_side(pred_label_img, min_resolution, "label")
            pred_label_img2 = resize_util.resize_img_by_short_side(pred_label_img2, min_resolution, "label")
            pred_label_img3 = resize_util.resize_img_by_short_side(pred_label_img3, min_resolution, "label")

        metadata = None
        frame_visualizer = Visualizer(rgb_img, metadata)
        frame_visualizer2 = Visualizer(rgb_img, metadata)
        frame_visualizer3 = Visualizer(rgb_img, metadata)
        # overlaid_img = frame_visualizer.overlay_instances(
        #     label_map=pred_label_img, id_to_class_name_map=id_to_class_name_map
        # )
        overlaid_img2 = frame_visualizer2.overlay_instances(
            label_map=pred_label_img2, id_to_class_name_map=id_to_class_name_map
        )
        overlaid_img3 = frame_visualizer3.overlay_instances(
            label_map=pred_label_img3, id_to_class_name_map=id_to_class_name_map
        )
        # imageio.imwrite(output_demo_fpath + f'{i}' + '.jpg', overlaid_img)
        imageio.imwrite(output_demo_fpath + f'{i}' + 'predicitions24' + '.jpg', overlaid_img2)
        imageio.imwrite(output_demo_fpath + f'{i}' + 'labelmap24' + '.jpg', overlaid_img3)

        # imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img)
        imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img2)
        imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img3)





def compute_result_35(image_path=None,logits=None,labelmap=None,original_image=None,min_resolution=1080):


    id_to_class_name_map = {info[0]: classname for classname,info in labels.items()}
    id_to_class_name_map[34]='unlabeled'

    in_fname_stem = Path(image_path).stem
    output_gray_fpath = os.path.join('/Users/hlj/Documents/NoSync.nosync/FA/no_backups/s1434/oasisbackup/pretrained_models/msegsemantic/content/result',
                                     in_fname_stem + '_gray_35')
    output_demo_fpath = os.path.join('/Users/hlj/Documents/NoSync.nosync/FA/no_backups/s1434/oasisbackup/pretrained_models/msegsemantic/content/result',
                                     in_fname_stem + '_overlaid_classes_35')



    predictions = torch.argmax(logits, dim=1)
    batch_size = logits.shape[0]
    # dump results to disk
    predictions = predictions.data.cpu().numpy()
    gray_batch = np.uint8(predictions)


    labelmap = labelmap.cpu().detach().numpy()
    gray_batch2 = np.uint8(predictions)
    gray_batch3 = np.uint8(labelmap)
    miou = compute_iou(pred_mask=predictions, true_mask=labelmap, num_classes=35)
    print(miou)

    for i in range(batch_size):
        # rgb_img = np.transpose(input[i,:,:,:].cpu().detach().numpy(),(1,2,0))
        rgb_img = original_image

        # pred_label_img = gray_batch[i, :, :]
        pred_label_img2 = gray_batch2[i, :, :]
        pred_label_img3 = gray_batch3[i, :, :]
        if np.amin(rgb_img.shape[:2]) < min_resolution:
            rgb_img = resize_util.resize_img_by_short_side(rgb_img, min_resolution, "rgb")
            # pred_label_img = resize_util.resize_img_by_short_side(pred_label_img, min_resolution, "label")
            pred_label_img2 = resize_util.resize_img_by_short_side(pred_label_img2, min_resolution, "label")
            pred_label_img3 = resize_util.resize_img_by_short_side(pred_label_img3, min_resolution, "label")

        metadata = None
        frame_visualizer = Visualizer(rgb_img, metadata)
        frame_visualizer2 = Visualizer(rgb_img, metadata)
        frame_visualizer3 = Visualizer(rgb_img, metadata)
        # overlaid_img = frame_visualizer.overlay_instances(
        #     label_map=pred_label_img, id_to_class_name_map=id_to_class_name_map
        # )
        overlaid_img2 = frame_visualizer2.overlay_instances(
            label_map=pred_label_img2, id_to_class_name_map=id_to_class_name_map
        )
        overlaid_img3 = frame_visualizer3.overlay_instances(
            label_map=pred_label_img3, id_to_class_name_map=id_to_class_name_map
        )
        # imageio.imwrite(output_demo_fpath + f'{i}' + '.jpg', overlaid_img)
        imageio.imwrite(output_demo_fpath + f'{i}' + 'predicitions24' + '.jpg', overlaid_img2)
        imageio.imwrite(output_demo_fpath + f'{i}' + 'labelmap24' + '.jpg', overlaid_img3)

        # imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img)
        imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img2)
        imageio.imwrite(output_gray_fpath + f'{i}' + '.jpg', pred_label_img3)


if __name__ =="__main__":
    opt = config.read_arguments(train=True)
    opt.gpu_ids = '-1'
    opt.load_size = 512
    opt.crop_size = 512
    opt.label_nc = 34
    opt.contain_dontcare_label = True
    opt.semantic_nc = 35  # label_nc + unknown
    opt.cache_filelist_read = False
    opt.cache_filelist_write = False
    opt.aspect_ratio = 2.0
    opt.checkpoints_dir = '/Users/hlj/Documents/NoSync.nosync/FA/no_backups/s1434/oasisbackup/checkpoints/Unet_dis_16_35_miou_05_2'

    device = "cpu" if opt.gpu_ids == '-1' else 'cuda'
    print("nb of gpus: ", torch.cuda.device_count())
    model = OASIS_model(opt)
    model = put_on_multi_gpus(model, opt)
    which_iter = opt.which_iter
    path = os.path.join(opt.checkpoints_dir, opt.name, "models", str(which_iter) + "_")
    model.module.netD.load_state_dict(torch.load(path + "D.pth",map_location='cpu'))
    if not opt.no_EMA:
        model.module.netEMA.load_state_dict(torch.load(path + "EMA.pth",map_location='cpu'))
    image_path = '/Users/hlj/Documents/NoSync.nosync/FA/no_backups/s1434/oasisbackup/pretrained_models/msegsemantic/content/city2-480-360SS.png'
    label_path = '/Users/hlj/Documents/NoSync.nosync/FA/no_backups/s1434/oasisbackup/pretrained_models/msegsemantic/content/city2label.png'
    origin_image = np.array(Image.open(image_path))
    image = torchvision.transforms.functional.to_tensor(Image.open(image_path).convert('RGB'))
    image = torchvision.transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).unsqueeze(0)
    label = (torchvision.transforms.functional.to_tensor(Image.open(label_path)).unsqueeze(0)*255)[:,0,:,:]

    with torch.no_grad():
        logits = model.module.netEMA(image)

    compute_result_35(image_path,logits,label,origin_image,1080)