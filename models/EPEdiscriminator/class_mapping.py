import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as tf


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

def compute_mapping_matrix():
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
      print(unlabeled)
      class_mapping_matrix[24,:]=unlabeled
      print(torch.sum(class_mapping_matrix))
      # print(torch.argmax(class_mapping_matrix,dim = 1))
      # print(class_mapping_matrix)
      return class_mapping_matrix


def mapping_labelmap(input_label):
      batch_size = 2
      label_copy = np.empty(input_label.shape, dtype=np.uint8)
      for name,label_list in labels.items():
            label_copy[input_label == label_list[0]] = label_list[1]
      input_label = torch.tensor(label_copy).long().unsqueeze(0).expand(batch_size,1,256,512)

      class_mapping_matrix = torch.zeros(size = [batch_size,25,256,512]).scatter_(1, input_label, 1)

      return input_label.squeeze(1),class_mapping_matrix




# compute_mapping_matrix()