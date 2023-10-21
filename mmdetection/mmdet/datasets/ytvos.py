# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .base_video_dataset import BaseVideoDataset
import mmengine
import os

"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
from mmdet.datasets.utils_rvos import transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

from mmdet.datasets.utils_rvos.box_ops import box_xyxy_to_cxcywh

import mmcv

# -------------------------------------------------------------------------------------------------------------------
# 1. Ref-Youtube-VOS
ytvos_category_dict = {
    'airplane': 0, 'ape': 1, 'bear': 2, 'bike': 3, 'bird': 4, 'boat': 5, 'bucket': 6, 'bus': 7, 'camel': 8, 'cat': 9, 
    'cow': 10, 'crocodile': 11, 'deer': 12, 'dog': 13, 'dolphin': 14, 'duck': 15, 'eagle': 16, 'earless_seal': 17, 
    'elephant': 18, 'fish': 19, 'fox': 20, 'frisbee': 21, 'frog': 22, 'giant_panda': 23, 'giraffe': 24, 'hand': 25, 
    'hat': 26, 'hedgehog': 27, 'horse': 28, 'knife': 29, 'leopard': 30, 'lion': 31, 'lizard': 32, 'monkey': 33, 
    'motorbike': 34, 'mouse': 35, 'others': 36, 'owl': 37, 'paddle': 38, 'parachute': 39, 'parrot': 40, 'penguin': 41, 
    'person': 42, 'plant': 43, 'rabbit': 44, 'raccoon': 45, 'sedan': 46, 'shark': 47, 'sheep': 48, 'sign': 49, 
    'skateboard': 50, 'snail': 51, 'snake': 52, 'snowboard': 53, 'squirrel': 54, 'surfboard': 55, 'tennis_racket': 56, 
    'tiger': 57, 'toilet': 58, 'train': 59, 'truck': 60, 'turtle': 61, 'umbrella': 62, 'whale': 63, 'zebra': 64
}

ytvos_category_list = [
    'airplane', 'ape', 'bear', 'bike', 'bird', 'boat', 'bucket', 'bus', 'camel', 'cat', 'cow', 'crocodile', 
    'deer', 'dog', 'dolphin', 'duck', 'eagle', 'earless_seal', 'elephant', 'fish', 'fox', 'frisbee', 'frog', 
    'giant_panda', 'giraffe', 'hand', 'hat', 'hedgehog', 'horse', 'knife', 'leopard', 'lion', 'lizard', 
    'monkey', 'motorbike', 'mouse', 'others', 'owl', 'paddle', 'parachute', 'parrot', 'penguin', 'person', 
    'plant', 'rabbit', 'raccoon', 'sedan', 'shark', 'sheep', 'sign', 'skateboard', 'snail', 'snake', 'snowboard', 
    'squirrel', 'surfboard', 'tennis_racket', 'tiger', 'toilet', 'train', 'truck', 'turtle', 'umbrella', 'whale', 'zebra'
]

def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


@DATASETS.register_module()
class YTVOSDataset(Dataset):
    """YouTube VOS dataset for RVOS.
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.
    """

    def __init__(self, ytvos_dict, *args, **kwargs):
        self.img_folder = ytvos_dict['img_folder']
        self.ann_file = ytvos_dict['ann_file']
        self._transforms = make_coco_transforms(image_set='train', max_size=640)
        # self.return_masks = return_masks # not used
        self.num_frames = ytvos_dict['num_frames']
        self.num_clips = ytvos_dict['num_clips']
        # self.max_skip = max_skip
        self.sampler_interval = ytvos_dict['sampler_interval']
        # self.reverse_aug = args.reverse_aug
        # create video meta data
        self.prepare_metas()

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')

        # video sampler.
        self.sampler_steps: list = ytvos_dict['sampler_steps']
        self.lengths: list = ytvos_dict['sampler_lengths']
        self.current_epoch = 0
        print("sampler_steps={} lengths={}".format(self.sampler_steps, self.lengths))

    
    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            # prepare for neg text
            vid_dict_for_neg = {}
            for key, value in vid_data['expressions'].items():
                obj_id = value['obj_id']
                exp = value['exp']
                
                if obj_id not in vid_dict_for_neg:
                    vid_dict_for_neg[obj_id] = []
                
                vid_dict_for_neg[obj_id].append(exp)

            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # DONE: add neg. text for contrastive loss
                    meta['neg_exp'] = []
                    for k, v in vid_dict_for_neg.items():
                        if k != exp_dict['obj_id']:
                            meta['neg_exp'].extend(v)
                    meta['exp_for_debug'] = vid_data['expressions']
                    # get object category
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)
        
    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.current_epoch = self.current_epoch + 1

        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        self.period_idx = 0
        for i in range(len(self.sampler_steps)):
            if self.current_epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(self.current_epoch, self.period_idx))
        self.num_frames = self.lengths[self.period_idx]

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, obj_id, category, frames, frame_id = \
                meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
           
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = ytvos_category_dict[category]
            vid_len = len(frames)

            # TODO: obtain neg. text for contrastive loss
            all_exp_for_obj = meta['neg_exp']
            if len(all_exp_for_obj) > 0:
                neg_exp = random.choice(all_exp_for_obj)
            else:
                neg_exp = ''
            # clean up the caption (neg.)
            neg_exp = " ".join(neg_exp.lower().split())

            num_frames = self.num_frames * self.num_clips
            # random sparse sample
            sample_indx = [frame_id]
            if num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, self.sampler_interval)
                sample_id_after = random.randint(1, self.sampler_interval)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # maximum number of input frames is 5 for online mode
                if num_frames > 3:
                    sample_id_before = random.randint(1, self.sampler_interval)
                    sample_id_after = random.randint(1, self.sampler_interval)
                    local_indx = [max(0, frame_id - self.sampler_interval - sample_id_before),
                                  min(vid_len - 1, frame_id + self.sampler_interval + sample_id_after)]
                    sample_indx.extend(local_indx)

            sample_indx.sort()
            # if random.random() < self.reverse_aug:
            #     sample_indx = sample_indx[::-1]

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                
                # img = Image.open(img_path).convert('RGB')
                img = mmcv.imread(img_path)
                
                mask = Image.open(mask_path).convert('P')

                # create the target
                label = torch.tensor(category_id)
                mask = np.array(mask)
                mask = (mask == obj_id).astype(np.float32)  # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:  # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            # w, h = img.size
            width, height = img.shape[1], img.shape[0]
            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': torch.tensor(sample_indx),  # [T,]
                'labels': labels,  # [T,]
                'boxes': boxes,  # [T, 4], xyxy
                'masks': masks,  # [T, H, W]
                'valid': torch.tensor(valid),  # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(height), int(width)]),
                'size': torch.as_tensor([int(height), int(width)]),
                'neg_caption': neg_exp
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            # imgs, target = self._transforms(imgs, target)
            imgs = [torch.tensor(img).permute(2,0,1) for img in imgs]
            # if "boxes" in target:
            #     boxes = target["boxes"]
            #     boxes = box_xyxy_to_cxcywh(boxes)
            #     boxes = boxes / torch.tensor([width, height, width, height], dtype=torch.float32)
            #     target["boxes"] = boxes
            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]
            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)
        return imgs, target
    
    def get_data_info(self, idx):
        return self.__getitem__(idx)
    

