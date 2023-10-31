"""
Training script of OnlineRefer
Modified from ReferFormer (https://github.com/wjn922/ReferFormer)
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch_sam, evaluate, evaluate_a2d, evaluate_online_a2d, evaluate_a2d_g_sam, evaluate_jhmdb_g_sam
from models import build_model

from tools_refer.load_pretrained_weights import pre_trained_model_to_finetune

import opts

import ipdb
import os


from tensorboardX import SummaryWriter

from segment_anything import build_sam, SamPredictor, build_sam_hq

from sam_lora_image_encoder_mask_decoder import LoRA_Sam
# NOTE: these two follow OnlineRefer now (can be modified later if needed)
# from models.matcher import build_matcher
# from models.criterion import SetCriterion

import cv2
import mmcv
from mmdet.apis import DetInferencer


def main(args):
    args.masks = True

    # utils.init_distributed_mode(args)
    args.distributed = False
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    
    print(f'\n Run on {args.dataset_file} dataset.')
    print('\n')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    ## TODO: build G-DINO with mmdet.
    # load the model from mmdet
    # Specify the path to model config and checkpoint file
    # bash ./scripts/online_a2d.sh ./outputs_tmp_a2d ../Grounded-Segment-Anything/sam_hq_vit_h.pth --g_dino_config_path ./mmdetection/configs/grounding_dino/ours_framewise.py --g_dino_ckpt_path ./mm_weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth

    config_file = args.g_dino_config_path
    checkpoint_file = args.g_dino_ckpt_path
    
    # Build the model from a config file and a checkpoint file
    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda:0', show_progress=False)

    ## 2. Building SAM Model and SAM Predictor
    print('\n**** USE SAM ****\n')
    device = torch.device(args.device)
    # sam_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    # sam_hq_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_hq_vit_h.pth'
    sam_hq_checkpoint = args.sam_ckpt_path
    lora_sam_ckpt_path = args.lora_sam_ckpt_path
    if args.use_LORA_SAM:
        sam = build_sam_hq(checkpoint=sam_hq_checkpoint, mask_threshold=args.mask_threshold)
        # use_sam_hq
        sam.to(device=device)
        lora_sam = LoRA_Sam(sam, args.lora_rank).cuda()
        model = lora_sam
        ckpt = torch.load(lora_sam_ckpt_path)
        # print("=================================================================================")
        model.load_state_dict(ckpt['model'])
        # print("=================================================================================")
        sam_predictor = SamPredictor(model.sam)
        print("Load LoRA-SAM model from {}".format(lora_sam_ckpt_path))
    else:
        # sam = build_sam(checkpoint=sam_checkpoint)
        sam = build_sam_hq(checkpoint=sam_hq_checkpoint, mask_threshold=args.mask_threshold)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)

    # A2D-Sentences
    if args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
        dataset_val = build_dataset(args.dataset_file, image_set='val', args=args)
    output_dir = Path(args.output_dir)

    assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
                'Only A2D-Sentences and JHMDB-Sentences datasets support evaluation'
    
    if args.dataset_file == 'a2d':
        test_stats = evaluate_a2d_g_sam(sam_predictor, inferencer, dataset_val, device, args)
    elif args.dataset_file == 'jhmdb':
        test_stats = evaluate_jhmdb_g_sam(sam_predictor, inferencer, dataset_val, device, args)

    with open(os.path.join(output_dir, 'log.json'), 'w') as f:
        json.dump(test_stats, f)
    print('Inference done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



