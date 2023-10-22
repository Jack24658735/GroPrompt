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
from engine import train_one_epoch_sam, evaluate, evaluate_a2d, evaluate_online_a2d, evaluate_a2d_g_sam
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

    utils.init_distributed_mode(args)
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

    # model, criterion, postprocessor = build_model(args)
    # model.to(device)
    

    #### TODO: multi-gpu is not supported yet
    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # no validation ground truth for ytvos dataset
    # dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)



    # if args.distributed:
    #     if args.cache_mode:
    #         sampler_train = samplers.NodeDistributedSampler(dataset_train)
    #     else:
    #         sampler_train = samplers.DistributedSampler(dataset_train)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    

    
    # A2D-Sentences
    if args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
        dataset_val = build_dataset(args.dataset_file, image_set='val', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)

    

    # if args.dataset_file == "jhmdb":
    #     assert args.resume is not None, "Please provide the checkpoint to resume for JHMDB-Sentences"
    #     print("============================================>")
    #     print("JHMDB-Sentences are directly evaluated using the checkpoint trained on A2D-Sentences")
    #     print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
    #     # load checkpoint in the args.resume
    #     print("============================================>")

    # # for Ref-Youtube-VOS and A2D-Sentences
    # # finetune using the pretrained weights on Ref-COCO
    # if args.dataset_file != "davis" and args.dataset_file != "jhmdb" and args.pretrained_weights is not None:
    #     print("============================================>")
    #     print("Load pretrained weights from {} ...".format(args.pretrained_weights))
    #     checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
    #     checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
    #     model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
    #     print("============================================>")


    output_dir = Path(args.output_dir)
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    #     unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    #     if len(missing_keys) > 0:
    #         print('Missing Keys: {}'.format(missing_keys))
    #     if len(unexpected_keys) > 0:
    #         print('Unexpected Keys: {}'.format(unexpected_keys))
    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         import copy
    #         p_groups = copy.deepcopy(optimizer.param_groups)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         for pg, pg_old in zip(optimizer.param_groups, p_groups):
    #             pg['lr'] = pg_old['lr']
    #             pg['initial_lr'] = pg_old['initial_lr']
    #         # print(optimizer.param_groups)
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
    #         args.override_resumed_lr_drop = True
    #         if args.override_resumed_lr_drop:
    #             print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
    #             lr_scheduler.step_size = args.lr_drop
    #             lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
    #         lr_scheduler.step(lr_scheduler.last_epoch)
    #         args.start_epoch = checkpoint['epoch'] + 1

    assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
                'Only A2D-Sentences and JHMDB-Sentences datasets support evaluation'
    # if args.semi_online:
    #     test_stats = evaluate_online_a2d(model, data_loader_val, postprocessor, device, args)
    # else:
    #     test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
    test_stats = evaluate_a2d_g_sam(sam_predictor, inferencer, data_loader_val, device, args)
    with open(os.path.join(output_dir, 'log.json'), 'w') as f:
        json.dump(test_stats, f)
    print('Inference done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



