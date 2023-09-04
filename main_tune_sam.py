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
from engine import train_one_epoch_sam, evaluate, evaluate_a2d, evaluate_online_a2d
from models import build_model

from tools_refer.load_pretrained_weights import pre_trained_model_to_finetune

# NOTE: modified from opts.py
import opts_tune_sam as opts

import ipdb


from tensorboardX import SummaryWriter

from segment_anything import build_sam, SamPredictor, build_sam_hq

from sam_lora_image_encoder_mask_decoder import LoRA_Sam
# NOTE: these two follow OnlineRefer now (can be modified later if needed)
from models.matcher import build_matcher
from models.criterion import SetCriterion

def build_criterion(args):
    if args.binary:
        num_classes = 1
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:  # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha)
    criterion.to(args.device)
    return criterion

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


    ## TODO: build model SAM
    ## NOTE: If there is bug, then tune SAM first, since I cannot ensure the correctness of the LORA model
    print('\n**** USE SAM ****\n')
    ## 2. Building SAM Model and SAM Predictor
    device = torch.device(args.device)
    # sam_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    # sam_hq_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_hq_vit_h.pth'
    sam_hq_checkpoint = args.sam_ckpt_path
    
    
    # sam = build_sam(checkpoint=sam_checkpoint)
    sam = build_sam_hq(checkpoint=sam_hq_checkpoint)
    # use_sam_hq
    sam.to(device=device)
    
    ## DONE! adapter here?
    # lora_sam = LoRA_Sam(sam, r=4) # r is the rank of LORA
    model = LoRA_Sam(sam, args.lora_rank).cuda()
    # names = []
    # for k, p in model.named_parameters():
    #     if p.requires_grad:
    #         names.append(k)
    # print(names)
    # print(len(names))

    # sam_predictor = SamPredictor(lora_sam.sam)
    # model = sam_predictor.model



    # model, criterion, postprocessor = build_model(args)
    # model.to(device)

    # TODO: build criterion???
    # criterion = build_criterion(args)
    

    #### TODO: multi-gpu is not supported yet
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    print('number of params for LORA:', n_parameters)
    print('Total number of params for original SAM:', total_parameters)


    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad],
            "lr": args.lr,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    # no validation ground truth for ytvos dataset
    dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)



    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    

    
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

    
    if args.dataset_file == "davis":
        assert args.pretrained_weights is not None, "Please provide the pretrained weight to finetune for Ref-DAVIS17"
        print("============================================>")
        print("Ref-DAVIS17 are finetuned using the checkpoint trained on Ref-Youtube-VOS")
        print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        # if 'output' in args.pretrained_weights:
        #     checkpoint = {'model': checkpoint['model']}
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")

    if args.dataset_file == "jhmdb":
        assert args.resume is not None, "Please provide the checkpoint to resume for JHMDB-Sentences"
        print("============================================>")
        print("JHMDB-Sentences are directly evaluated using the checkpoint trained on A2D-Sentences")
        print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
        # load checkpoint in the args.resume
        print("============================================>")

    # for Ref-Youtube-VOS and A2D-Sentences
    # finetune using the pretrained weights on Ref-COCO
    if args.dataset_file != "davis" and args.dataset_file != "jhmdb" and args.pretrained_weights is not None:
        print("============================================>")
        print("Load pretrained weights from {} ...".format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")


    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    # if args.eval:
    #     assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
    #                 'Only A2D-Sentences and JHMDB-Sentences datasets support evaluation'
    #     if args.semi_online:
    #         test_stats = evaluate_online_a2d(model, data_loader_val, postprocessor, device, args)
    #     else:
    #         test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
    #     return


    print("Start training")
    writer = SummaryWriter('log/{}'.format(args.output_dir.split('/')[-1]))
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        print('Current number of training frames: {}'.format(dataset_train.num_frames))
        # dataset_train.step_epoch()
        # test_stats = evaluate_online_a2d(model, data_loader_val, postprocessor, device, args)
        train_stats = train_one_epoch_sam(
            model, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args, writer)
        
        torch.cuda.empty_cache()
        lr_scheduler.step()
        dataset_train.step_epoch()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # if args.dataset_file == 'a2d':
        #     if args.semi_online:
        #         test_stats = evaluate_online_a2d(model, data_loader_val, postprocessor, device, args)
        #         log_stats.update({**{f'{k}': v for k, v in test_stats.items()}})
        #     else:
        #         test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
        #         log_stats.update({**{f'{k}': v for k, v in test_stats.items()}})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



