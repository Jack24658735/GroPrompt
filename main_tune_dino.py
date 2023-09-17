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
from engine import train_one_epoch_gdino, evaluate, evaluate_a2d, evaluate_online_a2d, train_one_epoch

# from models import build_model

from tools_refer.load_pretrained_weights import pre_trained_model_to_finetune

# NOTE: modified from opts.py
import opts_tune_gdino as opts

import ipdb

from huggingface_hub import hf_hub_download

# Grounding DINO
# import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.models import build_model

from tensorboardX import SummaryWriter

# from gdino_lora import LoRA_GDino

from GroundingDINO.groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from GroundingDINO.groundingdino.models.GroundingDINO.backbone import build_backbone
from GroundingDINO.groundingdino.models.GroundingDINO.transformer import build_transformer

from models.matcher import build_matcher_GDINO
from models.criterion import SetCriterion


def build_groundingdino_finetune(args, args_ours):
     ## TODO:
    num_classes = 1
    # device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    matcher = build_matcher_GDINO(args_ours)
    weight_dict = {}
    weight_dict['loss_ce'] = args_ours.cls_loss_coef
    weight_dict['loss_bbox'] = args_ours.bbox_loss_coef
    weight_dict['loss_giou'] = args_ours.giou_loss_coef

    # TODO this is a hack
    # if args_ours.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args_ours.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args_ours.eos_coef,
        losses=losses,
        focal_alpha=args_ours.focal_alpha)


    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    ## TODO: is needed to postprocessor?

    return model, criterion



def load_model_hf(args, repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
   
    args_gdino = SLConfig.fromfile(cache_config_file)
    if args.finetune_gdino_mode:
        args_gdino.finetune_gdino_mode = True
        # args_gdino.modelname = 'groundingdino_finetune'
        model, criterion = build_groundingdino_finetune(args_gdino, args)
        print('Build G-DINO for fine-tuning.')
    else:
        model = build_model(args_gdino)
        print('Build G-DINO for inference.')


    args_gdino.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    if args_gdino.finetune_gdino_mode:
        model.train()
        return model, criterion
    else:
        _ = model.eval()
        return model


def main(args):
    ### NOTE: this flag sould be set to False for training since we only tune the bbox
    args.masks = False

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

    # print('\n**** USE SAM ****\n')
    device = torch.device(args.device)
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    # grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    if args.finetune_gdino_mode:
        model, criterion = load_model_hf(args, ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
        model.to(device)
        criterion.to(device)
    else:
        model = load_model_hf(args, ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
        model.to(device)
    
    # TODO: LoRA_GDino is under construction
    # model = LoRA_GDino(grounding_dino_model, args.lora_rank).cuda()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    ## Freeze backbone
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    ## Freeze transformer
    # for param in model.module.transformer.parameters():
    #     param.requires_grad = False
    ## Freeze input_proj
    # for param in model.module.input_proj.parameters():
    #     param.requires_grad = False
    ## Freeze bert
    for param in model.module.bert.parameters():
        param.requires_grad = False

    # # Gather parameter counts for each module in the model
    # module_params = [(name, param.numel()) for name, param in model.named_parameters()]

    # # Sort modules by number of parameters in descending order
    # sorted_modules = sorted(module_params, key=lambda x: x[1], reverse=True)

    # # Print sorted list
    # for name, param_count in sorted_modules:
    #     print(f"Module: {name}, Parameters: {param_count}")
    

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    print('number of params for tuning:', n_parameters)
    print('Total number of params for original G-DINO:', total_parameters)


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
        train_stats = train_one_epoch_gdino(
            model, criterion, data_loader_train, optimizer, device, epoch,
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



