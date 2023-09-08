'''
Inference code for OnlineRefer, on Ref-DAVIS17
Modified from ReferFormer (https://github.com/wjn922/ReferFormer)
Ref-Davis17 does not support visualize
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import torch.nn.functional as F
import json

import opts
from tqdm import tqdm

import multiprocessing as mp
import threading

from tools_refer.colormap import colormap

import supervision as sv
import torchvision
# from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
from segment_anything import build_sam, SamPredictor, build_sam_hq

# Grounding DINO
# import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict


from libs.model import Model_switchGTfixdot_swCC_Res as Model
from libs.utils import norm_mask

import pandas as pd

import libs.transforms_pair as transforms

import torch.nn as nn
import ipdb
import csv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

### Settings for Grounding Dino & SAM
# GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"



BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

def load_model_hf(args, repo_id, filename, ckpt_config_filename, device='cpu'):
    # particularly import build_model since the function name is same as onlinerefer
    if args.use_SAM:
        from GroundingDINO.groundingdino.models import build_model
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
   
    args_gdino = SLConfig.fromfile(cache_config_file) 
    model = build_model(args_gdino)
    args_gdino.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def main(args):
    args.dataset_file = "davis"
    args.masks = True
    args.batch_size == 1
    print("Inference only supports for batch size = 1")
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path(args.davis_path)  # data/ref-davis
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    # create subprocess
    thread_num = args.ngpu
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    ### Note: workaround to avoid the multi-process issue...
    sub_video_list = video_list[0:]
    sub_processor(lock, 0, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, sub_video_list)

    # for i in range(thread_num):
    #     if i == thread_num - 1:
    #         sub_video_list = video_list[i * per_thread_video_num:]
    #     else:
    #         sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
    #     p = mp.Process(target=sub_processor, args=(lock, i, args, data,
    #                                                save_path_prefix, save_visualize_path_prefix,
    #                                                img_folder, sub_video_list))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" % (total_time))

def transform_topk(aff, frame1, k, h2=None, w2=None):
    """
    INPUTS:
        - aff: affinity matrix, b * N * N
        - frame1: reference frame
        - k: only aggregate top-k pixels with highest aff(j,i)
        - h2, w2, frame2's height & width
    OUTPUT:
        - frame2: propagated mask from frame1 to the next frame
    """
    b,c,h,w = frame1.size()
    b, N1, N2 = aff.size()
    # b * 20 * N
    tk_val, tk_idx = torch.topk(aff, dim = 1, k=k)
    # b * N
    tk_val_min,_ = torch.min(tk_val,dim=1)
    tk_val_min = tk_val_min.view(b,1,N2)
    aff[tk_val_min > aff] = 0
    frame1 = frame1.contiguous().view(b,c,-1)
    
    # print(frame1.sum())
    # print(aff.sum())
    # aff = aff.unsqueeze(0)
    # aff = torch.nn.functional.interpolate(aff, size=(frame1.shape[-1], frame1.shape[-1]), mode='bilinear')
    # aff = aff.squeeze(0)

    frame2 = torch.bmm(frame1, aff)

    if(h2 is None):
        return frame2.view(b,c,h,w)
    else:
        return frame2.view(b,c,h2,w2)

def propagate(frame1, frame2, model, seg, img_folder, video_name):
    """
    propagate seg of frame1 to frame2
    """
    frame1 = preprocess_frame_uvc(frame1, img_folder, video_name)
    frame2 = preprocess_frame_uvc(frame2, img_folder, video_name)
    # print(frame1.sum())
    # print(frame2.sum())
    # print(f'frame1: {frame1.shape}')
    # print(f'frame2: {frame2.shape}')
    # TODO: seg is also need to be adjusted..
    seg = preprocess_seg_uvc(seg)
    # print(seg)
    # print(seg.sum())
    # print(seg.shape)

    n, c, h, w = frame1.size()
    frame1_gray = frame1[:,0].view(n,1,h,w)
    frame2_gray = frame2[:,0].view(n,1,h,w)
    frame1_gray = frame1_gray.repeat(1,3,1,1)
    frame2_gray = frame2_gray.repeat(1,3,1,1)

   

    output = model(frame1_gray, frame2_gray, frame1, frame2)
    ## TODO: use frame1_gray & frame2_gray to obtain aff. directly (low priority.)
        # (no use of model)
    aff = output[2]
    # print(seg)
    # print(seg.shape)
    # print(f'aff {aff.sum()}')
    # print(f'aff: {aff.shape}')
    # print(f'seg: {seg.shape}')
    # print(seg)

    frame2_seg = transform_topk(aff,seg.cuda(),k=args.topk)
    # print(f'frame2_seg {frame2_seg.sum()}')


    return frame2_seg

class NLM_woSoft(nn.Module):
    """
    Non-local mean layer w/o softmax on affinity
    """
    def __init__(self):
        super(NLM_woSoft, self).__init__()

    def forward(self, in1, in2):
        # print(in1.shape)
        # print(in2.shape)
        n,c,h,w = in1.size()    
        in1 = in1.view(n,c,-1)
        in2 = in2.view(n,c,-1)
        affinity = torch.bmm(in1.permute(0,2,1), in2)
        
        # affinity = torch.bmm(in1, in2.permute(0, 2, 1))
        return affinity

def propagate_feat(feat1, feat2, seg):
    """
    propagate seg of frame1 to frame2
    """
    ## TODO: use frame1_gray & frame2_gray to obtain aff. directly (low priority.)
        # (no use of model)
    nlm = NLM_woSoft()
    softmax_func = nn.Softmax(dim=1)
    seg = preprocess_seg_uvc(seg)

    feat1 = torch.nn.functional.interpolate(feat1, size=(seg.shape[-2], seg.shape[-1]), mode='bilinear')
    feat2 = torch.nn.functional.interpolate(feat2, size=(seg.shape[-2], seg.shape[-1]), mode='bilinear')
    # small_seg = np.array(Image.fromarray(seg_ori).resize((th//8,tw//8), 0)).astype(np.float32)
    aff_matrix_func = nlm(feat1, feat2)
    # print(aff_matrix_func)
    # print(aff_matrix_func.shape)
    # import ipdb
    # ipdb.set_trace()
    aff = softmax_func(aff_matrix_func * args.temp)
    # print(aff.sum())
    # print(aff)
    frame2_seg = transform_topk(aff,seg.cuda(),k=args.topk)


    return frame2_seg

def create_transforms():
    normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))
    t = []
    t.extend([transforms.ToTensor(),
    		  normalize])
    return transforms.Compose(t)

def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2,0,1).unsqueeze(0)


### BUG: seems related to here!!!
def preprocess_seg_uvc(seg_map):
    # seg = Image.open(seg_dir)
    h,w = seg_map.shape[-1], seg_map.shape[-2]

    if(len(args.scale_size) == 1):
        if(h > w):
            tw = args.scale_size[0]
            th = (tw * h) / w
            th = int((th // 64) * 64)
        else:
            th = args.scale_size[0]
            tw = (th * w) / h
            tw = int((tw // 64) * 64)
    else:
        tw = args.scale_size[1]
        th = args.scale_size[0]
    seg_map = seg_map.cpu().numpy().reshape((w,h,1))
    seg_ori = np.squeeze(seg_map)
    # print(seg_ori)
    # print(seg_ori.shape)
    # Note: resize for PIL, 0 means the nearest mode
    small_seg = np.array(Image.fromarray(seg_ori).resize((th//8,tw//8), 0)).astype(np.float32)
    # print('Before')
    # print(np.unique(small_seg))
    # print(small_seg.shape)
    # large_seg = np.array(Image.fromarray(seg_ori).resize((tw,th), 0))
    # small_seg = scipy.misc.imresize(seg_ori, (tw//8,th//8),"nearest",mode="F")
    # large_seg = scipy.misc.imresize(seg_ori, (tw,th),"nearest",mode="F")
    # print(small_seg)
    # print(small_seg.shape)

    t = []
    t.extend([transforms.ToTensor()])
    trans = transforms.Compose(t)
    pair = [small_seg, small_seg]
    transformed = list(trans(*pair))
    small_seg = transformed[0]
    # small_seg = trans(small_seg)
    
    # print('After')
    # print(np.unique(small_seg))
    # print(small_seg.shape)
    # large_seg = transformed[0]
    # small_seg = transformed[1]
    ### BUG: to one hot??
    # return small_seg.unsqueeze(0)

    # print(tmp)
    # print(tmp.shape)
    return to_one_hot(small_seg)

def preprocess_frame_uvc(frame_num, img_folder, video_name):
    # frame = cv2.imread(frame_dir)
    frame_tmp = cv2.imread(os.path.join(img_folder, video_name, frame_num + ".jpg"))
    ori_h,ori_w,_ = frame_tmp.shape
    # scale, makes height & width multiples of 64
    if(len(args.scale_size) == 1):
        if(ori_h > ori_w):
            tw = args.scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = args.scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        tw = args.scale_size[1]
        th = args.scale_size[0]
    frame_tmp = cv2.resize(frame_tmp, (tw,th))
    frame_tmp = cv2.cvtColor(frame_tmp, cv2.COLOR_BGR2LAB)

    pair = [frame_tmp, frame_tmp]
    trans = create_transforms()
    out = list(trans(*pair))
    return out[0].cuda().unsqueeze(0)


# Define a function to save bounding box data to a CSV file
def save_bbox_to_csv(output_csv, data):
    # Check if the file exists to decide whether to write headers
    write_headers = not os.path.exists(output_csv)

    with open(output_csv, 'a', newline='') as file:
        fieldnames = ["VideoName", "ImageName", "ObjectID", "xmin", "ymin", "xmax", "ymax", "Confidence", "Class"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        for entry in data:
            writer.writerow(entry)


def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)
    ## TODO: Build model with SAM
    if args.use_SAM:
        
        ## 1. Build Grounding dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        print('\n**** USE SAM ****\n')
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        # grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        grounding_dino_model = load_model_hf(args, ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        ## 2. Building SAM Model and SAM Predictor
        device = torch.device('cuda:0')
        # sam_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        sam_hq_checkpoint = args.sam_ckpt_path
       
        # sam = build_sam(checkpoint=sam_checkpoint)
        sam = build_sam_hq(checkpoint=sam_hq_checkpoint, mask_threshold=args.mask_threshold)
        # use_sam_hq
        sam.to(device=device)
        sam.eval()
        sam_predictor = SamPredictor(sam)

        if args.use_UVC:
            print('\n**** USE UVC to calibrate SAM!! ****\n')
            # loading pretrained model from UVC
            model_uvc = Model(pretrainRes=False, temp = args.temp, uselayer=4)
            # if(args.multiGPU):
            #     model = nn.DataParallel(model)
            checkpoint = torch.load(args.uvc_checkpoint_dir)
            best_loss = checkpoint['best_loss']
            # Note: this is needed if we do not use DDP explicitly
            model_uvc.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint['state_dict'].items()})
            print("=> loaded checkpoint '{} ({})' (epoch {})"
                .format(args.uvc_checkpoint_dir, best_loss, checkpoint['epoch']))
            model_uvc.to(device=device)
            model_uvc.eval()
    else:
        # model
        model, criterion, _ = build_model(args)
        device = args.device
        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if pid == 0:
            print('number of params:', n_parameters)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
        else:
            raise ValueError('Please specify the checkpoint for inference.')

        # start inference
        model.eval()

    # get palette
    palette_img = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette = Image.open(palette_img).getpalette()

    num_all_frames = 0
    font_path = "fonts/OpenSans-Regular.ttf"
    font = ImageFont.truetype(font_path, 30) # change the '30' to any size you want
    # 1. for each video
    for video in video_list:
        metas = []

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]  # start from 0
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # since there are 4 annotations
        num_obj = num_expressions // 4

        image_cache_for_video = {}
        # 2. for each annotator
        for anno_id in range(4):  # 4 annotators
            all_exps = []
            bbox_data = []
            anno_logits = []
            anno_masks = []  # [num_obj+1, video_len, h, w], +1 for background
            anno_boxes = []
            anno_frame_origin = []
            for obj_id in range(num_obj):
                i = obj_id * 4 + anno_id
                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]
                frames = meta[i]["frames"]
                video_len = len(frames)
                # NOTE: the im2col_step for MSDeformAttention is set as 64
                # so the max length for a clip is 64
                # store the video pred results
                all_pred_logits = []
                all_pred_masks = []
                all_pred_boxes = []
                all_frame_origin = []

                if args.semi_online:
                    num_clip_frames = args.num_frames
                else:
                    num_clip_frames = 1

                # TODO: setup for propagate func.

                ### NOT DONE
                prev_logits = torch.zeros(size=(1,))
                prev_frame = None
                prev_img = None
                prev_mask = None

                # 3. for each clip
                # track_res = model.generate_empty_tracks()
                for clip_id in range(0, video_len, num_clip_frames):
                    frames_ids = [x for x in range(video_len)]
                    clip_frames_ids = frames_ids[clip_id: clip_id + num_clip_frames]
                    clip_len = len(clip_frames_ids)

                    # load the clip images
                    imgs = []
                    pred_masks = []
                    pred_logits = []
                    pred_boxes = []
                  

                    for t in clip_frames_ids:
                        # print('current t', t)
                        # print('current clip_id', clip_id)
                        frame = frames[t]
                        img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                        
                        # Use the cached image if it's already loaded, otherwise load and cache it
                        if img_path not in image_cache_for_video:
                            image_cache_for_video[img_path] = Image.open(img_path).convert('RGB')
                        # img = Image.open(img_path).convert('RGB')
                        img = image_cache_for_video[img_path].copy()
                        origin_w, origin_h = img.size
                        imgs.append(transform(img))  # list[Img]

                        ## TODO: maybe directly perform prediction here? 
                        # so we can avoid the batch inference issues..
                        with torch.no_grad(): #### REMEMBER ADD THIS when infernce
                            if args.use_SAM:
                                # TODO: perform detection with Grounding DINO
                                # detect objects
                                boxes, logits, phrases, intermediate_feat = predict(
                                    model=grounding_dino_model, 
                                    image=transform(img), 
                                    caption=exp,
                                    box_threshold=BOX_THRESHOLD, 
                                    text_threshold=TEXT_THRESHOLD
                                )
                                # Note!!! handle special cases for "india"
                                if torch.numel(boxes) == 0:
                                    boxes_xyxy = torch.zeros((1, 4))
                                    ### DONE:
                                    # if this situation, no need SAM! (just all zeros)
                                    masks = torch.zeros(masks.shape).to(device)
                                    prev_frame = frame
                                    prev_mask = masks
                                    pred_masks.append(masks)
                                    pred_boxes.append(boxes_xyxy)
                                    pred_logits.append(torch.zeros((1,)))
                                else:
                                    #### MEMO:
                                    # G-DINO will have the logit for next frame
                                    # We have: 
                                        # prev. frame's mask and logit (from bbox)
                                    # use test.py from UVC to obtain the next frame prediction (by prop.)
                                    # still, we will run the SAM obtain the frame predcition 
                                    #   (this pred. will be used in the next round)
                                    # how to decide the final pred. => use the logit to select
                                        # logits will be also prop. if we really prop.!!
                                        ### Note: test.py does not need to include avg. operation here!
                                    
                                    max_logit, max_idx = torch.max(logits, dim=0)
                                    # select high conf. box
                                    boxes = boxes[max_idx].unsqueeze(0) ## shape: (1, 4)

                                    img_arr = np.asarray(img)
                                    H, W, _ = img_arr.shape
                                    # box: normalized box xywh -> unnormalized xyxy
                                    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                                    # TODO: perform selection
                                    # handle this to decide prop. or not
                                    # if prev_logits.item() > max_logit and t != 0:
                                    if max_logit < args.prop_thres and t != 0:
                                        # select prop. frames
                                        # print(f'{frame}, t: {t}, prop, logits: {max_logit}, prev logits: {prev_logits.item()}')
                                        # masks = propagate(prev_frame,
                                        #                 frame, 
                                        #                 model_uvc, 
                                        #                 prev_mask,
                                        #                 img_folder,
                                        #                 video_name)
                                        masks = propagate_feat(
                                            prev_intermediate_feat,
                                            intermediate_feat,
                                            prev_mask
                                        )
                                     
                                        # df = pd.DataFrame(masks.view(-1, masks.shape[-1]).cpu().numpy().astype(np.uint8))
                                        # df.to_csv(f'obj_{obj_id}_tmp_{t}.csv')
                                        # print(f'save {t}, obj: {obj_id}')
                                        # TODO: upsampling 
                                        masks = torch.nn.functional.interpolate(masks,scale_factor=8,mode='bilinear')
                                        masks = norm_mask(masks.squeeze(0))
                                        # masks = masks.squeeze(0)
                                        masks = torchvision.transforms.functional.resize(masks, (origin_h,origin_w), Image.NEAREST)
                                        masks = masks.unsqueeze(0)
                                        # masks = masks[0][1].unsqueeze(0)

                                        _, masks = torch.max(masks, dim=1)
                                        # print(prev_mask.shape)
                                        ## TODO: masks need to be prop. also!! (20230816)
                                        prev_frame = frame
                                        prev_intermediate_feat = intermediate_feat
                                        # print(masks.shape)
                                        prev_mask = masks.to(torch.uint8).squeeze(0)

                                        pred_masks.append(masks)
                                        ## TODO:
                                        # save pred_logit?
                                        pred_logits.append(max_logit.unsqueeze(0))
                                        pred_boxes.append(boxes_xyxy)
                                        ind = 'UVC'
                                    else:
                                        # select curr. frames by inferencing with SAM
                                        # TODO: perform seg. with SAM
                                        
                                        sam_predictor.set_image(img_arr)
                                        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, img_arr.shape[:2]).to(device)
                                        # print(transformed_boxes)
                                        # print(transformed_boxes.shape)
                                        masks, _, _ = sam_predictor.predict_torch(
                                                    point_coords = None,
                                                    point_labels = None,
                                                    boxes = transformed_boxes,
                                                    multimask_output = False,
                                                )
                                        ### TODO: check:
                                        ### if SAM segment nothing, or only few pixels compared with the size the bbox, then we also propagate the mask.
                                        # print(masks)
                                        # import ipdb
                                        # ipdb.set_trace()
                                        if masks.sum() < args.pixel_thres:
                                            # print('propagate')
                                            masks = propagate_feat(
                                                prev_intermediate_feat,
                                                intermediate_feat,
                                                prev_mask
                                            )
                                            masks = torch.nn.functional.interpolate(masks,scale_factor=8,mode='bilinear')
                                            masks = norm_mask(masks.squeeze(0))
                                            masks = torchvision.transforms.functional.resize(masks, (origin_h,origin_w), Image.NEAREST)
                                            masks = masks.unsqueeze(0)
                                            _, masks = torch.max(masks, dim=1)
                                            prev_mask = masks.to(torch.uint8).squeeze(0)
                                            prev_frame = frame
                                            prev_logits = max_logit
                                            prev_intermediate_feat = intermediate_feat
                                            pred_masks.append(masks)
                                            ## TODO:
                                            # save pred_logit?
                                            pred_logits.append(max_logit.unsqueeze(0))
                                            pred_boxes.append(boxes_xyxy)
                                            ind = 'SAM'
                                        else:
                                            prev_mask = masks
                                            prev_frame = frame
                                            prev_logits = max_logit
                                            prev_intermediate_feat = intermediate_feat

                                            pred_masks.append(masks)
                                            ## TODO:
                                            # save pred_logit?
                                            pred_logits.append(max_logit.unsqueeze(0))
                                            pred_boxes.append(boxes_xyxy)
                                            ind = 'SAM'

                                        # print(masks.unique())
                                        # print(masks.shape)
                                        # if not the first frame, then it must pass once of GroundedSAM
                                        # that is, masks has value for prev.
                                        # If go into SAM and not is not the first frame
                                        # => update the queue (prev. variables)
                                        # print(f'{frame}, t: {t}, SAM, logits: {max_logit}, prev logits: {prev_logits.item()}')


                                        # prev_mask = torchvision.transforms.functional.pil_to_tensor(Image.open('/home/liujack/RVOS/OnlineRefer/data/ref-davis/valid/Annotations/bike-packing/00000.png'))
                                        # prev_mask[prev_mask==2] = 0
                                        
                                        # if masks.shape[0] != 1:
                                        #     prev_mask = masks[0][0].unsqueeze(0)
                                        # else:
                                        #     prev_mask = masks
                                     
                    # frame_origin = torch.cat(frame_origin, dim=0) 
                    # frame_origin shape: [t,]
                    pred_masks = torch.cat(pred_masks, dim=0)  # [t, h, w],

                    ## adjust to align num of dim for prop. and SAM
                    if pred_masks.dim() == 4:
                        pred_masks = pred_masks.squeeze(0)
                    pred_logits = torch.cat(pred_logits, dim=0)
                    pred_boxes = torch.cat(pred_boxes, dim=0)
                    all_pred_masks.append(pred_masks)
                    all_pred_logits.append(pred_logits)
                    ### save boxes
                    all_pred_boxes.append(pred_boxes)
                    all_frame_origin.append(ind)
                # print(all_frame_origin)
                # print(len(all_frame_origin))
                all_pred_logits = torch.cat(all_pred_logits, dim=0)  # (video_len, K)
                all_pred_masks = torch.cat(all_pred_masks, dim=0)  # (video_len, h, w)
                all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
                # all_frame_origin = torch.cat(all_frame_origin, dim=0)
                all_exps.append(exp)
                # print(all_pred_masks.shape)
                # for f in range(all_pred_masks_plot.shape[0]):
                #     df = pd.DataFrame(all_pred_masks_plot[f].astype(np.uint8))
                #     df.to_csv(f'obj_{obj_id}_tmp_{f}.csv')
                #     print(f'save {f}, obj: {obj_id}')
                # anno_logits.append(all_pred_logits)
                anno_logits.append(all_pred_logits)
                anno_masks.append(all_pred_masks)
                anno_boxes.append(all_pred_boxes)
                anno_frame_origin.append(all_frame_origin)

                # handle a complete image (all objects of a annotator)
            anno_logits = torch.stack(anno_logits)  # [num_obj, video_len, k]
            anno_masks = torch.stack(anno_masks)  # [num_obj, video_len, h, w]
            # print(anno_masks)
            # print(anno_masks.shape)
            t, h, w = anno_masks.shape[-3:]
            anno_masks[anno_masks < 0.5] = 0.0
            background = 0.1 * torch.ones(1, t, h, w).to(args.device)
            anno_masks = torch.cat([background, anno_masks], dim=0)  # [num_obj+1, video_len, h, w]
            out_masks = torch.argmax(anno_masks, dim=0)  # int, the value indicate which object, [video_len, h, w]
            out_masks = out_masks.detach().cpu().numpy().astype(np.uint8)  # [video_len, h, w]
            # print(anno_frame_origin)
            # print(len(anno_frame_origin))
            # print(anno_frame_origin[0])
            # print(len(anno_frame_origin[0]))
           
            # save results
            anno_save_path = os.path.join(save_path_prefix, f"anno_{anno_id}", video)
            if not os.path.exists(anno_save_path):
                os.makedirs(anno_save_path)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))

                if args.visualize:
                    exp_idx = 0
                    # for k in range(num_obj, 0, -1):
                    image_cache = {}
                    for k in range(1, num_obj + 1):
                        img_path = os.path.join(img_folder, video_name, frames[f] + ".jpg")
                        # Use the cached image if it's already loaded, otherwise load and cache it
                        if img_path not in image_cache:
                            image_cache[img_path] = Image.open(img_path).convert('RGBA')

                        # source_img = Image.open(img_path).convert('RGBA')
                        source_img = image_cache[img_path].copy()
                        origin_w, origin_h = source_img.size
                        draw = ImageDraw.Draw(source_img)
                        # text = expressions[expression_list[i]]["exp"]
                        # Example: bike-packing => ['a black bike', 'a man wearing a cap']
                        # print(all_exps)
                        # print(len(all_exps))
                        text = all_exps[exp_idx]
                        text = text + f', {anno_frame_origin[k - 1][f]}'

                        # import ipdb; ipdb.set_trace()
                        # font_path = os.path.join(os.getcwd(),"fonts/OpenSans-Regular.ttf")
                        position = (10, 10)
                        draw.text(position, text, (255, 0, 0), font=font)

                        ### if UVC, then does not need to plot box
                        draw_boxes = anno_boxes[k - 1][f].unsqueeze(0)
                        # draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()
                        xmin, ymin, xmax, ymax = draw_boxes[0]
                        if anno_frame_origin[k - 1][f] == 'SAM':
                            # draw_boxes = anno_boxes[k - 1][f].unsqueeze(0)
                            # draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()
                            # xmin, ymin, xmax, ymax = draw_boxes[0]
                            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[k%len(color_list)]), width=2)
                            # plot the bbox score
                            plot_bbox_score = f'{anno_logits[k - 1][f].item():.2f}'
                            draw.text((xmin, ymin), plot_bbox_score, (255, 0, 0), font=font)

                        
                        mask_arr = np.array(img_E)
                        plot_mask = np.zeros_like(mask_arr)
                        
                        # if num_obj > 1: # if more than 1 obj., adjust the mask
                        plot_mask[mask_arr==k] = k
                        # other wise do nothing

                        # df = pd.DataFrame(plot_mask)
                        # df.to_csv(f'tmp{k}.csv')
                        source_img = vis_add_mask(source_img, plot_mask, color_list[k%len(color_list)])
                        save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                        save_visualize_path_dir = os.path.join(save_visualize_path_dir, f'obj_{k}')
                        if not os.path.exists(save_visualize_path_dir):
                            os.makedirs(save_visualize_path_dir)
                        save_visualize_path = os.path.join(save_visualize_path_dir,  frames[f] + f'_{k}.png')
                        source_img.save(save_visualize_path)
                        exp_idx += 1
                        bbox_info = {
                            "VideoName": video_name,
                            "ImageName": frames[f] + ".jpg",
                            "ObjectID": k,
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "Confidence": anno_logits[k - 1][f].item()
                        }
                        bbox_data.append(bbox_info)
            # Define the path to the output CSV file
            output_csv = os.path.join(anno_save_path, "bounding_box_data.csv")
            # Call the function to save the bounding box data to the CSV file
            save_bbox_to_csv(output_csv, bbox_data)
        with lock:
            progress.update(1)
    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


# Post-process functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x - 10, y, x + 10, y), tuple(cur_color), width=4)
        draw.line((x, y - 10, x, y + 10), tuple(cur_color), width=4)


def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=tuple(cur_color), outline=tuple(cur_color), width=1)


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')  # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OnlineRefer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)

