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
from libs.track_utils import match_ref_tar, bbox_in_tar_scale, squeeze_all

import pandas as pd

import libs.transforms_pair as transforms

import torch.nn as nn
import ipdb

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


def adjust_bbox(bbox_now, bbox_pre, a, h, w):
    """
    Adjust a bounding box w.r.t previous frame,
    assuming objects don't go under abrupt changes.
    """

    cnt = 1
    bbox_now_h = (bbox_now[cnt].top  + bbox_now[cnt].bottom) / 2.0
    bbox_now_w = (bbox_now[cnt].left + bbox_now[cnt].right) / 2.0

    bbox_now_height_ = bbox_now[cnt].bottom - bbox_now[cnt].top
    bbox_now_width_  = bbox_now[cnt].right  - bbox_now[cnt].left

    bbox_pre = bbox_pre.squeeze()
    # bbox_pre (xyxy format): (x1, y1, x2, y2) 
    # define its bottom, top, left, right
    # x1, y1 being top left and x2, y2 being bottom right.
    x1, y1, x2, y2 = bbox_pre[0].item(), bbox_pre[1].item(), bbox_pre[2].item(), bbox_pre[3].item()
    bbox_pre_height = y2 - y1
    bbox_pre_width  = x2 - x1

    bbox_now_height = a * bbox_now_height_ + (1 - a) * bbox_pre_height
    bbox_now_width  = a * bbox_now_width_  + (1 - a) * bbox_pre_width

    bbox_now[cnt].left   = math.floor(bbox_now_w - bbox_now_width / 2.0)
    bbox_now[cnt].right  = math.ceil(bbox_now_w + bbox_now_width / 2.0)
    bbox_now[cnt].top    = math.floor(bbox_now_h - bbox_now_height / 2.0)
    bbox_now[cnt].bottom = math.ceil(bbox_now_h + bbox_now_height / 2.0)

    bbox_now[cnt].left = max(0, bbox_now[cnt].left)
    bbox_now[cnt].right = min(w, bbox_now[cnt].right)
    bbox_now[cnt].top = max(0, bbox_now[cnt].top)
    bbox_now[cnt].bottom = min(h, bbox_now[cnt].bottom)

    # for cnt in bbox_pre.keys():
    #     if(cnt == 0):
    #         continue
    #     if(cnt in bbox_now and bbox_pre[cnt] is not None and bbox_now[cnt] is not None):
    #         bbox_now_h = (bbox_now[cnt].top  + bbox_now[cnt].bottom) / 2.0
    #         bbox_now_w = (bbox_now[cnt].left + bbox_now[cnt].right) / 2.0

    #         bbox_now_height_ = bbox_now[cnt].bottom - bbox_now[cnt].top
    #         bbox_now_width_  = bbox_now[cnt].right  - bbox_now[cnt].left

    #         bbox_pre_height = bbox_pre[cnt].bottom - bbox_pre[cnt].top
    #         bbox_pre_width  = bbox_pre[cnt].right  - bbox_pre[cnt].left

    #         bbox_now_height = a * bbox_now_height_ + (1 - a) * bbox_pre_height
    #         bbox_now_width  = a * bbox_now_width_  + (1 - a) * bbox_pre_width

    #         bbox_now[cnt].left   = math.floor(bbox_now_w - bbox_now_width / 2.0)
    #         bbox_now[cnt].right  = math.ceil(bbox_now_w + bbox_now_width / 2.0)
    #         bbox_now[cnt].top    = math.floor(bbox_now_h - bbox_now_height / 2.0)
    #         bbox_now[cnt].bottom = math.ceil(bbox_now_h + bbox_now_height / 2.0)

    #         bbox_now[cnt].left = max(0, bbox_now[cnt].left)
    #         bbox_now[cnt].right = min(w, bbox_now[cnt].right)
    #         bbox_now[cnt].top = max(0, bbox_now[cnt].top)
    #         bbox_now[cnt].bottom = min(h, bbox_now[cnt].bottom)

    return bbox_now

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


def bbox2mask(bbox, shape):
    """
    Convert a bounding box to a binary mask using PyTorch.

    Parameters:
    - bbox: a tensor of the form [[cx, cy, w, h]]
            with normalized values between [0, 1]
    - shape: torch.Size object (e.g., torch.Size([1, 1, 480, 910]))

    Returns:
    - mask: a tensor of the given shape with values set to 1 inside the bbox and 0 outside
    """
    height, width = shape[2], shape[3]
    
    # Convert normalized bbox to absolute coordinates
    bbox_abs = (bbox * torch.tensor([width, height, width, height])).int()

    if len(bbox_abs.shape) != 2:
        bbox_abs = bbox_abs.unsqueeze(0)

    # Convert (cx, cy, w, h) format to (xmin, ymin, xmax, ymax) format
    cx, cy, w, h = bbox_abs[0][0].item(), bbox_abs[0][1].item(), bbox_abs[0][2].item(), bbox_abs[0][3].item()
    xmin = cx - w // 2
    xmax = cx + w // 2
    ymin = cy - h // 2
    ymax = cy + h // 2
    
    # Create a meshgrid of shape (height, width)
    rows = torch.arange(0, height, dtype=torch.int64).unsqueeze(1)
    cols = torch.arange(0, width, dtype=torch.int64)
    
    # Create the mask
    mask = (rows >= ymin) & (rows < ymax) & (cols >= xmin) & (cols < xmax)
    
    # Adjust the mask shape to match the given shape (e.g., [1, 1, 480, 910])
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    
    return mask

def compute_iou(boxA, boxB, width, height):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes using torchvision.
    
    Each bbox is of the form [1, 4]: [[xmin, ymin, xmax, ymax]]
    """
    if len(boxA.shape) != 2:
        boxA = boxA.unsqueeze(0)
    if len(boxB.shape) != 2:
        boxB = boxB.unsqueeze(0)
    # Scale bounding boxes
    boxA_scaled = boxA * torch.tensor([width, height, width, height])
    boxB_scaled = boxB * torch.tensor([width, height, width, height])
    boxA_xyxy = torchvision.ops.box_convert(boxA_scaled, in_fmt='cxcywh', out_fmt='xyxy')
    boxB_xyxy = torchvision.ops.box_convert(boxB_scaled, in_fmt='cxcywh', out_fmt='xyxy')

    # import ipdb; ipdb.set_trace()
    iou_matrix = torchvision.ops.box_iou(boxA_xyxy, boxB_xyxy)
    
    # Since both boxA and boxB are single boxes, the IoU will be a 1x1 matrix. We extract the scalar value.
    iou_value = iou_matrix[0, 0].item()

    return iou_value


def propagate_bbox(F_ref, F_tar, seg_ref, bbox_ref):
    # F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
    # seg_ref = seg_ref.squeeze(0)
    F_ref, F_tar = squeeze_all(F_ref, F_tar)
    
    ## OK: turn bbox into mask and send into match_ref_tar
    seg_ref = bbox2mask(bbox_ref, seg_ref.shape)

    seg_ref = preprocess_seg_uvc(seg_ref)

    F_ref = torch.nn.functional.interpolate(F_ref.unsqueeze(0), size=(seg_ref.shape[-2], seg_ref.shape[-1]), mode='bilinear')
    F_tar = torch.nn.functional.interpolate(F_tar.unsqueeze(0), size=(seg_ref.shape[-2], seg_ref.shape[-1]), mode='bilinear')
    
    F_ref = F_ref.squeeze(0)
    F_tar = F_tar.squeeze(0)
    seg_ref = seg_ref.squeeze(0)
    c, h, w = F_ref.size()
    

    # get coordinates of each point in the target frame
    coords_ref_tar = match_ref_tar(F_ref, F_tar, seg_ref, args.temp)
    # coordinates -> bbox
    # bbox_tar = bbox_in_tar_scale(coords_ref_tar, bbox_ref, h, w)
    # adjust bbox
    # bbox_tar = adjust_bbox(bbox_tar, bbox_ref, 0.1, h, w)
    
    ## DONE: need to confirm again the adjust_bbox usage...
    if coords_ref_tar.get(1) != None:
        min_vals, _ = torch.min(coords_ref_tar[1], dim=0)
        max_vals, _ = torch.max(coords_ref_tar[1], dim=0)
        # x1, y1, x2, y2
        bbox_tar = torch.cat([min_vals, max_vals])
        bbox_tar = bbox_tar / torch.tensor([w, h, w, h]).cuda()
    else:
        min_vals, _ = torch.min(coords_ref_tar[0], dim=0)
        max_vals, _ = torch.max(coords_ref_tar[0], dim=0)
        # x1, y1, x2, y2
        bbox_tar = torch.cat([min_vals, max_vals])
        bbox_tar = bbox_tar / torch.tensor([w, h, w, h]).cuda()
    
    return bbox_tar, coords_ref_tar



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
        sam_hq_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_hq_vit_h.pth'
       
        # sam = build_sam(checkpoint=sam_checkpoint)
        sam = build_sam_hq(checkpoint=sam_hq_checkpoint)
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

        # 2. for each annotator
        for anno_id in range(4):  # 4 annotators
            all_exps = []
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
                prev_bbox = torch.zeros(size=(1, 4))

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
                        img = Image.open(img_path).convert('RGB')
                        origin_w, origin_h = img.size
                        imgs.append(transform(img))  # list[Img]
                        masks = torch.zeros((1, 1, origin_h, origin_w)) # init for each frame

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
                                if torch.numel(boxes) == 0:
                                    boxes_xyxy = torch.zeros((1, 4))
                                    ### DONE:
                                    # if this situation, no need SAM! (just all zeros)
                                    masks = torch.zeros((1, 1, origin_h, origin_w)).to(device)
                                    prev_frame = frame
                                    prev_mask = masks
                                    prev_bbox = boxes_xyxy
                                    pred_masks.append(masks)
                                    pred_boxes.append(boxes_xyxy)
                                    pred_logits.append(torch.zeros((1,)))
                                    ind = 'Prop.'
                                    continue

                                max_logit, max_idx = torch.max(logits, dim=0)
                                is_all_zeros = torch.all(prev_bbox == 0).item()
                                if max_logit < args.prop_thres and t != 0 and not is_all_zeros:
                                    out_boxes, coords = propagate_bbox(prev_intermediate_feat,
                                                               intermediate_feat,
                                                               prev_mask,
                                                               prev_bbox, # need to store G
                                                            #    origin_h,
                                                            #    origin_w
                                                               ) # produce by UVC
                                    # recover to xyxy format
                                    # bbox_pre (xyxy format): (x1, y1, x2, y2) 
                                    # define its bottom, top, left, right
                                    # x1, y1 being top left and x2, y2 being bottom right.
                                    out_boxes = torchvision.ops.box_convert(out_boxes, in_fmt='xyxy', out_fmt='cxcywh').cpu()
                                    ind = 'Prop.'
                                else:
                                    # select high conf. box
                                    # TODO: iou score 
                                    # Need to filter out the thres hold accoding to the logits of G-DINO
                                    # Handle the case when there is no bbox from prop. (all zeros)
                                    
                                    # calculate IOU of prev_bbox and all boxes
                                    filtered_bboxes = boxes[logits > args.prop_thres]
                                    if torch.numel(filtered_bboxes) == 0:
                                        # this case, we found that prev_bbox is zero, thus we directly use SAM this frame
                                        # Thus, take the max.
                                        out_boxes = boxes[max_idx].unsqueeze(0).cpu() ## shape: (1, 4)
                                        # print('QQ')
                                        # print(f'{boxes} {logits} {filtered_bboxes} {conf_scores}')
                                    else:
                                        ious = [compute_iou(prev_bbox, bbox, origin_w, origin_h) for bbox in filtered_bboxes]
                                        conf_scores = args.iou_alpha * logits[logits > args.prop_thres] + (1 - args.iou_alpha) * torch.tensor(ious)
                                        # print(f'{boxes} {logits} {filtered_bboxes} {conf_scores}')
                                        # print(logits[logits > args.prop_thres])
                                        # print(ious)
                                        # print(scores)
                                        # print('----')
                                        _, conf_idx = torch.max(conf_scores, dim=0)
                                        out_boxes = boxes[conf_idx].unsqueeze(0).cpu() ## shape: (1, 4)
                                        ind = 'G-Dino'

                                # Note!!! handle special cases for "india"
                                # print(out_boxes)
                                if torch.numel(out_boxes) == 0:
                                    boxes_xyxy = torch.zeros((1, 4))
                                    ### DONE:
                                    # if this situation, no need SAM! (just all zeros)
                                    masks = torch.zeros(masks.shape).to(device)
                                    prev_frame = frame
                                    prev_mask = masks
                                    prev_bbox = boxes_xyxy
                                    pred_masks.append(masks)
                                    pred_boxes.append(boxes_xyxy)
                                    pred_logits.append(torch.zeros((1,)))
                                else:
                                    img_arr = np.asarray(img)
                                    H, W, _ = img_arr.shape
                                    # box: normalized box xywh -> unnormalized xyxy
                                    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_boxes) * torch.Tensor([W, H, W, H])
                                    # print(boxes_xyxy)
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
                                    prev_mask = masks
                                    prev_bbox = out_boxes
                                    prev_frame = frame
                                    prev_logits = max_logit
                                    prev_intermediate_feat = intermediate_feat
                                    pred_masks.append(masks)
                                    ## TODO:
                                    # save pred_logit?
                                    pred_logits.append(max_logit.unsqueeze(0))
                                    pred_boxes.append(boxes_xyxy)
                    pred_masks = torch.cat(pred_masks, dim=0)  # [t, h, w],
                    pred_boxes = torch.cat(pred_boxes, dim=0)
                    pred_logits = torch.cat(pred_logits, dim=0)
                    # print(f'{pred_masks.shape}, {ind}')


                    ## adjust to align num of dim for prop. and SAM
                    if all_pred_masks: # not empty
                        if pred_masks.dim() < all_pred_masks[-1].dim():
                            all_pred_masks[-1] = all_pred_masks[-1].view(pred_masks.shape)
                        elif pred_masks.dim() > all_pred_masks[-1].dim():
                            pred_masks = pred_masks.view(all_pred_masks[-1].shape)

                    if pred_masks.dim() == 4:
                        pred_masks = pred_masks.squeeze(0)
                    # if pred_masks.dim() == 2:
                    #     pred_masks = pred_masks.unsqueeze(0)

                    if pred_boxes.dim() == 1:
                        pred_boxes = pred_boxes.unsqueeze(0)
                    all_pred_masks.append(pred_masks)
                    all_pred_logits.append(pred_logits)
                    ### save boxes
                    all_pred_boxes.append(pred_boxes)
                    all_frame_origin.append(ind)
                all_pred_logits = torch.cat(all_pred_logits, dim=0)  # (video_len, K)
                all_pred_masks = torch.cat(all_pred_masks, dim=0)  # (video_len, h, w)
                all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
                all_exps.append(exp)
              
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
                    for k in range(1, num_obj + 1):
                        img_path = os.path.join(img_folder, video_name, frames[f] + ".jpg")
                        source_img = Image.open(img_path).convert('RGBA')
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
                        font_path = "fonts/OpenSans-Regular.ttf"
                        font = ImageFont.truetype(font_path, 30) # change the '30' to any size you want
                        position = (10, 10)
                        draw.text(position, text, (255, 0, 0), font=font)

                        ### if UVC, then does not need to plot box
                        # if anno_frame_origin[k - 1][f] == 'SAM':
                        draw_boxes = anno_boxes[k - 1][f].unsqueeze(0)
                        # draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()
                        xmin, ymin, xmax, ymax = draw_boxes[0]
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

