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
# from models import build_model
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
from segment_anything import build_sam, SamPredictor, build_sam_hq

import mmcv
from mmdet.apis import DetInferencer
import torch.nn as nn

import csv

from sam_lora_image_encoder_mask_decoder import LoRA_Sam

from libs.track_utils import match_ref_tar, squeeze_all
import libs.transforms_pair as transforms

import warnings
import torchvision
from collections import Counter
warnings.filterwarnings("ignore")

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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
    # global result_dict
    # result_dict = mp.Manager().dict()

    processes = []
    # lock = threading.Lock()
    lock = mp.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    ### Note: workaround to avoid the multi-process issue...
    sub_video_list = video_list[0:]
    sub_processor(lock, 0, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, sub_video_list)

    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   save_path_prefix, save_visualize_path_prefix,
                                                   img_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    # result_dict = dict(result_dict)
    # num_all_frames_gpus = 0
    # for pid, num_all_frames in result_dict.items():
    #     num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" % (total_time))


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


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x

def add_lora(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    r = 4 # LORA rank (can be modified)
    assert r > 0
    dim = model.decoder.layers[0].cross_attn.value_proj.in_features

    for param in model.parameters():
        param.requires_grad = False
    # create for storage, then we can init them or load weights
    w_As = []  # These are linear layers
    w_Bs = []
    for layer in model.decoder.layers:
        # layer.cross_attn.value_proj
        # layer.cross_attn.output_proj
        w_q_linear = layer.cross_attn.value_proj
        w_v_linear = layer.cross_attn.output_proj
        w_a_linear_q = nn.Linear(dim, r, bias=False)
        w_b_linear_q = nn.Linear(r, dim, bias=False)
        w_a_linear_v = nn.Linear(dim, r, bias=False)
        w_b_linear_v = nn.Linear(r, dim, bias=False)
        w_As.append(w_a_linear_q)
        w_Bs.append(w_b_linear_q)
        w_As.append(w_a_linear_v)
        w_Bs.append(w_b_linear_v)
        layer.cross_attn.value_proj = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
        layer.cross_attn.output_proj = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)
    
    for w_A in w_As:
        nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
    for w_B in w_Bs:
        nn.init.zeros_(w_B.weight)
    model = model.cuda()
    return model

def bbox2mask(bbox, shape):
    """
    Convert a bounding box to a binary mask using PyTorch.

    Parameters:
    - bbox: a tensor of the form [[xmin, ymin, xmax, ymax]]
    - shape: torch.Size object (e.g., torch.Size([1, 1, 480, 910]))

    Returns:
    - mask: a tensor of shape [1, 1, 320, 160] with values set to 1 inside the bbox and 0 outside
    """
    height, width = shape[0], shape[1]

    # Convert un-normalized bbox to absolute coordinates
    bbox_abs = bbox.int()

    if len(bbox_abs.shape) != 2:
        bbox_abs = bbox_abs.unsqueeze(0)

    # Extract coordinates from (xmin, ymin, xmax, ymax) format
    xmin, ymin, xmax, ymax = bbox_abs[0][0].item(), bbox_abs[0][1].item(), bbox_abs[0][2].item(), bbox_abs[0][3].item()

    # Clip coordinates to be within image bounds
    xmin = max(0, min(width, xmin))
    xmax = max(0, min(width, xmax))
    ymin = max(0, min(height, ymin))
    ymax = max(0, min(height, ymax))

    # Create a meshgrid of shape (height, width)
    rows = torch.arange(0, height, dtype=torch.int64).unsqueeze(1)
    cols = torch.arange(0, width, dtype=torch.int64)

    # Create the mask
    mask = (rows >= ymin) & (rows < ymax) & (cols >= xmin) & (cols < xmax)

    # Adjust the mask shape to match the given shape (e.g., [1, 1, 320, 160])
    mask = mask.float().unsqueeze(0).unsqueeze(0)

    return mask

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

    # Note: resize for PIL, 0 means the nearest mode
    small_seg = np.array(Image.fromarray(seg_ori).resize((th//8,tw//8), 0)).astype(np.float32)

    t = []
    t.extend([transforms.ToTensor()])
    trans = transforms.Compose(t)
    pair = [small_seg, small_seg]
    transformed = list(trans(*pair))
    small_seg = transformed[0]
  
    return to_one_hot(small_seg)

def propagate_bbox(F_ref, F_tar, seg_ref_shape, bbox_ref):
    # F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
    # seg_ref = seg_ref.squeeze(0)
    F_ref, F_tar = squeeze_all(F_ref, F_tar)
    
    ## OK: turn bbox into mask and send into match_ref_tar
    seg_ref = bbox2mask(bbox_ref, seg_ref_shape)

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


def compute_iou(boxA, boxB):
    if len(boxA.shape) != 2:
        boxA = boxA.unsqueeze(0)
    if len(boxB.shape) != 2:
        boxB = boxB.unsqueeze(0)
    iou_matrix = torchvision.ops.box_iou(boxA.cuda(), boxB.cuda())
    # Since both boxA and boxB are single boxes, the IoU will be a 1x1 matrix. We extract the scalar value.
    iou_value = iou_matrix[0, 0].item()

    return iou_value


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
        print('\n**** USE SAM ****\n')

        # TODO: load the model from mmdet
        # Specify the path to model config and checkpoint file
        config_file = args.g_dino_config_path
        checkpoint_file = args.g_dino_ckpt_path
        
        # Build the model from a config file and a checkpoint file
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')
        inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda', show_progress=False)
        if args.use_gdino_LORA:
            inferencer.model = add_lora(inferencer.model)
            checkpoint = torch.load(args.g_dino_ckpt_path, map_location='cpu')
            inferencer.model.load_state_dict(checkpoint['state_dict'])
            inferencer.model.eval()
            print('Reload the ckpt for LORA')
        

        ## 2. Building SAM Model and SAM Predictor
        # device = torch.device('cuda:0')
        # sam_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        # sam_hq_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_hq_vit_h.pth'
        sam_hq_checkpoint = args.sam_ckpt_path
        lora_sam_ckpt_path = args.lora_sam_ckpt_path
        if args.use_LORA_SAM:
            sam = build_sam_hq(checkpoint=sam_hq_checkpoint, mask_threshold=args.mask_threshold)
            # use_sam_hq
            sam.cuda()
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
            sam.cuda()
            sam_predictor = SamPredictor(sam)

    # get palette
    palette_img = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette = Image.open(palette_img).getpalette()

    num_all_frames = 0

    font_path = "fonts/OpenSans-Regular.ttf"
    font = ImageFont.truetype(font_path, 30) # change the '30' to any size you want
    print(f'*** Num of anno. we run for DAVIS: {args.run_anno_id}***')

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
        for anno_id in range(args.run_anno_id):  # 4 annotators
            all_exps = []
            bbox_data = []
            anno_logits = []
            anno_masks = []  # [num_obj+1, video_len, h, w], +1 for background
            anno_boxes = []
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

                if args.semi_online:
                    num_clip_frames = args.num_frames
                else:
                    num_clip_frames = 1

                # TODO: memory contain at most 4 frames (bbox)
                frame_mem = []
                #  memory contain at most 4 frames (feats)
                feats_mem = []


                # 3. for each clip
                # track_res = model.generate_empty_tracks()
                for clip_id in range(0, video_len, num_clip_frames):
                    frames_ids = [x for x in range(video_len)]
                    clip_frames_ids = frames_ids[clip_id: clip_id + num_clip_frames]
                    clip_len = len(clip_frames_ids)

                    # load the clip images
                    # imgs = []
                    pred_masks = []
                    pred_logits = []
                    pred_boxes = []
                    for t in clip_frames_ids:
                        frame = frames[t]
                        img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                        # Use the cached image if it's already loaded, otherwise load and cache it
                        if img_path not in image_cache_for_video:
                            image_cache_for_video[img_path] = mmcv.imread(img_path)
                        img = image_cache_for_video[img_path].copy()
                        # origin_w, origin_h = img.size
                        # imgs.append(transform(img))  # list[Img]

                        ## TODO: maybe directly perform prediction here? 
                        # so we can avoid the batch inference issues..
                        with torch.no_grad(): #### REMEMBER ADD THIS when infernce
                            if args.use_SAM:
                                result = inferencer(img, texts=exp, frame_idx=t)
                                logits = torch.tensor(result['predictions'][0]['scores'])
                                boxes = torch.tensor(result['predictions'][0]['bboxes'])

                                # NOTE: intermediate feat for prop.
                                intermediate_feat = result['predictions'][0]['intermediate_feats']

                                if t > args.num_prop_bbox - 1:
                                    # prop.
                                    if len(frame_mem) > args.num_prop_bbox:
                                        frame_mem.pop(0)
                                        feats_mem.pop(0)


                                    if len(boxes) == 0:
                                        boxes_xyxy = torch.zeros((1, 4))
                                        ### DONE:
                                        # if this situation, no need SAM! (just all zeros)
                                        masks = torch.zeros(masks.shape).cuda()
                                        pred_masks.append(masks)
                                        pred_boxes.append(boxes_xyxy)
                                        pred_logits.append(torch.zeros((1,)))
                                    else:
                                        # TODO: prop 5 bboxes
                                        # prev_bboxes has shape (5, 4)
                                        # prev_intermediate_feat has shape (5, X, X, X)
                                        # intermediate_feat has shape (1, X, X, X)
                                        # out_boxes has shape (1, 4)
                                        out_bboxes_list = []
                                        # prev_bbox[i]
                                        for i in range(args.num_prop_bbox):
                                            out_boxes, coords = propagate_bbox(feats_mem[i],
                                                                intermediate_feat,
                                                                img.shape,
                                                                frame_mem[i])
                                            out_bboxes_list.append(out_boxes)


                                        # TODO: with mem. => affect the logits
                                        conf_idx_list = []
                                        for mem_box in out_bboxes_list:
                                            ious = [compute_iou(mem_box, bbox) for bbox in boxes]
                                            conf_scores = (1 - args.iou_alpha) * logits + args.iou_alpha * torch.tensor(ious)
                                            conf_score, conf_idx = torch.max(conf_scores, dim=0)
                                            conf_idx_list.append(conf_idx)
                                        
                                        # TODO: take max for current frame
                                        max_logit, max_idx = torch.max(logits, dim=0)
                                        conf_idx_list.append(max_idx)

                                        conf_idx_int_list = [val.item() for val in conf_idx_list]
                                        counter = Counter(conf_idx_int_list)
                                        vote_idx = counter.most_common(1)[0][0]

                                        boxes = boxes[vote_idx].unsqueeze(0).cpu() ## shape: (1, 4)

                                        
                                        # img_arr = np.asarray(img)
                                        sam_predictor.set_image(img, image_format='BGR')
                                        # # box: normalized box xywh -> unnormalized xyxy
                                        # H, W, _ = img.shape
                                        boxes_xyxy = boxes
                                    
                                        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, img.shape[:2]).cuda()
                                        # print(transformed_boxes)
                                        # print(transformed_boxes.shape)
                                        masks, _, _ = sam_predictor.predict_torch(
                                                    point_coords = None,
                                                    point_labels = None,
                                                    boxes = transformed_boxes,
                                                    multimask_output = False,
                                                )
                                        # print(f'{frame} {masks.shape}')
                                        # print(f'obj: {obj_id}, t: {t}, box shape: {masks.shape}')
                                        pred_masks.append(masks)
                                        pred_boxes.append(boxes_xyxy)
                                        pred_logits.append(max_logit.unsqueeze(0))

                                        frame_mem.append(boxes_xyxy)
                                        # TODO: save intermediate feat.
                                        feats_mem.append(intermediate_feat)
                                else:
                                    # framewise
                                    if len(boxes) == 0:
                                        boxes_xyxy = torch.zeros((1, 4))
                                        ### DONE:
                                        # if this situation, no need SAM! (just all zeros)
                                        masks = torch.zeros(masks.shape).cuda()
                                        pred_masks.append(masks)
                                        pred_boxes.append(boxes_xyxy)
                                        pred_logits.append(torch.zeros((1,)))
                                    else:
                                        max_logit, max_idx = torch.max(logits, dim=0)
                                        boxes = boxes[max_idx].unsqueeze(0) ## shape: (1, 4)
                                        # img_arr = np.asarray(img)
                                        sam_predictor.set_image(img, image_format='BGR')
                                        # # box: normalized box xywh -> unnormalized xyxy
                                        # H, W, _ = img.shape
                                        boxes_xyxy = boxes
                                    
                                        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, img.shape[:2]).cuda()
                                        # print(transformed_boxes)
                                        # print(transformed_boxes.shape)
                                        masks, _, _ = sam_predictor.predict_torch(
                                                    point_coords = None,
                                                    point_labels = None,
                                                    boxes = transformed_boxes,
                                                    multimask_output = False,
                                                )
                                        # print(f'{frame} {masks.shape}')
                                        # print(f'obj: {obj_id}, t: {t}, box shape: {masks.shape}')
                                        pred_masks.append(masks)
                                        pred_boxes.append(boxes_xyxy)
                                        pred_logits.append(max_logit.unsqueeze(0))
                                        
                                        frame_mem.append(boxes_xyxy)
                                        # TODO: save intermediate feat.
                                        feats_mem.append(intermediate_feat)


                               
                                
                    pred_masks = torch.cat(pred_masks, dim=0)  # [t, h, w],
                    pred_masks = pred_masks.squeeze(0) # so that it would not have 4-dim
                    pred_boxes = torch.cat(pred_boxes, dim=0)
                    all_pred_masks.append(pred_masks)
                    ### save boxes
                    pred_logits = torch.cat(pred_logits, dim=0)
                    all_pred_boxes.append(pred_boxes)
                    all_pred_logits.append(pred_logits)

                all_pred_logits = torch.cat(all_pred_logits, dim=0)  # (video_len, K)
                all_pred_masks = torch.cat(all_pred_masks, dim=0)  # (video_len, h, w)
                anno_logits.append(all_pred_logits)
            
                anno_masks.append(all_pred_masks)
                all_exps.append(exp)
                all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
                anno_boxes.append(all_pred_boxes)

                # handle a complete image (all objects of a annotator)
            anno_logits = torch.stack(anno_logits)  # [num_obj, video_len, k]
            anno_masks = torch.stack(anno_masks)  # [num_obj, video_len, h, w]
            t, h, w = anno_masks.shape[-3:]
            anno_masks[anno_masks < 0.5] = 0.0
            background = 0.1 * torch.ones(1, t, h, w).cuda()
            anno_masks = torch.cat([background, anno_masks], dim=0)  # [num_obj+1, video_len, h, w]
            out_masks = torch.argmax(anno_masks, dim=0)  # int, the value indicate which object, [video_len, h, w]
            out_masks = out_masks.detach().cpu().numpy().astype(np.uint8)  # [video_len, h, w]
         
         
            # save results
            anno_save_path = os.path.join(save_path_prefix, f"anno_{anno_id}", video)
            if not os.path.exists(anno_save_path):
                os.makedirs(anno_save_path)
            for f in range(out_masks.shape[0]):
                # print(out_masks[f].shape)
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
                        position = (10, 10)
                        # draw.text(position, text, (255, 0, 0), font=font)

                        
                        draw_boxes = anno_boxes[k - 1][f].unsqueeze(0)
                        # draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()
                        # print(draw_boxes)
                        # print(draw_boxes.shape)
                        
                        xmin, ymin, xmax, ymax = draw_boxes[0]

                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[k%len(color_list)]), width=2)
                        # plot the bbox score
                        plot_bbox_score = f'{anno_logits[k - 1][f].item():.2f}'
                        # draw.text((xmin, ymin), plot_bbox_score, (255, 0, 0), font=font)

                        mask_arr = np.array(img_E)
                        # print(mask_arr.shape)
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
    # result_dict[str(pid)] = num_all_frames
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
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser('OnlineRefer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)

