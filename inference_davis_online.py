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

# import supervision as sv
# import torchvision
# from groundingdino.util.inference import Model
# from segment_anything import sam_model_registry, SamPredictor
# from huggingface_hub import hf_hub_download
# from segment_anything import build_sam, SamPredictor

# Grounding DINO
# import GroundingDINO.groundingdino.datasets.transforms as T
# from GroundingDINO.groundingdino.util import box_ops
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

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



BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25

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
        sam_checkpoint = '/home/liujack/RVOS/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
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

        # get palette
        palette_img = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
        palette = Image.open(palette_img).getpalette()

        # start inference
        num_all_frames = 0
        model.eval()

   


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

                # 3. for each clip
                track_res = model.generate_empty_tracks()
                for clip_id in range(0, video_len, num_clip_frames):
                    frames_ids = [x for x in range(video_len)]
                    clip_frames_ids = frames_ids[clip_id: clip_id + num_clip_frames]
                    clip_len = len(clip_frames_ids)

                    # load the clip images
                    imgs = []
                    for t in clip_frames_ids:
                        frame = frames[t]
                        img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                        img = Image.open(img_path).convert('RGB')
                        origin_w, origin_h = img.size
                        imgs.append(transform(img))  # list[Img]

                        ## TODO: maybe directly perform prediction here? 
                        # so we can avoid the batch inference issues..
                        if args.use_SAM:
                            # TODO: perform detection with Grounding DINO
                            # detect objects
                            boxes, logits, phrases = predict(
                                model=grounding_dino_model, 
                                image=transform(img), 
                                caption=exp,
                                box_threshold=BOX_TRESHOLD, 
                                text_threshold=TEXT_TRESHOLD
                            )
                            # annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)
                            # annotated_frame = annotated_frame[...,::-1] # BGR to RGB

                            # TODO: perform seg. with SAM
                            img_arr = np.asarray(img)
                            sam_predictor.set_image(img_arr)
                            # box: normalized box xywh -> unnormalized xyxy
                            H, W, _ = img_arr.shape
                            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, img_arr.shape[:2]).to(device)
                            masks, _, _ = sam_predictor.predict_torch(
                                        point_coords = None,
                                        point_labels = None,
                                        boxes = transformed_boxes,
                                        multimask_output = False,
                                    )
                            print(masks[0][0].cpu().numpy())
                            print(masks[0][0].cpu().numpy().shape)
                            import pandas as pd
                            df = pd.DataFrame(masks[0][0].cpu().numpy())
                            df.to_csv(f'tmp.csv')
                            exit()
                            # Show masks is needed?
                            # annotated_frame_with_mask = show_mask(masks[0][0].cpu(), annotated_frame)

                        #### NOT DONE

                    imgs = torch.stack(imgs, dim=0).to(args.device)  # [video_len, 3, H, W]
                    img_h, img_w = imgs.shape[-2:]
                    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                    target = {"size": size}

                    with torch.no_grad():
                        outputs = model.inference([imgs], track_res, [exp], [target])
                    track_res = model.post_process_single_image(outputs, track_res, is_last=False)

                    pred_logits = outputs["pred_logits"][0]  # [t, q, k]
                    pred_masks = outputs["pred_masks"][0]  # [t, q, h, w]
                    # TODO: save pred box and plot also
                    # pred_boxes = outputs["pred_boxes"][0]


                    # according to pred_logits, select the query index
                    pred_scores = pred_logits.sigmoid()  # [t, q, k]
                    pred_scores = pred_scores.mean(0)  # [q, K]
                    max_scores, _ = pred_scores.max(-1)  # [q,]
                    _, max_ind = max_scores.max(-1)  # [1,]
                    max_inds = max_ind.repeat(clip_len)
                    pred_masks = pred_masks[range(clip_len), max_inds, ...]  # [t, h, w]
                    pred_masks = pred_masks.unsqueeze(0)

                    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear',
                                               align_corners=False)
                    pred_masks = pred_masks.sigmoid()[0]  # [t, h, w], NOTE: here mask is score
                    # store the clip results
                    pred_logits = pred_logits[range(clip_len), max_inds]  # [t, k]
                    all_pred_logits.append(pred_logits)
                    all_pred_masks.append(pred_masks)

                    
                    # TODO: save pred box and plot
                    # pred_boxes = pred_boxes[range(clip_len), max_inds]
                    # all_pred_boxes.append(pred_boxes)

                all_pred_logits = torch.cat(all_pred_logits, dim=0)  # (video_len, K)
                all_pred_masks = torch.cat(all_pred_masks, dim=0)  # (video_len, h, w)
                anno_logits.append(all_pred_logits)
                anno_masks.append(all_pred_masks)
                all_exps.append(exp)
                # all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
                # anno_boxes.append(all_pred_boxes)

                # handle a complete image (all objects of a annotator)
            anno_logits = torch.stack(anno_logits)  # [num_obj, video_len, k]
            anno_masks = torch.stack(anno_masks)  # [num_obj, video_len, h, w]
            t, h, w = anno_masks.shape[-3:]
            anno_masks[anno_masks < 0.5] = 0.0
            background = 0.1 * torch.ones(1, t, h, w).to(args.device)
            anno_masks = torch.cat([background, anno_masks], dim=0)  # [num_obj+1, video_len, h, w]
            out_masks = torch.argmax(anno_masks, dim=0)  # int, the value indicate which object, [video_len, h, w]
            # print('origin anno')
            # print(anno_masks[:, 1, :, :])
            # print('after anno')
            # print(out_masks[:, :, :])
            # exit()
            out_masks = out_masks.detach().cpu().numpy().astype(np.uint8)  # [video_len, h, w]
            # anno_masks = anno_masks.detach().cpu().numpy().astype(np.uint8)
            # anno_boxes = torch.stack(anno_boxes)
            # print(all_exps)
            # print(len(all_exps))
            # exit()
            ### TODO: visualize implementation
            # if args.visualize:
            #     for t, frame in enumerate(frames):
            #         img_path = os.path.join(img_folder, video_name, frame + ".jpg")
            #         source_img = Image.open(img_path).convert('RGB')
                    
            #         # boxes do not exist in this implementation?
            #         draw = ImageDraw.Draw(source_img)
            #         # draw_boxes = anno_boxes[t].unsqueeze(0)
            #         # draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()

            #         # our_colors = np.array([[0, 255, 0], [255, 0, 0]]).astype('uint8').tolist()
            #         # xmin, ymin, xmax, ymax = draw_boxes[0]
            #         # draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(our_colors[0]), width=2)

            #         # draw text
            #         ## Note: str(i) is the corresponding text annotation!
            #         text = expressions[expression_list[i]]["exp"]
            #         font_path = "/home/liujack/RVOS/OnlineRefer/fonts/OpenSans-Regular.ttf"
            #         font = ImageFont.truetype(font_path, 30) # change the '30' to any size you want
            #         position = (10, 10)
            #         draw.text(position, text, (0, 0, 0), font=font)
                    
            #         # draw mask
            #         source_img = vis_add_mask(source_img, out_masks[t], color_list[i%len(color_list)])
            #         # save 
            #         save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
            #         if not os.path.exists(save_visualize_path_dir):
            #             os.makedirs(save_visualize_path_dir)
            #         save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
            #         source_img.save(save_visualize_path)
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
                        draw = ImageDraw.Draw(source_img)
                        # text = expressions[expression_list[i]]["exp"]
                        # Example: bike-packing => ['a black bike', 'a man wearing a cap']
                        # print(all_exps)
                        # print(len(all_exps))
                        text = all_exps[exp_idx]
                        font_path = "/home/liujack/RVOS/OnlineRefer/fonts/OpenSans-Regular.ttf"
                        font = ImageFont.truetype(font_path, 30) # change the '30' to any size you want
                        position = (10, 10)
                        draw.text(position, text, (255, 0, 0), font=font)
                        
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

