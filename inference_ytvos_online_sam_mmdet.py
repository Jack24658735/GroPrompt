'''
Inference code for OnlineRefer, on Ref-Youtube-VOS
Modified from ReferFormer (https://github.com/wjn922/ReferFormer)
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
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

from sam_lora_image_encoder_mask_decoder import LoRA_Sam

import mmcv
from mmdet.apis import DetInferencer


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
    args.masks = True
    args.batch_size == 1
    print("Inference only supports for batch size = 1")

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
    root = Path(args.ytvos_path)  # data/ref-youtube-vos
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # for some reasons the competition's validation expressions dict contains both the validation (202) &
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    # create subprocess
    thread_num = args.ngpu
    # global result_dict
    # result_dict = mp.Manager().dict()

    processes = []
    # lock = threading.Lock()
    lock = mp.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    ### Note: workaround to avoid the multi-process issue...
    # sub_video_list = video_list[0:]
    # sub_processor(lock, 0, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, sub_video_list)
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

    if args.use_SAM:
        ## 1. Build Grounding dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        print('\n**** USE SAM ****\n')

        # load the model from mmdet
        # Specify the path to model config and checkpoint file
        config_file = args.g_dino_config_path
        checkpoint_file = args.g_dino_ckpt_path
        # Build the model from a config file and a checkpoint file
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')
        inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda', show_progress=False)

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

    num_all_frames = 0
    fps_time = 0.0
    fps_frames = 0.0
    
    font_path = "fonts/OpenSans-Regular.ttf"
    font = ImageFont.truetype(font_path, 30) # change the '30' to any size you want
    # 1. For each video
    # video_list = [video_list[video_list.index('0a598e18a8')]]
    
    for video in video_list:
        metas = []  # list[dict], length is number of expressions

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        # video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas
        image_cache_for_video = {}
        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            if args.semi_online:
                num_clip_frames = args.num_frames
            else:
                num_clip_frames = 1

            video_len = len(frames)
            clip_list = [frames[clip_i:clip_i + num_clip_frames] for clip_i in range(0, video_len, num_clip_frames)]
            all_pred_masks = []
            all_pred_boxes = []
            # track_res = model.generate_empty_tracks()
            # TODO: for SAM model
            #### NOT TEST
            for idx, clip in enumerate(clip_list):
                frames = clip
                clip_len = len(frames)

                # store images
                # imgs = []
                pred_masks = []
                pred_boxes = []
                pred_logits = []
                for t in range(clip_len):
                    frame = frames[t]
                    img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                    # Use the cached image if it's already loaded, otherwise load and cache it
                    if img_path not in image_cache_for_video:
                        image_cache_for_video[img_path] = mmcv.imread(img_path)
                    img = image_cache_for_video[img_path].copy()
                    origin_w, origin_h = img.shape[0], img.shape[1]
                    # imgs.append(transform(img))  # list[img]
                    with torch.no_grad(): #### REMEMBER ADD THIS when infernce
                        if args.use_SAM:
                            # TODO: perform detection with Grounding DINO
                            # detect objects
                            result = inferencer(img, texts=exp, frame_idx=idx)
                            logits = torch.tensor(result['predictions'][0]['scores'])
                            boxes = torch.tensor(result['predictions'][0]['bboxes'])

                            # Note!!! handle special cases for "india"
                            if len(boxes) == 0:
                                boxes_xyxy = torch.zeros((1, 4))
                                ### DONE:
                                # if this situation, no need SAM! (just all zeros)
                                masks = torch.zeros(img.shape[:2]).unsqueeze(0).unsqueeze(0).cuda()
                                # masks.shape: (1, h, w)
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
                                masks, iou_predictions, _ = sam_predictor.predict_torch(
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

                pred_masks = torch.cat(pred_masks, dim=0)  # [t, h, w],
                pred_boxes = torch.cat(pred_boxes, dim=0)
                pred_logits = torch.cat(pred_logits, dim=0)
                # all_pred_masks.append(pred_masks[0])
                # all_pred_boxes.append(pred_boxes)

                # pred_logits = outputs["pred_logits"][0]
                # pred_boxes = outputs["pred_boxes"][0]
                # pred_masks = outputs["pred_masks"][0]
                # pred_ref_points = outputs["reference_points"][0]
                # pred_init_points = outputs["reference_inits"][0]

                # according to pred_logits, select the query index
                pred_scores = pred_logits.sigmoid()  # [t, q, k]
                pred_scores = pred_scores.mean(0)  # [q, k]
                max_scores, _ = pred_scores.max(-1)  # [q,]
                _, max_ind = max_scores.max(-1)  # [1,]
                max_inds = max_ind.repeat(clip_len)
                pred_masks = pred_masks[range(clip_len), max_inds, ...]  # [t, h, w]
                pred_masks = pred_masks.unsqueeze(0)
                pred_masks = pred_masks.squeeze(0).detach().cpu().numpy()
                # pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()

                # store the video results
                # all_pred_logits = pred_logits[range(clip_len), max_inds]
                # all_pred_boxes = pred_boxes[range(clip_len), max_inds]
                all_pred_boxes = pred_boxes
                # all_pred_ref_points = pred_ref_points[range(clip_len), max_inds]
                    # all_pred_init_points = pred_init_points[range(clip_len)][0]
                all_pred_masks = pred_masks

                if args.visualize:
                    for t, frame in enumerate(frames):
                        # original
                        img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                        source_img = Image.open(img_path).convert('RGBA')  # PIL image

                        draw = ImageDraw.Draw(source_img)
                        draw_boxes = all_pred_boxes[t].unsqueeze(0)
                        # draw_boxes = all_pred_boxes[t]
                        # draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()

                        our_colors = np.array([[0, 255, 0], [255, 0, 0]]).astype('uint8').tolist()
                        # draw init point
                        # if args.draw_init_point:
                        #     draw_init_points(draw, all_pred_init_points, source_img.size, our_colors[-1], source_img)

                        # draw boxes, color = color_list[i % len(color_list)]
                        xmin, ymin, xmax, ymax = draw_boxes[0]
                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(our_colors[0]), width=2)

                        text = exp
                        position = (10, 10)
                        # draw.text(position, text, (255, 0, 0), font=font)

                        # draw inter reference point
                        # ref_points = all_pred_ref_points[t].unsqueeze(0).detach().cpu().tolist()
                        # draw_reference_points(draw, ref_points, source_img.size, color=our_colors[0])

                        # draw mask
                        source_img = vis_add_mask(source_img, all_pred_masks[t], our_colors[0])

                        # save
                        save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                        if not os.path.exists(save_visualize_path_dir):
                            os.makedirs(save_visualize_path_dir)
                        save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.jpg')
                        source_img.save(save_visualize_path)

                # save binary image
                save_path = os.path.join(save_path_prefix, video_name, exp_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for j in range(clip_len):
                    frame_name = frames[j]
                    mask = all_pred_masks[j].astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frame_name + ".png")
                    mask.save(save_file)

        with lock:
            progress.update(1)
    # print('*' * 20)
    # print('Frames:{}, Time:{}, FPS:{}'.format(fps_frames, fps_time, fps_frames/fps_time))
    # print('*' * 20)
    # print(fps_frames/frame_time)
    # result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


def draw_init_points(draw, all_pred_init_points, img_size, color, source_img):

    if all_pred_init_points.shape[-1] == 2:
        # the first frame has multiple reference points
        ref_points = all_pred_init_points[..., :2].detach().cpu().tolist()
        draw_reference_points(draw, ref_points, source_img.size, color=color)
    elif all_pred_init_points.shape[-1] == 4:
        # other frames have only one reference box
        draw_boxes = all_pred_init_points
        draw_boxes = rescale_bboxes(draw_boxes.detach(), img_size).tolist()
        for draw_box in draw_boxes:
            xmin, ymin, xmax, ymax = draw_box
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color),
                           width=2)

        # draw center point
        # ref_points = all_pred_init_points[..., :2].detach().cpu().tolist()
        # draw_reference_points(draw, ref_points, source_img.size, color=color)
    else:
        raise ValueError('Please specify the correct initial points.')


# visuaize functions
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
