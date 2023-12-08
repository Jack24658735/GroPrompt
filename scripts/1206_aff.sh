CUDA_VISIBLE_DEVICES=0, bash ./scripts/online_davis_sam_mmdet_aff.sh "./outputs_1206_aff" ../Grounded-Segment-Anything/sam_hq_vit_h.pth \
    --g_dino_ckpt_path ./CVPR_tuned_weight/boxselect.pth \
    --g_dino_config_path "./mmdetection/configs/grounding_dino/ours_framewise.py" --run_anno_id 1 -s 480