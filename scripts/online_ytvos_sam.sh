#OUTPUT_DIR=output/online_swinl_f2_0.4_sampler1_lrdrop
# OUTPUT_DIR=work_dirs/online_ytvos_swinl

# training
# python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 --use_env \
# main.py --with_box_refine --binary --freeze_text_encoder \
# --epochs 6 --lr_drop 3 5 \
# --lr=1e-5 \
# --lr_backbone=5e-6 \
# --num_frames=2 \
# --sampler_steps 4  \
# --sampler_lengths 2 3 \
# --sampler_interval=5 \
# --pretrained_weights=pretrained_weights/swin_large_pretrained.pth \
# --output_dir=${OUTPUT_DIR} \
# --online \
# --backbone swin_l_p4w7 \
# --use_checkpoint_for_more_frames \

OUTPUT_DIR=$1
SAM_CHECKPOINT=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

# echo "Load model weights from: ${CHECKPOINT}"
# inference
# CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
python3 inference_ytvos_online_sam.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --online --use_SAM --sam_ckpt_path ${SAM_CHECKPOINT} ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"
