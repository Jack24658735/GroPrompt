

OUTPUT_DIR=$1
# CHECKPOINT=$2
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this

# echo "Load model weights from: ${CHECKPOINT}"

# test using the model trained on ref-youtube-vos directly
CUDA_VISIBLE_DEVICES=0, python3 inference_davis_online_sam_prop_dino_bbox.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --dataset_file davis \
--online --use_SAM -s 480 --sam_ckpt_path "/home/liujack/RVOS/Grounded-Segment-Anything/sam_hq_vit_h.pth" ${PY_ARGS}

# # evaluation
ANNO0_DIR=${OUTPUT_DIR}/"valid"/"anno_0"
ANNO1_DIR=${OUTPUT_DIR}/"valid"/"anno_1"
ANNO2_DIR=${OUTPUT_DIR}/"valid"/"anno_2"
ANNO3_DIR=${OUTPUT_DIR}/"valid"/"anno_3"
python3 eval_davis.py --results_path=${ANNO0_DIR}
python3 eval_davis.py --results_path=${ANNO1_DIR}
python3 eval_davis.py --results_path=${ANNO2_DIR}
python3 eval_davis.py --results_path=${ANNO3_DIR}
