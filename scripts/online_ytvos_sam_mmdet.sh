
OUTPUT_DIR=$1
SAM_CHECKPOINT=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

# echo "Load model weights from: ${CHECKPOINT}"
# inference
# CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
python3 inference_ytvos_online_sam_mmdet.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --online --use_SAM --sam_ckpt_path ${SAM_CHECKPOINT} --visualize ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"