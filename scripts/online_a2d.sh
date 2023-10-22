OUTPUT_DIR=$1
SAM_CHECKPOINT=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

# training
export NCCL_ASYNC_ERROR_HANDLING=1
python3 inference_a2d_mmdet.py --dataset_file a2d --online \
--masks --output_dir ${OUTPUT_DIR} --binary --num_frames 1  --use_SAM --sam_ckpt_path ${SAM_CHECKPOINT} ${PY_ARGS} \