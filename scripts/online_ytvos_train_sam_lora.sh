OUTPUT_DIR=$1
SAM_CHECKPOINT=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

# training
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env \
main_tune_sam.py --dataset_file ytvos --online \
--masks --output_dir ${OUTPUT_DIR} --sam_ckpt_path ${SAM_CHECKPOINT} --binary --num_frames 1 ${PY_ARGS} \