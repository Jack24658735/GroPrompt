OUTPUT_DIR=$1
# SAM_CHECKPOINT=$2
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this

# training
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env \
main_tune_dino.py --dataset_file ytvos --online \
--masks --output_dir ${OUTPUT_DIR} --binary --num_frames 1 ${PY_ARGS} \