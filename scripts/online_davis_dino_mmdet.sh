

OUTPUT_DIR=$1
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this

# test using the model trained on ref-youtube-vos directly
python3 inference_davis_online_dino_mmdet.py --binary  \
--output_dir=${OUTPUT_DIR} --dataset_file davis \
--online --visualize ${PY_ARGS}

# evaluation
ANNO0_DIR=${OUTPUT_DIR}/"valid"/"anno_0"
ANNO1_DIR=${OUTPUT_DIR}/"valid"/"anno_1"
ANNO2_DIR=${OUTPUT_DIR}/"valid"/"anno_2"
ANNO3_DIR=${OUTPUT_DIR}/"valid"/"anno_3"
python3 eval_davis.py --results_path=${ANNO0_DIR} --eval_bbox
python3 eval_davis.py --results_path=${ANNO1_DIR} --eval_bbox
python3 eval_davis.py --results_path=${ANNO2_DIR} --eval_bbox
python3 eval_davis.py --results_path=${ANNO3_DIR} --eval_bbox

# echo "Working path is: ${OUTPUT_DIR}"