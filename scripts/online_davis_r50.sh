OUTPUT_DIR=$1
CHECKPOINT=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

echo "Load model weights from: ${CHECKPOINT}"

# test using the model trained on ref-youtube-vos directly

python3 inference_davis_online.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT} --dataset_file davis \
--online ${PY_ARGS}

# evaluation
ANNO0_DIR=${OUTPUT_DIR}/"valid"/"anno_0"
ANNO1_DIR=${OUTPUT_DIR}/"valid"/"anno_1"
ANNO2_DIR=${OUTPUT_DIR}/"valid"/"anno_2"
ANNO3_DIR=${OUTPUT_DIR}/"valid"/"anno_3"
python3 eval_davis.py --results_path=${ANNO0_DIR}
python3 eval_davis.py --results_path=${ANNO1_DIR}
python3 eval_davis.py --results_path=${ANNO2_DIR}
python3 eval_davis.py --results_path=${ANNO3_DIR}

echo "Working path is: ${OUTPUT_DIR}"

# OUTPUT_DIR=work_dirs/online_v11_davis

# CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
# CHECKPOINT=/data/wudongming/referonline/output/online_v11/checkpoint.pth
# inference
# python3 inference_davis_online.py --with_box_refine --binary --freeze_text_encoder \
# --output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT} \


# # evaluation
# ANNO0_DIR=${OUTPUT_DIR}/"valid"/"anno_0"
# ANNO1_DIR=${OUTPUT_DIR}/"valid"/"anno_1"
# ANNO2_DIR=${OUTPUT_DIR}/"valid"/"anno_2"
# ANNO3_DIR=${OUTPUT_DIR}/"valid"/"anno_3"
# python3 eval_davis.py --results_path=${ANNO0_DIR}
# python3 eval_davis.py --results_path=${ANNO1_DIR}
# python3 eval_davis.py --results_path=${ANNO2_DIR}
# python3 eval_davis.py --results_path=${ANNO3_DIR}

# echo "Working path is: ${OUTPUT_DIR}"
