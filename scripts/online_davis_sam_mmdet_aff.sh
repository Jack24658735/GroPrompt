unset LD_LIBRARY_PATH

# GPUS='0,1'
GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=${GPUS}
# echo "using gpus ${GPUS}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

OUTPUT_DIR=$1
SAM_CHECKPOINT=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

# test using the model trained on ref-youtube-vos directly
python3 inference_davis_online_sam_mmdet_aff.py --binary  \
--output_dir=${OUTPUT_DIR} --dataset_file davis \
--online --use_SAM --sam_ckpt_path ${SAM_CHECKPOINT} --visualize --ngpu=${GPUS_PER_NODE} ${PY_ARGS}

# evaluation
ANNO0_DIR=${OUTPUT_DIR}/"valid"/"anno_0"
ANNO1_DIR=${OUTPUT_DIR}/"valid"/"anno_1"
ANNO2_DIR=${OUTPUT_DIR}/"valid"/"anno_2"
ANNO3_DIR=${OUTPUT_DIR}/"valid"/"anno_3"
python3 eval_davis.py --results_path=${ANNO0_DIR} --eval_bbox --eval_mask
python3 eval_davis.py --results_path=${ANNO1_DIR} --eval_bbox --eval_mask
python3 eval_davis.py --results_path=${ANNO2_DIR} --eval_bbox --eval_mask
python3 eval_davis.py --results_path=${ANNO3_DIR} --eval_bbox --eval_mask

python3 j_f_avg.py --anno0=${ANNO0_DIR} --anno1=${ANNO1_DIR} --anno2=${ANNO2_DIR} --anno3=${ANNO3_DIR} --result_path=${OUTPUT_DIR}

echo "Working path is: ${OUTPUT_DIR}"
