# Example: bash run.sh /home/ubuntu/efs/cifar_logs baseline 0 1
LOG_BASE_DIR=$1
EXP_NAME=$2
TRAIN_GPU_INDEX=$3
EVAL_GPU_INDEX=$4

rm -rf ${LOG_BASE_DIR}/${EXP_NAME}
mkdir -p ${LOG_BASE_DIR}/${EXP_NAME}_train ${LOG_BASE_DIR}/${EXP_NAME}_eval

screen -dmS ${EXP_NAME}_train bash
screen -S ${EXP_NAME}_train -X stuff "CUDA_VISIBLE_DEVICES='${TRAIN_GPU_INDEX}' python cifar10_train.py --train_dir ${LOG_BASE_DIR}/${EXP_NAME}_train
"
screen -dmS ${EXP_NAME}_eval bash
screen -S ${EXP_NAME}_eval -X stuff "CUDA_VISIBLE_DEVICES='${EVAL_GPU_INDEX}' python cifar10_eval.py --eval_dir ${LOG_BASE_DIR}/${EXP_NAME}_eval --checkpoint_dir ${LOG_BASE_DIR}/${EXP_NAME}_train
"

