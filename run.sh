# Example: bash run.sh /home/ubuntu/efs/cifar_logs exp_name 4 "0,1,2,3"

set -e

LOG_BASE_DIR=$1
EXP_NAME=$2
NUM_GPUS=$3
GPUS=$4

rm -rf ${LOG_BASE_DIR}/${EXP_NAME}
mkdir -p ${LOG_BASE_DIR}/${EXP_NAME}_train ${LOG_BASE_DIR}/${EXP_NAME}_eval

# Train
screen -dmS ${EXP_NAME}_train bash
screen -S ${EXP_NAME}_train -X stuff "CUDA_VISIBLE_DEVICES=${GPUS} mpirun -np ${NUM_GPUS} -H localhost:${NUM_GPUS} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl \^openib python cifar10_train.py --train_dir ${LOG_BASE_DIR}/${EXP_NAME}_train
"

# Eval
#screen -dmS ${EXP_NAME}_eval bash
#screen -S ${EXP_NAME}_eval -X stuff "CUDA_VISIBLE_DEVICES='' python cifar10_eval.py --eval_dir ${LOG_BASE_DIR}/${EXP_NAME}_eval --checkpoint_dir ${LOG_BASE_DIR}/${EXP_NAME}_train
#"

