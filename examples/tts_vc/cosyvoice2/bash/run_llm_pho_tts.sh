#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0
stop_stage=0

# train llm_pho
export CUDA_VISIBLE_DEVICES="4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=1
prefetch=100
train_engine=torch_ddp
exp_name=llm_pho_31w1_tts2
exp_conf=cosyvoice_pho_tts
portnum=2101
pretrained_model_dir=exp/$exp_name

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi

# --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
run_command() {
  for model in $exp_name; do
    OMP_NUM_THREADS=4 \
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --master_port $portnum   \
      cosyvoice/bin/train_phoneme_online_codec.py \
      --timeout  60    \
      --train_engine $train_engine \
      --config conf/$exp_conf.yaml \
      --model llm \
      --checkpoint $pretrained_model_dir \
      --model_dir `pwd`/exp/$model \
      --tensorboard_dir `pwd`/tensorboard/$model \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory  \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
}

# 循环运行
while true; do
    echo "Starting the command..."
    run_command
    if [ $? -eq 0 ]; then
        echo "Command executed successfully."
        break
    else
        echo "An error occurred. Retrying..."
        sleep 5  # 等待5秒后重试
    fi
done

fi


# inference
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  for mode in sft zero_shot; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 0 \
      --config conf/cosyvoice.yaml \
      --prompt_data data/test/parquet/data.list \
      --prompt_utt2data data/test/parquet/utt2data.list \
      --tts_text `pwd`/tts_text.json \
      --llm_model $pretrained_model_dir/llm.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir `pwd`/exp/cosyvoice/test/$mode
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi