#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0
stop_stage=0

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M-25Hz

# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=4
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi

  for model in llm; do
    OMP_NUM_THREADS=4 \
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      cosyvoice/bin/train_online_codec.py \
      --train_engine $train_engine \
      --config conf/cosyvoice.yaml \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
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