CUDA_VISIBLE_DEVICES=0 python danka.py \
  --model_name_or_path "/root/.cache/modelscope/hub/Qwen/Qwen2.5-3B" \
  --quantization_bit 8 \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --output_dir "./output" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 3 \
  --learning_rate 5e-4 \
  --warmup_steps 500 \
  --save_steps 500 \
  --logging_steps 50 \
  --fp16



####    --train_file "/root/data/ChID/processed/train.json" \
    #    --eval_file "/root/data/ChID/processed/dev.json" \
