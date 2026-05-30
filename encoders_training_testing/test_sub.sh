# Uses best/ under outputs_fullfinetune_sub if present
CUDA_VISIBLE_DEVICES=2 python test-per-class.py \
  --model_dir  {model_dir}\
  --test_json test-real.json \
  --max_len 512 \
  --stride 128 \
  --score_mode joint \
  --pred_json polarity_eval_alldis-test-real-biobert-all.json