# Uses best/ under outputs_fullfinetune_sub if present
CUDA_VISIBLE_DEVICES=2 python /home/tmunna/project-glintt/new-code/test-per-class.py \
  --model_dir /home/tmunna/project-glintt/new-code/output-medialbertina-pt-pt-900m-realtest/best \
  --test_json /home/tmunna/project-glintt/final-data/new-split-test-real/test-real.json \
  --max_len 512 \
  --stride 128 \
  --score_mode joint \
  --pred_json polarity_eval_alldis-test-real-biobert-all.json