# Run examples:

### 1. Run in background even the connection is closed
<!-- export PYTHONPATH=./ ; python ./mains/exec_tmp_G03.py -->
export PYTHONPATH="./:./Graph_Sampling" ; python ./mains/exec_tmp_G03.py --config ./configs/GRPG/default.json



### 2. Run in background even the connection is closed
export CUDA_VISIBLE_DEVICES=7 ; nohup python -m experiments.LN_prompt --exp_name=ours_1b1f      --data_dir=./datasets/Sketchy --seed 42 --clip_LN_lr=0.00001 --two_distinct_sk_img_branchs=0 --train_visual_proj=0 --prompt_lr=0.0001  --use_quadruplet=1 --lambda_quadruplet_2=1   --lambda_quadruplet_3=1   --lambda_quadruplet_4=1   --is_photo_prompt_used=0 --freeze_top_m_sk_LN=7  >/dev/null 2>&1 
