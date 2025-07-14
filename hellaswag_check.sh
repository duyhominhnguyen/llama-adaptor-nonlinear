python hellaswag_check.py \
     --ckpt_dir ./alpaca_finetuning_v1/LLaMA-7B \
     --adapter_path ./example_weight/non_linear_prompt_7B.pth \
     --typ_act hypermodel \
     --hid_acti_func relu \
     --random_init False

