python check.py \
     --ckpt_dir ./LLaMA-7B \
     --adapter_path ./alpaca_finetuning_v1/checkpoint_adapter_layer30_hypermodel64_random_initFalse_batchsize16_epoch5_7B_test/adapter_adapter_len10_layer30_epoch5.pth \
     --typ_act hypermodel \
     --hid_acti_func relu \
     --random_init False

