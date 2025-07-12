typ_act=identity
typ_gate=random
hid_acti_func=none
hidden_dim=0

batch_size=16
epoch=5
name=7B
adapter_layer=30

torchrun --nnodes=1 --nproc_per_node=3 --master_port=25012 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path ./LLaMA-${name} \
    --data_path ../alpaca_data.json \
    --adapter_layer ${adapter_layer} \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size ${batch_size} \
    --epochs ${epoch} \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --typ_act ${typ_act} \
    --typ_gate ${typ_gate} \
    --hid_acti_func ${hid_acti_func}\
    --hidden_dim ${hidden_dim}\
    --output_dir ./checkpoint_adapter_layer${adapter_layer}_${typ_act}${hidden_dim}${hid_acti_func}_batchsize${batch_size}_epoch${epoch}_${name}_test/
