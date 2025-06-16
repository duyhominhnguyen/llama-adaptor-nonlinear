build_llama_image:
	clear
	srun -p A100-IML --ntasks 1 --gpus-per-task 1 --time=00:03:00 --immediate=1800 --cpus-per-gpu=64 --mem-per-cpu 4G --container-mounts=/netscratch/duynguyen:/netscratch/duynguyen --container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh --container-save=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh --pty /bin/bash
.PHONY: build_llama_image

inference:
	clear
	srun -p A100-IML -t 1-10:59:59 --ntasks 1 \
					--gpus-per-task 1 \
					--cpus-per-gpu=5 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=0" \
					--export="CUDA_VISIBLE_DEVICES=0,1" \
					torchrun --nnodes=1 --nproc_per_node=1 --master_port=25010 example.py \
						--typeQues truthqa \
						--typ_act hypermodel \
						--hid_acti_func relu \
						--random_init False \
						--max_seq_len 2048 \
						--ckpt_dir /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/LLaMA-7B\
						--tokenizer_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/LLaMA-7B/tokenizer.model \
						--adapter_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/alpaca_finetuning_v1/checkpoint_adapter_layer30_hypermodel64_random_initFalse_batchsize16_epoch5_7B_test/adapter_adapter_len10_layer30_epoch5.pth
.PHONY: inference

dump_inference:
	clear
	srun -p A100-IML -t 1-10:59:59 --ntasks 1 \
					--gpus-per-task 1 \
					--cpus-per-gpu=5 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=0" \
					--export="CUDA_VISIBLE_DEVICES=0,1" \
					torchrun --nnodes=1 --nproc_per_node=1 --master_port=25000 example.py \
						--typeQues none \
						--typ_act identity \
						--hid_acti_func none \
						--random_init False \
						--max_seq_len 2048 \
						--ckpt_dir /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/LLaMA-7B\
						--tokenizer_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/LLaMA-7B/tokenizer.model \
						--adapter_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/example_weight/llama_adapter_len10_layer30_release.pth
.PHONY: dump_inference
# /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/example_weight/llama_adapter_len10_layer30_release.pth
# /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/alpaca_finetuning_v1/checkpoint_identity_random_initFalse_test/adapter_adapter_len10_layer30_epoch5.pth


arc_inference:
	clear
	srun -p A100-IML -t 1-10:59:59 --ntasks 1 \
					--gpus-per-task 1 \
					--cpus-per-gpu=5 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=0" \
					--export="CUDA_VISIBLE_DEVICES=0,1" \
					torchrun --nnodes=1 --nproc_per_node=1 --master_port=25020 example.py \
						--typeQues arc \
						--typ_act hypermodel \
						--hid_acti_func relu \
						--random_init False \
						--max_seq_len 2048 \
						--ckpt_dir /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/LLaMA-7B\
						--tokenizer_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/LLaMA-7B/tokenizer.model \
						--adapter_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/alpaca_finetuning_v1/checkpoint_adapter_layer30_hypermodel64_random_initFalse_batchsize16_epoch5_7B_test/adapter_adapter_len10_layer30_epoch5.pth
.PHONY: arc_inference


extract_adapter:
	clear
	srun -p A100-IML -t 0-6:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=0" \
					--export="CUDA_VISIBLE_DEVICES=0,1" \
					python alpaca_finetuning_v1/extract_adapter_from_checkpoint.py \
					--folder /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/alpaca_finetuning_v1/checkpoint_adapter_layer_0.5_38_hypermodel64relu_random_initFalse_batchsize8_epoch5_13B_test
.PHONY: extract_adapter


combine_res:
	clear
	srun -p A100-IML -t 0-6:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=0" \
					--export="CUDA_VISIBLE_DEVICES=0,1" \
					python combine_res.py 
.PHONY: combine_res

remove_check:
	clear
	srun -p A100-IML -t 0-6:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llama_adapter.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=0" \
					--export="CUDA_VISIBLE_DEVICES=0,1" \
					python alpaca_finetuning_v1/remove_check.py 
.PHONY: remove_check