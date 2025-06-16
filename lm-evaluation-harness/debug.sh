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
                    lm_eval --model hf \
                        --model_args pretrained=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaMA-Adapter/Alpaca-7B-HF \
                        --tasks hellaswag \
						--num_fewshot 10 \
                        --device cuda:0 \
                        --batch_size 8