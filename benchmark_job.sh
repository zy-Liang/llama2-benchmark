#!/bin/bash
#SBATCH --account=dinov99
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=100g

module purge
module load gcc cuda/11.7.1 cudnn/11.7-v8.7.0 python3.9-anaconda
source /nfs/turbo/umms-dinov/LLaMA/2.0.0/bin/activate

# 7B model
torchrun --nproc_per_node 1 /home/zyliang/llama-test/llama/benchmark.py \
    --ckpt_dir /nfs/turbo/umms-dinov/LLaMA/2.0.0/llama/modeltoken/llama-2-7b \
    --tokenizer_path /nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/tokenizer.model
