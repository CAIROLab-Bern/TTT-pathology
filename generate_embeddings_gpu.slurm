#!/bin/bash
#SBATCH --job-name=ttt_embeddings_hoptimus1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090
#SBATCH --qos=job_gratis
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=logs/embeddings_%j.log
#SBATCH --error=logs/embeddings_%j.err

source .venv/bin/activate
# Run embedding generation
python generate_embeddings.py \
    --model hoptimus1 \
    --datadir data/PathoROB/tolkach_esca \
    --outputfolder data/PathoROB/precomputed_embeddings_tolkach/embs_hoptimus1

echo "Job completed successfully!"
