#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH -J fdr_AandBkernels_datagen
#SBATCH --mail-user=jair.fead@icf.unam.mx
#SBATCH --mail-type=ALL
#SBATCH -A desi_g
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 32

# Se necesitan 32 cores para un nodo de gpu compartido
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load conda
conda activate cosmo

# Run
srun -n 1 -c 32 --cpu-bind=cores -G 1 --gpu-bind=single:1 \
    python /global/u2/j/jairf/chamba/AandBkernels/numerical/AandB.py