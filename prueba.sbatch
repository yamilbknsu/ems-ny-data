#!/bin/bash
#
#SBATCH -J Prueba_Slurm_Yamil # A single job name for the array
#SBATCH -p main # Partition
#--SBATCH-- -N 1 # on one node
#SBATCH --mem 5G # Memory request of X GB
#SBATCH --array=0-0
#SBATCH -o  bin_outFile_%A_CONF_%a.out
#SBATCH -e  bin_errFile_%A_CONF_%a.err
#SBATCH --exclude=luthier01
python3.6 prueba.py $SLURM_ARRAY_JOB_ID