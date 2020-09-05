#!/bin/bash
#
#SBATCH -J Prueba_Slurm_Yamil
#SBATCH -p main # Partition
#SBATCH -N 1
#SBATCH --mem 10G # Memory request of X GB
#SBATCH --array=0-0
#SBATCH -o  bin_outFile_%A_CONF_%a.out
#SBATCH -e  bin_errFile_%A_CONF_%a.err
#SBATCH --exclude=luthier02

#conda activate yessus.py3.6.geo

#El comando para activar el ambiente me lanza un error de que la linea
#de comandos requiere de un conda init, pero no fue necesario activar el
#ambiente asique no me preocupe


cd prueba
cd ems-ny-data
python runExperiments.py -i $SLURM_ARRAY_TASK_ID