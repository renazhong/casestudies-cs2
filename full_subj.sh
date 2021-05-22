#!/bin/bash
#SBATCH --account=sta440-f20
#SBATCH -o full_subj.out
#SBATCH -p common
#SBATCH -N1
#SBATCH -c5
#SBATCH --mem-per-cpu=40G

module load R/3.6.0
date
Rscript full_subj_model.R
date
