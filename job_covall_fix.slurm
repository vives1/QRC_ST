#!/bin/bash
#SBATCH --job-name=QRC_runs      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G        # memory per cpu-core (4G is default)
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vives@princeton.edu

export OPENBLAS_NUM_THREADS=1

module purge
module load anaconda3
conda activate QRC_qiskit_2

# MNIST size
new_size=28
nr=4

# either True (0) or Flase (1)
isNoisy=1
isTrain=0

#noise_model='Depol_005_05'
noise_model='no_noise'

# manually input train/test size
#nimg=1000

img_iteration=0

min_img=2000
max_img=9999
#max_img=$(($nimg-1))

for img_number in $(seq $min_img $max_img)
do
 python QRC_MNIST_mhalf_pairE_covs_GOOD.py $new_size $img_number $nr $isNoisy $isTrain $noise_model $img_iteration &
done

wait
