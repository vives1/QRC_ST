#!/bin/bash
#SBATCH --job-name=QRC_runs      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vives@princeton.edu

module purge
module load anaconda3
conda activate QRC_qiskit

# MNIST size
new_size = 8
nr = 4

# wither True (0) or Flase (1)
isNoisy = 1
isTrain = 0

# manually input train/test size
nimg = 2000

img_iteration = 0


max_img=$(($nimg-1))

for img_number in $(seq 0 $max_img)
do
 python QRC_MNIST.py $new_size $img_number $nr $isNoisy $isTrain $img_iteration &
done

wait
