#!/bin/bash
#SBATCH --job-name=QRC_runs      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=16               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:45:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vives@princeton.edu

module purge
module load anaconda3
conda activate QRC_qiskit_1

# MNIST size
new_size=8
nr=2

# wither True (0) or Flase (1)
isNoisy=1
isTrain=0

# img_number=0
imgs=(138 307 308 524 596 607 639 1029 1332 1592 1640 1652 1692 1720)

img_iteration=0


for i in "${imgs[@]}"
do
 python QRC_MNIST.py $new_size $i $nr $isNoisy $isTrain $img_iteration &
done

wait
 
