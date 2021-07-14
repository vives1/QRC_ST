#!/bin/bash                                                                                                                             
#SBATCH --job-name=QRC_runs      # create a short name for your job                                                                     
#SBATCH --nodes=1                # node count                                                                                           
#SBATCH --ntasks=1               # total number of tasks across all nodes                                                               
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)                                                      
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)                                                                  
#SBATCH --time=01:03:00          # total run time limit (HH:MM:SS)                                                                      
#SBATCH --mail-type=fail         # send email if job fails                                                                              
#SBATCH --mail-user=vives@princeton.edu                                                                                                 

export OPENBLAS_NUM_THREADS=1

module purge
module load anaconda3
conda activate QRC_qiskit_2


python QRC_MNIST_GATE_FIXED.py $1 $2 $3 $4 $5 $6 $7 
