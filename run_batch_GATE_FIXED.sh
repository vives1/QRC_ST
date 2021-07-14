#!/bin/bash                                                                                   

# MNIST size                                                                                                                            
new_size=8
nr=8

# either True (0) or Flase (1)                                                                                                          
isNoisy=1
isTrain=0

#noise_model='Depol_005_05'                                                                                                             
noise_model='no_noise'
#noise_model='Noise_thermal_8q'                                                                                                         
                                                                                                                             
img_iteration=0

min_img=5000
max_img=5999

for i in $(seq $min $max)
do
 sbatch job_single_pm1.slurm $new_size $i $nr $isNoisy $isTrain $noise_model $img_iteration 
done










