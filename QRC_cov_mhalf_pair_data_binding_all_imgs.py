# Bind individual files into train and test set

import numpy as np
import scipy as sp

import json
import sys
import os
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


new_size = int(sys.argv[1])
nr = int(sys.argv[2])
nc = int(new_size/nr)*new_size

# whether True (0) or Flase (1)
isNoisy = int(sys.argv[3])
isTrain = int(sys.argv[4])

noise_m = str(sys.argv[5])

if (isNoisy == 0):
    nVar = noise_m
else:
    nVar = "Noiseless"

if (isTrain == 0):
    tVar = "Train"
    nimg = 10000
    lab_div = 1000
else:
    tVar = "Test"
    nimg = 1000
    lab_div = 100



img_iteration = int(sys.argv[6])

# label 
label_data = []

# training set 
rc_nodes = []

n_meas = new_size*int(new_size/4)

for img_number in range(nimg):
    s = "QRC_zs_true_MNIST_{}x{}_{}_{}_nq{}_{}x{}_img{}_iter{}.txt".format(new_size,new_size,tVar,nVar,3*int(nr/2),nr,int(nc/2),img_number,img_iteration)

    with open(s, "r") as fp:
        meas = json.load(fp)


    Zs = np.array(meas)

    Zs_mean = np.mean(Zs,axis=1)
    Zs_mean = Zs_mean.tolist()

    
    covm = np.cov(Zs)

    upper_tri = list(covm[np.triu_indices(n_meas)])

    # ADD <Z> VALUES TO COV MAT
    upper_tri.extend(Zs_mean)

    label_data.append(int(img_number/lab_div))
    rc_nodes.append(upper_tri)

    # remove file to declutter maybe not hahah
    # os.remove(s)


res = [label_data,rc_nodes]

sn = "QRC_cov+zs_true_MNIST_{}x{}_{}{}_{}_nq{}_{}x{}_iter{}.txt".format(new_size,new_size,tVar,nimg,nVar,3*int(nr/2),nr,int(nc/2),img_iteration)

save_obj(res, sn)



