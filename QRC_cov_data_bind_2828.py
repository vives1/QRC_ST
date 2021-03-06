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


new_size = 28
nr = 4
nc = int(new_size/nr)*new_size

nVar = "Noiseless"

tVar = "Train"

lab_div = 1000



img_iteration = 0

# label 
label_data = []

# training set 
rc_nodes = []

n_meas = new_size*int(new_size/4)

for img_number in range(2000,10000):
    s = "QRC_cov_MNIST_{}x{}_{}_{}_nq{}_{}x{}_img{}_iter{}.txt".format(new_size,new_size,tVar,nVar,3*int(nr/2),nr,int(nc/2),img_number,img_iteration)

    with open(s, "r") as fp:
        meas = json.load(fp)

    meas = np.array(meas)
    covm = meas.reshape(-1,n_meas)

    upper_tri = list(covm[np.triu_indices(n_meas)])
    

    label_data.append(int(img_number/lab_div))
    rc_nodes.append(upper_tri)

    # remove file to declutter maybe not hahah
    # os.remove(s)

print(len(rc_nodes))

# add to the 8000 the the first 2000
sn = "QRC_cov_MNIST_{}x{}_{}{}_{}_nq{}_{}x{}_iter{}.txt".format(new_size,new_size,tVar,2000,nVar,3*int(nr/2),nr,int(nc/2),img_iteration)

with open(sn, "r") as fp:
    res2000first = json.load(fp)

label_full = []
rc_full = []

label_full.extend(res2000first[0])
rc_full.extend(res2000first[1])

label_full.extend(label_data)
rc_full.extend(rc_nodes)


print(len(rc_full))


# save full 10k train images
res = [label_full,rc_full]

sn = "QRC_cov_MNIST_{}x{}_{}{}_{}_nq{}_{}x{}_iter{}.txt".format(new_size,new_size,tVar,10000,nVar,3*int(nr/2),nr,int(nc/2),img_iteration)

with open(sn, "w") as fp:
    json.dump(res, fp)

save_obj(res,sn)

