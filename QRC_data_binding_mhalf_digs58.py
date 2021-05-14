# Bind individual files into train and test set

import numpy as np
import scipy as sp

import json
import sys
import os


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
    range1 = np.arange(4000,5000)
    range2 = np.arange(7000,8000)
    label1 = (np.ones(1000)*5).tolist()
    label2 = (np.ones(1000)*8).tolist()
else:
    tVar = "Test"
    range1 = np.arange(400,500)
    range2 = np.arange(700,800)
    label1 = (np.ones(100)*5).tolist()
    label2 = (np.ones(100)*8).tolist()


ranges=[range1,range2]

label1.extend(label2)
label_data = label1


# Do a file for Zs and a file for covs

img_iteration = int(sys.argv[6])


# training set 
rc_nodes_zs = []

rc_nodes_covs = []


n_meas = new_size*int(new_size/2)

for rangee in ranges:

    for img_number in rangee:
        s = "QRC_zs_CasabZZs_MNIST_{}x{}_{}_{}_nq{}_{}x{}_img{}_iter{}.txt".format(new_size,new_size,tVar,nVar,3*int(nr/2),nr,nc,img_number,img_iteration)

        with open(s, "r") as fp:
            meas = json.load(fp)

        Zs = np.array(meas)

        Zs_mean = np.mean(Zs,axis=1)
        Zs_mean = Zs_mean.tolist()


        rc_nodes_zs.append(Zs_mean)


        covm = np.cov(Zs)
        upper_tri = list(covm[np.triu_indices(n_meas)])

        rc_nodes_covs.append(upper_tri)

        # remove file to declutter maybe not hahah
        # os.remove(s)


res_zs = [label_data,rc_nodes_zs]

res_covs = [label_data,rc_nodes_covs]


sn_zs = "QRC_zs_CasabZZs_MNIST_{}x{}_{}_digs58_{}_nq{}_{}x{}_iter{}.txt".format(new_size,new_size,tVar,nVar,3*int(nr/2),nr,nc,img_iteration)
sn_covs = "QRC_covs_CasabZZs_MNIST_{}x{}_{}_digs58_{}_nq{}_{}x{}_iter{}.txt".format(new_size,new_size,tVar,nVar,3*int(nr/2),nr,nc,img_iteration)


with open(sn_zs, "w") as fp:
    json.dump(res_zs, fp)

with open(sn_covs, "w") as fp:
    json.dump(res_covs, fp)

