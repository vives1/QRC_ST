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
	nimg = 2000
	lab_div = 1000
else:
	tVar = "Test"
	nimg = 200
	lab_div = 100



img_iteration = int(sys.argv[6])

# label 
label_data = []

# training set 
rc_nodes = []

n_meas = new_size*int(new_size/4)

for img_number in range(nimg):
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


res = [label_data,rc_nodes]

sn = "QRC_cov_MNIST_{}x{}_{}{}_{}_nq{}_{}x{}_iter{}.txt".format(new_size,new_size,tVar,nimg,nVar,3*int(nr/2),nr,int(nc/2),img_iteration)


with open(sn, "w") as fp:
	json.dump(res, fp)

