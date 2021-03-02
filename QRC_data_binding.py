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

if (isNoisy == 0):
	nVar = "Noisy"
else:
	nVar = "Noiseless"

if (isTrain == 0):
	tVar = "Train"
	nimg = 2000
else:
	tVar = "Test"
	nimg = 200


img_iteration = int(sys.argv[5])

# label 
label_data = []

# training set 
rc_nodes = []


for img_number in range(nimg):
	s = "QRC_MNIST_{}x{}_{}_{}_nq{}_{}x{}_img{}_iter{}.txt".format(new_size,new_size,tVar,nVar,2*nr,nr,nc,img_number,img_iteration)

	with open(s, "r") as fp:
		meas = json.load(fp)
	 
	label_data.append(int(img_number/1000))
	rc_nodes.append(meas)

	# remove file to declutter maybe not hahah
	# os.remove(s)


res = [label_data,rc_nodes]
sn = "QRC_MNIST_{}x{}_{}{}_{}_nq{}_{}x{}_iter{}.txt".format(new_size,new_size,tVar,nimg,nVar,2*nr,nr,nc,img_iteration)

print(res)

with open(s, "w") as fp:
    json.dump(res, fp)

