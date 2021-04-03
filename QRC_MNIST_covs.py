
# Marti Vives
# QRC MNIST with qubit/temporal domain resizing


import qiskit
from qiskit import *
from qiskit import IBMQ


from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

# Import measurement calibration functions
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
												 CompleteMeasFitter, TensoredMeasFitter)

from qiskit.providers.aer.noise import NoiseModel

from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator

import cv2 

import numpy as np
import scipy as sp
from scipy.special import expit, softmax #logistic fns 
from sklearn.metrics import confusion_matrix

import scipy.optimize as optimize
from sklearn import linear_model

import itertools
import json
import sys
import pickle



#IBMQ.save_account('488a01d1ced1b582bb3182aa638da3e3571b07af5f757f78ad1d05a9f0dfec77bbbf44eb3563259f5238ff2b0fadd3963202ab72b45a65671caa9c47b492882f')
 

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)    
    

# Function that converts a 1D vectorized image into a (nr x nc) 2D array
def unpackcw(x,nr,nc):
	A = x.reshape(nc,nr)
	return A.T
	
# resize img 2D array to specified nr,nc
def resize_img(img,nr,nc):
	return cv2.resize(img, dsize=(nr, nc), interpolation=cv2.INTER_CUBIC)

def resize_dataset(data, o_size, nrc):
	
	new_data = np.zeros((nrc*nrc, len(data[0])))
	for i in range(len(data[0])):
		img = unpackcw(data[:,i],o_size,o_size)
		img = resize_img(img,nrc,nrc)
		img = img.T
		img = img.reshape(nrc*nrc)
		new_data[:,i] = img
	return new_data

 
def reshape_img_MNIST(data,size,nr,nc):
	
	
	reshaped_data = np.zeros((nr,nc))
	
	div = int(size/nr)
	for j in range(div):
		for i in range(size):  
			reshaped_data[:,j*size+i] = data[j*nr:(j+1)*nr,i]
			
	return reshaped_data
	
def runQRC_any(data,shots,noise_m,isNoisy=False):
	
	# data is 2D image, nrxnc
	
	nr = data.shape[0]
	nc = data.shape[1]

	qr = QuantumRegister(2*nr)
	cr = ClassicalRegister(nr*nc)

	circuit = QuantumCircuit(qr, cr) 

	for i in range(nc):
		v = data[:,i]       
				
		# U
		
		
		# RZ's
		# first nr rz
		for j in range(nr):
			phi = np.pi-v[j]
			circuit.rz(2*phi,qr[j])
			
		# last nr rz
		for j in range(nr-1):
			phi = (np.pi-v[j])*(np.pi-v[j+1])
			circuit.rz(2*phi,qr[j+nr])
		
		phi_20 = (np.pi-np.sum(v))
		circuit.rz(2*phi_20,qr[2*nr-1])
		
		
		# ZZ's, pairs containing a measurement qubit have phi = pi/2
		for j in range(2*nr):
			for k in range(j,2*nr):
				phi = np.pi/2
				if (j != k):
					if (j < nr and k < nr):
						if (j < k):
							phi = v[j]*(np.pi-v[k])
						else:
							phi = (np.pi-v[j]+v[k])
					elif (j < nr and k >= nr): #try something with meas qubits
						phi = v[j]*(np.pi-v[k-nr]+v[j])
						
					# ZZ gate
					if (k != 2*nr-1):
						circuit.cx(qr[j], qr[k])
						circuit.rz(2*phi, qr[k])
						circuit.cx(qr[j], qr[k])
					else:                           
						circuit.cx(qr[k], qr[j])
						circuit.rz(2*phi, qr[j])
						circuit.cx(qr[k], qr[j])                        
		
		
		# H's
		for j in range(2*nr):
			circuit.h(qr[j])
			
			  
		# U
		
		
		# RZ's
		# first nr rz
		for j in range(nr):
			phi = np.pi-v[j]
			circuit.rz(2*phi,qr[j])
			
		# last nr rz
		for j in range(nr-1):
			phi = (np.pi-v[j])*(np.pi-v[j+1])
			circuit.rz(2*phi,qr[j+nr])
		
		phi_20 = (np.pi-np.sum(v))
		circuit.rz(2*phi_20,qr[2*nr-1])
		
		
		# ZZ's, pairs containing a measurement qubit have phi = pi/2
		for j in range(2*nr):
			for k in range(j,2*nr):
				phi = np.pi/2
				if (j != k):
					if (j < nr and k < nr):
						if (j < k):
							phi = v[j]*(np.pi-v[k])
						else:
							phi = (np.pi-v[j]+v[k])
					elif (j < nr and k >= nr): #try something with meas qubits
						phi = v[j]*(np.pi-v[k-nr]+v[j])
						
					# ZZ gate
					if (k != 2*nr-1):
						circuit.cx(qr[j], qr[k])
						circuit.rz(2*phi, qr[k])
						circuit.cx(qr[j], qr[k])
					else:                           
						circuit.cx(qr[k], qr[j])
						circuit.rz(2*phi, qr[j])
						circuit.cx(qr[k], qr[j])                        
		
		
		# H's
		for j in range(2*nr):
			circuit.h(qr[j])
			
			
		# Measure
		for j in range(nr):
			circuit.measure(qr[j+nr],cr[j+i*nr])
			
#         print("col "+str(i)+" done")
			
	
	if (isNoisy == True):

		# if applying a certain noise model
		if (noise_m[0] == 'D' or noise_m[0] == 'N'):

			res = load_obj(noise_m)

			noise_model = NoiseModel.from_dict(res)
			basis_gates = noise_model.basis_gates

			result = execute(circuit, Aer.get_backend('qasm_simulator'),
                 basis_gates=basis_gates,
                 noise_model=noise_model).result()
			counts = result.get_counts(0)


		# if device noise model
		else:
		    # noisy simulation
		    res = load_obj(noise_m)
		    coupling_map=res[0]
		    basis_gates=res[1]
		    noise_model = NoiseModel.from_dict(res[2])

		    # Noisy simulation
		    result = execute(circuit, Aer.get_backend('qasm_simulator'),
		                 coupling_map=coupling_map,
		                 basis_gates=basis_gates,
		                 noise_model=noise_model).result()
		    
		    counts = result.get_counts(0)
	else:
		# Use Aer's qasm_simulator
		backend = Aer.get_backend('qasm_simulator')
		job = execute(circuit, backend, shots=shots)
		result = job.result()
		counts = result.get_counts(circuit)
	
	return counts


train_data = np.load('MNISTcwtrain1000.npy')
train_data = train_data.astype(dtype='float64')
test_data = np.load('MNISTcwtest100.npy')
test_data = test_data.astype(dtype='float64')

# resize to desired size
o_size = 28
new_size = int(sys.argv[1])

new_train_data = resize_dataset(train_data, o_size, new_size)
new_test_data = resize_dataset(test_data, o_size, new_size)

# restrict to 0 and 1
train = new_train_data[:,:2000]
test = new_test_data[:,:200]

# normalize grayscale
train = train/255.0
test = test/255.0


img_number = int(sys.argv[2])
nr = int(sys.argv[3])
nc = int(new_size/nr)*new_size

# wither True (0) or Flase (1)
IntisNoisy = int(sys.argv[4])
isTrain = int(sys.argv[5])

noise_m = str(sys.argv[6])

# for storing information purposes
img_iteration = int(sys.argv[7])


if (IntisNoisy == 0):
	nVar = noise_m
	isNoisy = True
else:
	nVar = "Noiseless"
	isNoisy = False


if (isTrain == 0):
	tVar = "Train"
	img = unpackcw(train[:,img_number],new_size,new_size)
else:
	tVar = "Test"
	img = unpackcw(test[:,img_number],new_size,new_size)



new_img = reshape_img_MNIST(img,new_size,nr,nc)

# always this
shots = 1024
n_meas = new_size*new_size

# have to build each count individually
Zs = []
for i in range(n_meas):
    Zs.append([])

 
for i in range(shots):
    counts = runQRC_any(new_img,1,noise_m,isNoisy=isNoisy)

    for key in counts:
        for r in range(n_meas):
            Zs[n_meas-r-1].append(int(key[r])*counts[key]) 


s = "QRC_cov_MNIST_{}x{}_{}_{}_nq{}_{}x{}_img{}_iter{}.txt".format(new_size,new_size,tVar,nVar,2*nr,nr,nc,img_number,img_iteration)

# use append "a" for parallel computing
with open(s, "w") as fp:
	json.dump(Zs, fp)




