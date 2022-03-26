import numpy as np
import matplotlib.image as mpimg
from os import walk
import cv2
import scipy.signal
import cPickle as pickle
import gzip
import time

###############################

mypath = '/home/ram/DeepFuse/data/3_Images/patches/training/input'
mypath_test_in = '/home/lokesh/ram/DeepFuse/iccv/DeepFuse/input_test_pairs/'

###############################
f = []
for (dirpath, dirnames, filenames) in walk(mypath): 
	f.extend(filenames) 

arr = np.arange(len(f))
np.random.shuffle(arr)
arr = arr.tolist()

batch_size = 50
batch_path = []

for i in range(len(f)/batch_size):
	batch_path.append([f[ii] for ii in arr[i*batch_size:(i+1)*batch_size]])

##############################
f1 = []
for (dirpath1, dirnames1, filenames1) in walk(mypath_test_in):  
	f1.extend(filenames1) 

arr1 = np.arange(len(f1))
arr1 = arr1.tolist()

batch_size1 = 1
batch_path1 = []

for i in range(len(f1)/batch_size1):
	batch_path1.append([f1[ii1] for ii1 in arr1[i*batch_size1:(i+1)*batch_size1]])

#################################### training batch #################################

def get_trainbatch_in(batch_idx):
	x1 = 77#66
	x2 = 77#66
	#im_batch = np.zeros([batch_size,3,66,66], dtype=np.dtype('<f'))
	im_batch1 = np.zeros([batch_size,1,x1,x2], dtype=np.dtype('<f'))
	im_batch2 = np.zeros([batch_size,1,x1,x2], dtype=np.dtype('<f'))
	im_batch3 = np.zeros([batch_size,1,x1,x2], dtype=np.dtype('<f'))
	for i in range(batch_size):
		t0 = batch_path[batch_idx][i]
		t1 = cv2.imread("%s/%s" %(mypath, t0), cv2.IMREAD_COLOR)
		t1 = cv2.resize(t1, (77, 77)) 
		t2 = t1.astype(np.float32)
		t21 = t2[:,:,0]
		t21 = np.array([t21])
		t22 = t2[:,:,1]
		t22 = np.array([t22])
		t23 = t2[:,:,2]
		t23 = np.array([t23])
		
		#im_batch[i,:,:,:] = np.rollaxis(t2, -1)
		im_batch1[i,:,:,:] = t21
		im_batch2[i,:,:,:] = t22
		im_batch3[i,:,:,:] = t23
	return im_batch1, im_batch3

#################################### Validation batch #################################

def get_testbatch_in(batch_idx):
	x1 = 77#66#342
	x2 = 77#66#516
	#im_batch = np.zeros([batch_size1,3,x1,x2], dtype=np.dtype('<f'))
	im_batch1 = np.zeros([batch_size1,1,x1,x2], dtype=np.dtype('<f'))
 	im_batch2 = np.zeros([batch_size1,1,x1,x2], dtype=np.dtype('<f'))
	im_batch3 = np.zeros([batch_size1,1,x1,x2], dtype=np.dtype('<f'))
	for i in range(batch_size1):
		#t0 = batch_path1[batch_idx][i].strip('_wmap.png')
		t0 = batch_path1[batch_idx][i]
		t1 = cv2.imread("%s/%s" %(mypath_test_in, t0), cv2.IMREAD_COLOR)
		t1 = cv2.resize(t1, (77, 77)) 
		t2 = np.asarray(t1, dtype = np.float32)
		t21 = t2[:,:,0]
		t21 = np.array([t21])
		t22 = t2[:,:,1]
		t22 = np.array([t22])
		t23 = t2[:,:,2]
		t23 = np.array([t23])
		
		#im_batch[i,:,:,:] = np.rollaxis(t2, -1)
		im_batch1[i,:,:,:] = t21
		im_batch2[i,:,:,:] = t22
		im_batch3[i,:,:,:] = t23
		#im_batch[i,:,:,:] = np.rollaxis(t2, -1)
	return im_batch1, im_batch3

#################################### Testing batch #################################

def get_testbatch_in_test(batch_idx):
	for i in range(batch_size1):
		t0 = batch_path1[batch_idx][i]#.strip('_wmap.png')
		t1 = cv2.imread("%s/%s" %(mypath_test_in, t0), cv2.IMREAD_COLOR)
		#t1 = cv2.resize(t1, (0,0), fx=0.5,fy=0.5)
		x1,x2,x3 = t1.shape
		im_batch1 = np.zeros([batch_size1,1,x1,x2], dtype=np.dtype('<f'))
		im_batch2 = np.zeros([batch_size1,1,x1,x2], dtype=np.dtype('<f'))
		im_batch3 = np.zeros([batch_size1,1,x1,x2], dtype=np.dtype('<f'))
		t2 = t1.astype(np.float32)
		t21 = t2[:,:,0]
		t21 = np.array([t21])
		t22 = t2[:,:,1]
		t22 = np.array([t22])
		t23 = t2[:,:,2]
		t23 = np.array([t23])
		
		#im_batch[i,:,:,:] = np.rollaxis(t2, -1)
		im_batch1[i,:,:,:] = t21
		im_batch2[i,:,:,:] = t22
		im_batch3[i,:,:,:] = t23
	return im_batch1, im_batch3, t0
