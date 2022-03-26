import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import pylab
#from mef_ssim_score import *
execfile('main.py')

Num_of_test_images 	= 3
with np.load("DF_model_Epoch99.npz") as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(l_conv5, param_values)

for idx in range(Num_of_test_images):
        vt11,vt13, t0 = get_testbatch_in_test(idx)
        t_predict = valid_fn_gen(vt11,vt13)
        t2 = np.squeeze(t_predict)	
        scipy.misc.imsave('testimgs/'+t0, t2)