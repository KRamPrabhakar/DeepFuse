import numpy as np
import lasagne
from theano import tensor as T
from lasagne.layers import *
import time
from load_data import *
#from mef_ssim_loss_wo_wmap import *
#from sp import l1_loss
import theano
from theano.compile.debugmode import DebugMode
from theano.compile.nanguardmode import NanGuardMode

# Prepare Theano variables for inputs and targets as both are 3d variable
input_var1 = T.tensor4('inputs')
input_var3 = T.tensor4('inputs')
weight_maps_1 = T.tensor4('weight_maps_1')
weight_maps_3 = T.tensor4('weight_maps_3')


#network config
l_in1 = InputLayer(shape=(None, 1, None, None), input_var=input_var1) 
l_in3 = InputLayer(shape=(None, 1, None, None), input_var=input_var3)

###############
l_conv1 = Conv2DLayer(l_in1, 32, 3, stride=(1,1), pad='same', untie_biases=False, W=lasagne.init.GlorotNormal(), b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)

l_conv3 = Conv2DLayer(l_in3, 32, 3, stride=(1,1), pad='same', untie_biases=False, W=l_conv1.W, b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)

###############
l_conv11 = Conv2DLayer(l_conv1, 64, 5, stride=(1,1), pad='same', untie_biases=False, W=lasagne.init.GlorotNormal(), b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)

l_conv13 = Conv2DLayer(l_conv3, 64, 5, stride=(1,1), pad='same', untie_biases=False, W=l_conv11.W, b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)

############### MERGE LAYER
l_merge = lasagne.layers.ElemwiseMergeLayer([l_conv11, l_conv13], T.add, cropping=None)
###############
l_conv40 = Conv2DLayer(l_merge, 64, 7, stride=(1,1), pad='same', untie_biases=False, W=lasagne.init.GlorotNormal(), b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)
###############
l_conv4 = Conv2DLayer(l_conv40, 32, 5, stride=(1,1), pad='same', untie_biases=False, W=lasagne.init.GlorotNormal(), b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)
###############
l_conv41 = Conv2DLayer(l_conv4, 16, 5, stride=(1,1), pad='same', untie_biases=False, W=lasagne.init.GlorotNormal(), b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)
###############
l_conv5 = Conv2DLayer(l_conv41, 1, 7, stride=(1,1), pad='same', untie_biases=False, W=lasagne.init.GlorotNormal(), b=None, nonlinearity=lasagne.nonlinearities.leaky_rectify, flip_filters=True)

######################## Creating train function #######################################
nw_output = lasagne.layers.get_output(l_conv5)

##+++++++++++++ Uncomment lines 49-80 for training +++++++++++++
# ######################## Converting output to 16 to 235 range ##########################
# arr_min_axis2 = nw_output.min(axis=2, keepdims=True)
# arr_min_axis3 = arr_min_axis2.min(axis=3, keepdims=True)
# arr_max_axis2 = nw_output.max(axis=2, keepdims=True)
# arr_max_axis3 = arr_max_axis2.max(axis=3, keepdims=True)
# num = nw_output - arr_min_axis3 #(100,1,1,1)
# dem = arr_max_axis3 - arr_min_axis3 #(1,1,1,1)
# th_norm_0to1 = num/dem #(100,1,64,64)
# prediction = th_norm_0to1 * 255.0 + (1.0 - th_norm_0to1)*0.0
# ########################################################################################

# lss_mef_ssim = 100*mef_ssim_batch(prediction,input_var1,input_var3) ############# mef_ssim_loss_batch ??
# cost = lss_mef_ssim.mean()
# params = lasagne.layers.get_all_params(l_conv5, trainable=True)
# lr = 0.001
# ## SGD updates 
# #updates = lasagne.updates.sgd(cost, params, learning_rate=lr)

# ## SGD updates with nesterov momentum
# #updates = lasagne.updates.sgd(cost, params, learning_rate=lr)
# #updates_nes_sgd = lasagne.updates.apply_nesterov_momentum(updates, params, momentum=0.9)

# ## RMS prop
# # updates_rms = lasagne.updates.rmsprop(cost, params, learning_rate=lr)

# ## ADAGRAD
# #updates_adagrad = lasagne.updates.adagrad(cost, params, learning_rate=lr)

# ## ADAM
# updates_adam = lasagne.updates.adam(cost, params,learning_rate=lr,beta1=0.9,beta2=0.999)

# train_fn = theano.function([input_var1, input_var3], cost, updates=updates_adam)

######################## Creating test function #########################################
t_nw_output = lasagne.layers.get_output(l_conv5, deterministic=True)


######################## Converting output to 16 to 235 range ##########################
arr_min_axis21 = t_nw_output.min(axis=2, keepdims=True)
arr_min_axis31 = arr_min_axis21.min(axis=3, keepdims=True)
arr_max_axis21 = t_nw_output.max(axis=2, keepdims=True)
arr_max_axis31 = arr_max_axis21.max(axis=3, keepdims=True)
num1 = t_nw_output - arr_min_axis31 #(100,1,1,1)
dem1 = arr_max_axis31 - arr_min_axis31 #(1,1,1,1)
th_norm_0to1_1 = num1/dem1 #(100,1,64,64)
t_prediction = th_norm_0to1_1 * 255.0 + (1.0 - th_norm_0to1_1)*0.0
########################################################################################
##+++++++++++++ Uncomment lines 97-101 for validation +++++++++++++
# t_lss_mef_ssim = 100*mef_ssim_batch(t_prediction,input_var1,input_var3) ###### mef_ssim_loss_batch  ??
# t_cost = t_lss_mef_ssim.mean()

# ## a theano function used to calculate loss, used in training
# valid_fn = theano.function([input_var1, input_var3], [t_cost, t_prediction])
## a theano function used to generate predictions only, not loss, used in testing
valid_fn_gen = theano.function([input_var1, input_var3], t_prediction)