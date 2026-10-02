#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:40:57 2026

@author: benavoli
"""

import numpy as np
from scipy.special import eval_legendre
import time
from kipc import  eval_KIPC, hypothesis_testing
import pandas as pd


#set batch size for computing kernel IPC, its size should be based on the GPU memory
batch_size=5000
# length of the input signal
nT =batch_size*20#(2,20,60, 120) it should be a multiple of the batch size
# select kernel (legendre)
kernel = "leg"
#maximum delay (m in the paper)
max_lag = 8
# rho parameter
rho = 0.95
# run Unbiased KIPC formula (set to False in Example 1)
unbiased = True
calibr=[]
for seed in range(1):

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    input_signal = np.random.permutation( rng.uniform(-1, 1, nT))

    
    delainp1 = np.roll(input_signal, 1)
    delainp2 = np.roll(input_signal, 2)
    delainp3 = np.roll(input_signal, 3)
    delainp4 = np.roll(input_signal, 4)
    delainp5 = np.roll(input_signal, 5)
    delainp6 = np.roll(input_signal, 6)

    X1 = np.hstack([
                          eval_legendre(1,delainp1)[:,None], #degree 1
                          eval_legendre(2,delainp1)[:,None],  #degree 2
                          eval_legendre(2,delainp1)[:,None]*eval_legendre(1,delainp2)[:,None],
                          ])
    X2 = np.hstack([      eval_legendre(1,delainp2)[:,None],
                          eval_legendre(50,delainp1)[:,None],
                          eval_legendre(2,delainp1)[:,None]*eval_legendre(1,delainp2)[:,None],
                          ])
        


    X = X1 # choose the reservoir
    
    ### Run model in torch ####
    start = time.time()
    
    input_sig = input_signal[:,None]
    KIPC, rank = eval_KIPC(X,input_sig,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                               rho=rho, unbiased=unbiased)
    end = time.time()
    print("reservoir rank: ", rank)
    print("KIPC: ",KIPC)
    print("time: ",end-start)
    
    