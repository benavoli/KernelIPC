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

def plot_hypo(samples,kipc1,kipc2,alpha = 0.05, namefig=None ):
    import matplotlib.pyplot as plt
    import seaborn as sn
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib
    from matplotlib.ticker import MultipleLocator
    #plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    
    #matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = "12"
    
    
    # observed difference
    Delta = kipc1-kipc2 
    plt.figure(figsize=(8,4))
        
    lower_bound = np.percentile(samples, (alpha / 2) * 100)
    upper_bound = np.percentile(samples, (1 - alpha / 2) * 100)
    
    # plot distribution
    sn.histplot(samples, kde=True, stat='density')
    
    # plot observed value
    plt.axvline(
        Delta, color="red", linestyle="-", label="$\\Delta$"
    )
    

    
    plt.axvline(
        lower_bound,
        color="black",
        linestyle="--",
    )
    plt.axvline(
        upper_bound,
        color="black",
        linestyle="--",
    )
    
    #plt.legend()
    plt.tight_layout()
    #if namefig is not None:
    plt.savefig(namefig)
    

#set batch size for computing kernel IPC, its size should be based on the GPU memory
batch_size=5000
# length of the input signal
nT =batch_size*2#(2,20,60, 120) it should be a multiple of the batch size
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
    X2 = np.hstack([      eval_legendre(1,delainp1)[:,None],
                          eval_legendre(50,delainp1)[:,None],
                          eval_legendre(2,delainp1)[:,None]*eval_legendre(1,delainp2)[:,None],
                          ])
    
    X3 = np.hstack([
                          eval_legendre(1,delainp2)[:,None], #degree 1
                          eval_legendre(2,delainp2)[:,None],  #degree 2
                          eval_legendre(2,delainp2)[:,None]*eval_legendre(1,delainp3)[:,None],
                          ])
    X4 = np.hstack([      eval_legendre(1,delainp2)[:,None],
                          eval_legendre(50,delainp2)[:,None],
                          eval_legendre(2,delainp2)[:,None]*eval_legendre(1,delainp3)[:,None],
                          ])
        

    X=[X1,X2,X3,X4]
    
    
    ### Run model in torch ####
    KIPC = []
    RANK = []
    for ii in range(4):
        start = time.time()
        
        input_sig = input_signal[:,None]
        kipc, rank = eval_KIPC(X[ii],input_sig,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                                   rho=rho, unbiased=unbiased)
        end = time.time()
        print("reservoir 1 rank: ", rank)
        print("KIPC: ",kipc)
        print("time: ",end-start)
        KIPC.append(kipc)
        RANK.append(rank)
        
    nsample=500
    samples = hypothesis_testing(X1,X3,input_sig,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                                rho=rho,  nsample=nsample  )
    #print(samples)
   
    plot_hypo(samples,KIPC[0],KIPC[2],alpha = 0.05, namefig="diff_13.pdf" )
    
    
    samples = hypothesis_testing(X2,X4,input_sig,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                                rho=rho,  nsample=nsample  )
    #print(samples)
   
    plot_hypo(samples,KIPC[1],KIPC[3],alpha = 0.05, namefig="diff_24.pdf" )
    
    samples = hypothesis_testing(X1,X2,input_sig,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                                rho=rho,  nsample=nsample  )
    #print(samples)
   
    plot_hypo(samples,KIPC[0],KIPC[1],alpha = 0.05, namefig="diff_12.pdf" )
    
    samples = hypothesis_testing(X3,X4,input_sig,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                                rho=rho,  nsample=nsample  )
    #print(samples)
   
    plot_hypo(samples,KIPC[2],KIPC[3],alpha = 0.05, namefig="diff_34.pdf" )
    
    #Assess standalone significance with few samples:
    samples=[]
    for ns in range(300):
        print(ns)
        input_sig_perm = np.random.permutation(input_sig)
        kipc, _ = eval_KIPC(X1,input_sig_perm,max_lag= max_lag,kernel=kernel, rho=rho,
                                       batch_size=batch_size, comp_svd=True)
        samples.append(kipc)
        
    #significance estimate based on Remark 2 in the paper
    alpha = 0.05
    samples = np.array(samples)
    significance = np.percentile(np.abs(samples), (1 - alpha) * 100)
    print(significance)

   

   

   


     
    

    
    