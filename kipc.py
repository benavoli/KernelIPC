#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:32:08 2026

@author: benavoli
"""


import numpy as np
from torchkernels import LegKernel
import torch
import gc
gc.collect()
torch.cuda.empty_cache()

def build_delayed(inputsignal,m,delay):
    y = np.roll(inputsignal,delay)
    y[0:delay]=0
    for i in range(delay+1,m+1):
        rolled = np.roll(inputsignal,i)
        rolled[0:i]=0.0
        y = np.hstack([y,rolled])
    return y

def compute_svd(X):
    X=X-np.mean(X,axis=0)[None,:]
    
    X, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    

    finfo = np.finfo(X.dtype).eps
    max_dim = max(X.shape) 
    eps = sigma.max() * max_dim * finfo
    
    index = np.where(sigma > eps)[0]
    X = X[:,index]
    rank=len(index)
    #print("rank=",rank)
    return X, rank
    

def eval_KIPC(
    X, inputsignal, max_lag=3, kernel='exp', p=30, gamma=1.0, 
    c=1.0, rbin=1.0, rho=0.9, powerho=None, batch_size=2000, 
    __delay=1, device="cuda", unbiased=True, comp_svd=True
):
    device = torch.device(device)
    rank=None
    if comp_svd==True:
        X,rank = compute_svd(X)
    
    
    
    
    # Build delay embedding
    Z_np = build_delayed(inputsignal, max_lag, __delay)
        
    if (kernel == "leg") or (kernel == "leg-lin"):
        kern = LegKernel(ard_num_dims=Z_np.shape[1], rho=rho)

        

    #kern(torch.tensor([[0.9]]),torch.tensor([[0.9]]),torch.tensor([0.95]))
    X = torch.from_numpy(X).to(device).float()
    Z_tensor = torch.from_numpy(Z_np).to(device).float()
    
       
    n, d = X.shape
    XtGX_tensor = torch.zeros((d, d), device=device)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            X_chunk_i = X[i : i + batch_size]
            Z_chunk_i = Z_tensor[i : i + batch_size]
            

            Y_i = torch.zeros_like(X_chunk_i)
            
            for j in range(0, n, batch_size):
                Z_chunk_j = Z_tensor[j : j + batch_size]
                X_chunk_j = X[j : j + batch_size]
                
                # compute kernel block
                if kernel == "leg":
                    G_block = kern(Z_chunk_i, Z_chunk_j, powerho=powerho).evaluate()
                elif kernel == "leg-lin":
                    G_block = kern(Z_chunk_i, Z_chunk_j, powerho=powerho, linear=True).to_dense() 
               

     
                if i==j:
                    if  unbiased==True:
                        G_block.fill_diagonal_(0) #remove diagonal elements
                    #sum_g_blocks += (G_block.diag()).sum()
                Y_i += torch.matmul(G_block, X_chunk_j)
            
            # compute trace component for block i
            XtGX_tensor += torch.matmul(X_chunk_i.T, Y_i)
    if  unbiased==True:     
        trace_PG = (torch.trace(XtGX_tensor) / (n-1)).item()
    else:
        trace_PG = (torch.trace(XtGX_tensor) / (n)).item()
    return trace_PG, rank
        


def hypothesis_testing(X1,X2, inputsignal, max_lag=3, kernel='exp', p=30, gamma=1.0, 
    c=1.0, rbin=1.0, rho=0.9, powerho=None, batch_size=2000, 
    __delay=1, device="cuda", mode="unbiased",nsample=100):
    
    X1,rank1 = compute_svd(X1)
    X2,rank2 = compute_svd(X2)
    res=[]
    for ns in range(nsample):
        print(ns)
        input_sig_perm = np.random.permutation(inputsignal)
        trace_PGnull1, _ = eval_KIPC(X1,input_sig_perm,max_lag= max_lag,kernel=kernel,p=p, gamma=gamma,
                                       c=c, rbin=rbin, rho=rho, powerho=powerho,
                                       batch_size=batch_size, __delay=__delay, device=device,comp_svd=False)
        trace_PGnull2, _ = eval_KIPC(X2,input_sig_perm,max_lag= max_lag,kernel=kernel,p=p, gamma=gamma,
                                       c=c, rbin=rbin, rho=rho, powerho=powerho,
                                       batch_size=batch_size, __delay=__delay, device=device,comp_svd=False)
        res.append(trace_PGnull1-trace_PGnull2)
    return np.hstack(res)
    
