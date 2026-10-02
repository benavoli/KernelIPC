import numpy as np
#from utilities import kernel_IPC, build_delayed
import pandas as pd
from kipc import  eval_KIPC
import time
#import sys
#sys.path.append("./rc-bootcamp-main/src/utils")
from qrc_functions import FullQRC



#from tqdm import trange
#make_it_even=True


batch_size = 5000
max_lag = 30

experiment = "parameter_h" #"parameter_W" | "parameter_h"
rho = 0.95
powerho=False
for seed in range(0,30): # 0
    

    N_QUBITS = 8
    t_washout = 1000
    t_sample = batch_size*4 ##120
    t_total = t_washout + t_sample
    
    
    
    # Grid sweeps
    deltapr=np.array([0.01,0.1,0.5,1.0,
                      1.25,1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 5.0, 10.0, 25.0, 50.0, 75.0, 200.0])
    h_vals = deltapr
    #W_vals = np.linspace(0.01, 15.0, 20)
    W_vals = deltapr

    
    rnd = np.random.default_rng(seed)
    us = rnd.uniform(-1, 1, t_total)
    
    n_features = 3 * N_QUBITS + (N_QUBITS * (N_QUBITS - 1)) // 2
    if experiment=="parameter_W":
        xs = np.zeros((t_total, len(W_vals), n_features))
        
        for idx, W in enumerate(W_vals):
            qrc = FullQRC(n_qubits=N_QUBITS, h=10.0, W=W, g=0.0, dt=10.0, seed=seed)
            xs[:, idx, :] = qrc.run_simulation(us)
        srs = W_vals
    elif experiment=="parameter_h":
        xs = np.zeros((t_total, len(h_vals), n_features))
        
        for idx, h in enumerate(h_vals):
            qrc = FullQRC(n_qubits=N_QUBITS, h=h, W=0, g=0.0, dt=10.0, seed=seed)
            xs[:, idx, :] = qrc.run_simulation(us)
        srs=h_vals
        
    us=np.array([us]).T
    
    
    
        
    Res=pd.DataFrame(columns=[experiment,"rho", "powerrho", "rank", "max_lag", "kipc", "time"])
    ii = 0
    kernel="leg" 

    for rhoind in range(len(srs)): #rhos:  
        param = srs[rhoind]
        print(param)
    
        # ESN
        X = xs[t_washout:,rhoind,:]
        zetatr=us[t_washout:]
        
        start = time.time()
        

        kipc, rank = eval_KIPC(X,zetatr,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                                   rho=rho, powerho=powerho)
        end = time.time()
        print("reservoir rank: ", rank)
        print("KIPC: ",kipc)
        print("time: ",end-start)
            

        Res.loc[ii]=np.array([param,rho,powerho,rank,max_lag, kipc ,end-start])
        print(Res)
        ii = ii+1
        Res.to_csv("qrc_ipcmodule_simulations_"+experiment+"_"+kernel+"_seed"+str(seed))
        
            
    Res=pd.DataFrame(columns=[experiment,"rho", "powerrho", "rank", "max_lag", "kipc", "time"])
    ii = 0
    kernel="leg-lin" 

    for rhoind in range(len(srs)): #rhos:  
        param = srs[rhoind]
        print(param)
    
        # ESN
        X = xs[t_washout:,rhoind,:]
        zetatr=us[t_washout:]
        
        start = time.time()
        

        kipc, rank = eval_KIPC(X,zetatr,kernel=kernel,max_lag= max_lag,batch_size=batch_size, 
                                   rho=rho, powerho=powerho)
        end = time.time()
        print("reservoir rank: ", rank)
        print("KIPC: ",kipc)
        print("time: ",end-start)
            

        Res.loc[ii]=np.array([param,rho,powerho,rank,max_lag, kipc ,end-start])
        print(Res)
        ii = ii+1
        Res.to_csv("qrc_ipcmodule_simulations_"+experiment+"_"+kernel+"_seed"+str(seed))
print(Res)