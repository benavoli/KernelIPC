import numpy as np
#from utilities import kernel_IPC, build_delayed
import pandas as pd
from kipc import  eval_KIPC
import time
import urllib.request
import zipfile
import sys
import os
repo_dir = "./rc-bootcamp-main"

if not os.path.exists(repo_dir):
    # Direct ZIP archive download link for github.com/rc-bootcamp/rc-bootcamp
    url = "https://github.com/rc-bootcamp/rc-bootcamp/archive/refs/heads/main.zip"
    zip_path = "rc-bootcamp-main.zip"
    
    print("Downloading repository...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting repository...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    os.remove(zip_path)

sys.path.append(os.path.join(repo_dir, "src", "utils/"))
from reservoir import  ESN, Linear




#from tqdm import trange
#make_it_even=True
reserv = "nstandard" # "standard" | "nstandard"
kernel="leg-lin" # "leg" | "leg-lin              "
batch_size = 5000
for seed in range(0,10): # 0
    

    dim = 50
    t_washout = 10000
    t_sample = batch_size*40 ##120
    print(t_sample)
    srs = np.linspace(0.1, 1.7, 17)#
    t_total = t_washout + t_sample
    display = True
    
    rnd = np.random.default_rng(seed)
    np.random.seed(seed)
    w_in = Linear(1, dim, bound=0.7, bias=0.0, rnd=rnd)
    net = ESN(dim, sr=srs[:, None], f=np.tanh, p=1, rnd=rnd)
    
    x0 = np.zeros((srs.shape[0], dim))
    us = rnd.uniform(-1, 1, (t_total, 1))
    
    
    
    x = x0
    xs = np.zeros((t_total, *x0.shape))
    for idx in range(t_total):
        #if make_it_even==True:
        zeta_esn = 0.5 * us[idx] + 0.5
        if reserv=="nstandard":
            zeta_esn = (zeta_esn)**9
            

        x = net(x, w_in(zeta_esn))
        xs[idx] = x
        
    print("us:", us.shape)
    print("xs:", xs.shape)
    
    

        
    Res=pd.DataFrame(columns=["rho","kernel", "rank", "max_lag", "kipc", "time"])
    ii = 0
    
   
    for rhoind in range(len(srs)): #rhos:  
        rho = srs[rhoind]
        print(rho)
    
        # ESN
        x = xs[t_washout:,rhoind,:]
        uinp = us[t_washout:]
            
        perm=False #0.182 
        if perm==True:
            uinp = np.random.permutation(uinp)
       
        max_lag=15
        start = time.time()
        kipc, rank = eval_KIPC(x,uinp,kernel=kernel,max_lag= max_lag,batch_size=batch_size,  rho=0.95)
        end = time.time()
        #kipc = trace_PG/len(zeta)
        Res.loc[ii]=np.array([rho,kernel, rank,max_lag, kipc ,end-start])
        print(Res)
        ii = ii+1
        Res.to_csv("esn_ipcmodule_simulations_"+reserv+"_"+kernel+"_seed"+str(seed))
print(Res)