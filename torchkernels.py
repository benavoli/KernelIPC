#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:10:30 2026

@author: benavoli
"""

import torch
#import torch.nn as nn

from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval


class LegKernel(Kernel):
    """Legendre polynomial kernel with per-dimension estimable rho parameters."""
    
    def __init__(self, ard_num_dims=None, rho=0.7, **kwargs):
        super().__init__(**kwargs)
        
        # ard_num_dims = number of dimensions (x.shape[1])
        self.ard_num_dims = ard_num_dims
        
        # Register raw rho parameter (log-transformed for  (0, 1) constraint)
        rho_tensor = torch.ones(ard_num_dims) * rho
        self.register_parameter(
            name='raw_rho',
            parameter=torch.nn.Parameter(torch.log(rho_tensor / (1 - rho_tensor)))
        )
        
        # Constrain rho to be in (0, 1)
        self.register_constraint("raw_rho", Interval(0, 1))
        
    @property
    def rho(self):
        """Actual rho parameter in (0, 1) per dimension."""
        return self.raw_rho_constraint.transform(self.raw_rho)
    
    @rho.setter
    def rho(self, value):
        self._set_rho(value)
    
    def _set_rho(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho)
        self.initialize(raw_rho=self.raw_rho_constraint.inverse_transform(value))
    
    def forward(self, x1, x2, diag=False,powerho=False, linear=False, **params):
        """
        Compute the Legendre kernel K(x,y).
        
        Args:
            x1: First input tensor (n1, d) or (batch, n1, d)
            x2: Second input tensor (n2, d) or (batch, n2, d)
            diag: If True, compute diagonal elements only
            
        Returns:
            Kernel matrix or diagonal vector
        """
        # Ensure same number of dimensions
        if x1.shape[-1] != x2.shape[-1]:
            raise ValueError("x1 and x2 must have same number of dimensions")
        
        # Get rho for each dimension
        rho = self.rho
        if linear==False:
            return self.compute_kern(x1, x2, rho, diag=diag,powerho=powerho)
        elif linear==True:
            return self.computer_kern_linear(x1, x2, rho, diag=diag,powerho=powerho)

   

    def computer_kern_linear(self, x, y, rho, diag=False, powerho=False, eps=1e-10, chunk=10):
        D = x.shape[1]
        ret = torch.as_tensor(torch.ones(1, dtype=x.dtype),device=x.device)
        rho = torch.as_tensor(rho, device=x.device, dtype=x.dtype)
        for i in range(0, D):
            Legpol1 = torch.special.legendre_polynomial_p(x[:,[i]], n=1)
            Legpol2 = torch.special.legendre_polynomial_p(y[:,[i]], n=1)
            ret = ret + (rho[i])*Legpol1@Legpol2.T
        return ret
   
           
    @torch.compile(mode="reduce-overhead")
    def __torch_ellipk(self, m):
        """
        Approximates the complete elliptic integral of the first kind.
        """
        g0 = (1.0 - m).clamp_min_(0.0).sqrt_()
        a = (g0 + 1.0).mul_(0.5)
        g = g0.sqrt()
    
        for _ in range(6):  # 5
            ag = a * g
            a.add_(g).mul_(0.5)
            g = ag.sqrt_()
    
        return (torch.pi * 0.5) / a
        
    def compute_kern(self, x, y, rho, diag=False, powerho=False, eps=1e-10, chunk=10):
        pi = torch.pi
        D = x.shape[1]
        
        
    
        #
        rho = torch.as_tensor(rho, device=x.device, dtype=x.dtype)
        if powerho:
            exps = torch.arange(1, D + 1, device=x.device, dtype=x.dtype)
            rho_vec = rho ** exps          
        else:
            rho_vec = rho                  
    
        if not diag:
            ret = None
            for start in range(0, D, chunk):
                end = min(start + chunk, D)
                xi = x[:, start:end]
                yi = y[:, start:end]
        
     
                xT = xi.T.unsqueeze(-1)
                yT = yi.T.unsqueeze(1)
                rho_r = rho_vec[start:end].view(-1, 1, 1)
    
                t1 = 1 - 2 * xT * yT * rho_r + rho_r ** 2
                sx = torch.sqrt(1 - xT ** 2)
                sy = torch.sqrt(1 - yT ** 2)
                t2 = 2 * rho_r * sx * sy
                dd = t1 + t2
                z = 2 * t2 / dd
    
                ellip = self.__torch_ellipk(z)
                kern = (2 * ellip) / (pi * torch.sqrt(dd))
                chunk_ret = torch.prod(kern, dim=0)     # (N, M)

        
                ret = chunk_ret if ret is None else ret * chunk_ret   #tensor product kernel
                del chunk_ret
    
        else:  # diagonal case
            outs = []
            for start in range(0, D, chunk):
                end = min(start + chunk, D)
                xi = x[:, start:end]           # 
                yi = y[:, start:end]           # 
    
    
                xT = xi.T               # 
                yT = yi.T                # 
                rho_r = rho_vec[start:end].view(-1, 1)  # 

                t1 = 1 - 2 * xT * yT * rho_r + rho_r ** 2
                t2 = -2 * rho_r * torch.sqrt(1 - xT ** 2) * torch.sqrt(1 - yT ** 2)
                z = 2 * t2 / (t2 - t1)

                ellip = self.__torch_ellipk(z)               # 
                val = 2 * ellip / (pi * torch.sqrt(t1 - t2))
                outs.append(torch.prod(val, dim=0))          # 

    
            ret = torch.stack(outs, dim=0).prod(dim=0)           # 
    
        return ret

