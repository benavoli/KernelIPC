#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import expm
from tqdm.auto import tqdm
class FullQRC:
    """
    Optimized Quantum Reservoir Computing Class.
    Substrate: H = sum_{j<k} J_jk X_j X_k + sum_j (h + w_j) Z_j
    Injection: Erase-and-write with |psi(s)> = sqrt(1-s)|0> + sqrt(s)|1>
    """
    def __init__(self, n_qubits=6, Js=1.0, h=0.5, W=0.0, g=0.0, dt=0.5, seed=42):
        self.n = n_qubits
        self.dim = 2**n_qubits
        self.h = h
        self.W = W
        self.g = g  # Fixed to 0.0
        self.dt = dt
        
        np.random.seed(seed)
        
        # J_{jk} ~ U([-Js/2, Js/2])
        self.J_matrix = np.zeros((self.n, self.n))
        for j in range(self.n):
            for k in range(j + 1, self.n):
                val = np.random.uniform(-Js/2, Js/2)
                self.J_matrix[j, k] = val
                self.J_matrix[k, j] = val
                
        # w_j ~ U([-W, W])
        if W > 0:
            self.w_j = np.random.uniform(-W, W, self.n)
        else:
            self.w_j = np.zeros(self.n)
            
        self._build_operators()
        self.H = self._build_hamiltonian()
        self.U = expm(-1j * self.H * self.dt)
        self.U_dag = self.U.conj().T

    def _build_operators(self):
        # Single-qubit Pauli matrices
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        self.SX = []
        self.SY = []
        self.SZ = []
        
        # Construct operators using Kron
        for i in range(self.n):
            op_x, op_y, op_z = [I] * self.n, [I] * self.n, [I] * self.n
            op_x[i], op_y[i], op_z[i] = sx, sy, sz
            
            mat_x, mat_y, mat_z = op_x[0], op_y[0], op_z[0]
            for k in range(1, self.n):
                mat_x = np.kron(mat_x, op_x[k])
                mat_y = np.kron(mat_y, op_y[k])
                mat_z = np.kron(mat_z, op_z[k])
                
            self.SX.append(mat_x)
            self.SY.append(mat_y)
            self.SZ.append(mat_z)

        # Pre-compute two-qubit operators SX_i @ SX_j
        self.SXX = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.SXX[(i, j)] = self.SX[i] @ self.SX[j]

    def _build_hamiltonian(self):
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.n):
            for k in range(j + 1, self.n):
                H += self.J_matrix[j, k] * self.SXX[(j, k)]
        for j in range(self.n):
            H += (self.h + self.w_j[j]) * self.SZ[j]
        return H

    def _partial_trace_qubit0(self, rho):
        """Optimized partial trace over qubit 0 via tensor reshaping."""
        half_dim = self.dim // 2
        rho_reshaped = rho.reshape(2, half_dim, 2, half_dim)
        # Summing over block diagonals (tracing out 1st qubit)
        return rho_reshaped[0, :, 0, :] + rho_reshaped[1, :, 1, :]

    def extract_observables(self, rho):
        features = []
        # Fast trace using inner product: Tr(A @ B) = sum(A.T * B)
        # For Hermitian operators A = A.T.conj(), Tr(A @ rho) = real(sum(A.conj() * rho))
        
        # Single-qubit <X_i>, <Y_i>, and <Z_i>
        for i in range(self.n):
            features.append(np.real(np.sum(self.SX[i] * rho)))
            features.append(np.real(np.sum(self.SY[i] * rho)))
            features.append(np.real(np.sum(self.SZ[i] * rho)))
            
        # Two-qubit <X_i X_j> correlations
        for i in range(self.n):
            for j in range(i + 1, self.n):
                features.append(np.real(np.sum(self.SXX[(i, j)] * rho)))
                
        return np.array(features)

    def run_simulation(self, u_seq):
        rho = np.eye(self.dim, dtype=complex) / self.dim
        features_list = []
        
        s_seq = np.clip(0.5 * (u_seq.ravel() + 1.0), 0.0, 1.0)
        
        for s in tqdm(s_seq, desc="Simulating QRC steps", leave=True):
            # Erase-and-Write state injection
            psi_in = np.array([np.sqrt(1.0 - s), np.sqrt(s)], dtype=complex)
            rho_in = np.outer(psi_in, np.conj(psi_in))
            
            rho_rem = self._partial_trace_qubit0(rho)
            rho_injected = np.kron(rho_in, rho_rem)
            
            # Unitary propagation
            rho = self.U @ rho_injected @ self.U_dag
            
            # Fast Feature extraction
            features_list.append(self.extract_observables(rho))
            
        return np.array(features_list)