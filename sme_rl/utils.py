import torch as th
import torch.nn as nn
import numpy as np
import random
import os
import math
from pathlib import Path

class UniformLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        
        
        W = th.empty(n_out, n_in)
        nn.init.orthogonal_(W)
        
        W *= math.sqrt(12)
        
        self.weight = nn.Parameter(W)
        
        with th.no_grad():
            self.bias = nn.Parameter(-0.5 * self.weight.sum(dim=1))

    def forward(self, x):
        z = nn.functional.linear(x, self.weight, self.bias)
        return 0.5 * (1 + th.erf(z * 0.70710678))

class DeepUniformNetwork(nn.Module):
    def __init__(self, n_in, m_out, complexity=1, hidden_dim=None):
        super().__init__()
        
        hidden_dim = max(n_in, m_out, 128)
            
        layers = []
        
        current_dim = n_in
        
        for _ in range(complexity - 1):
            layers.append(UniformLayer(current_dim, hidden_dim))
            current_dim = hidden_dim
            
        # Output Layer
        layers.append(UniformLayer(current_dim, m_out))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class WaveTransition:
    def __init__(self, state_dim, action_dim, device=None, dtype=th.float32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.dtype = dtype

        self.weights = th.rand(size=(action_dim, state_dim))
        self.weights /= self.weights.sum(axis=1, keepdims=True)
        self.bias = th.rand(state_dim, device=device, dtype=dtype)

    def __call__(self, state, action):

        state_shift = self.bias + (action @ self.weights)
        raw_state = state + state_shift

        return th.acos(th.cos(2 * math.pi * raw_state)) / math.pi

def seed_everything(seed: int = 42):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True, warn_only=True)


def save_npz(path: Path, **arrays):

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)

def get_eval_samples(n_samples=10000, n_dims=8, min_val=-1.0, max_val=2.0, wd_weight=0.5):

    max_d = max(abs(min_val - 0), abs(max_val - 1))
    
    n_in = int(n_samples * wd_weight)
    n_out = n_samples - n_in
    
    if n_in > 0:
        pts_in = np.random.uniform(0, 1, (n_in, n_dims))
    else:
        pts_in = np.empty((0, n_dims))
    
    if n_out > 0:
        D = np.random.uniform(0, max_d, n_out)
        D_matrix = D[:, np.newaxis]
        
        pts_out = np.random.uniform(-D_matrix, 1 + D_matrix, (n_out, n_dims))
        
        snap_dims = np.random.randint(0, n_dims, n_out)
        snap_sides = np.random.randint(0, 2, n_out)
        
        snap_values = np.where(snap_sides == 0, -D, 1 + D)
        pts_out[np.arange(n_out), snap_dims] = snap_values
    else:
        pts_out = np.empty((0, n_dims))
    
    final_samples = np.vstack((pts_in, pts_out))
    np.random.shuffle(final_samples)
    
    return final_samples