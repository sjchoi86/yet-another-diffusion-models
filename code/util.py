import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance

# SE kernel function
def kernel_se(x1,x2,hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function """
    D = distance.cdist(x1/hyp['len'],x2/hyp['len'],'sqeuclidean')
    K = hyp['gain']*np.exp(-D)
    return K

# Numpy array to torch tensor
def np2torch(x_np,dtype=torch.float32,device='cpu'):
    x_torch = torch.tensor(x_np,dtype=dtype,device=device)
    return x_torch

# torch tensor to Numpy array
def torch2np(x_torch):
    x_np = x_torch.detach().cpu().numpy() # ndarray
    return x_np

def gather_and_reshape(values,steps,x_shape):
    values_gather = torch.gather(input=values,dim=-1,index=steps.cpu())
    n_batch = steps.shape[0]
    out_shape = (n_batch,) + ((1,)*(len(x_shape)-1))
    values_gather_reshape = values_gather.reshape(shape=out_shape)
    return values_gather_reshape.to(steps.device)


# Sample trajectories from GP
def gp_sampler(times=np.linspace(start=0.0,stop=1.0,num=100).reshape((-1,1)), # [L x 1]
               hyp_gain=1.0,hyp_len=1.0,meas_std=1e-8,n_traj=1):
    L = times.shape[0]
    K = kernel_se(times,times,hyp={'gain':hyp_gain,'len':hyp_len}) # [L x L]
    K_chol = np.linalg.cholesky(K+1e-8*np.eye(L,L)) # [L x L]
    traj = K_chol @ np.random.randn(L,n_traj) # [L x n_traj]
    traj = traj + meas_std*np.random.randn(*traj.shape)
    return traj # [L x n_traj]

# Get RKHS projection
def get_rkhs_proj(times=np.linspace(start=0.0,stop=1.0,num=100).reshape((-1,1)), # [L x 1]
              hyp_len=1.0,meas_std=1e-8):
    """
        RKHS projection
    """
    L = times.shape[0]
    K = kernel_se(times,times,hyp={'gain':1.0,'len':hyp_len}) # [L x L]
    RKHS_proj = K @ np.linalg.inv(K + meas_std*np.eye(L,L)) # [L x L]
    return RKHS_proj    