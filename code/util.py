import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance

def get_quadratic_function(x, M):
    '''
    x: [L x 1]
    M = number of trajectories
    '''
    if isinstance(x, np.ndarray):
        y = np.zeros(shape=(x.shape[0], M))

        for i in range(M):
            a = np.random.binomial(n=1, p=0.5, size=(1,)) * 2 - 1
            
            eps = np.random.normal(loc=0., scale=1., size=(1,))
            # y = a * (x[:] ** 2) + eps
            y[:, i] = a * (x.squeeze(-1) ** 2) + eps

    elif isinstance(x, torch.Tensor):
        y = torch.zeros(size=(x.shape[0], M), device=x.device)

        for i in range(M):
            a = torch.randint(low=0, high=2, size=(1,), device=x.device) * 2 - 1
            eps = torch.normal(mean=0., std=1., size=(1,), device=x.device)
            y[:, i] = a * (x[:] ** 2) + eps

    else:
        raise NotImplementedError('Unknown datatype for input x.')
    
    return y

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

# Get V and A matrices to compute velocity and accelerations
def get_V_and_A(times):
    L = times.shape[0]
    V_np,A_np = np.zeros(shape=(L,L)),np.zeros(shape=(L,L)) # [L x L]
    for l_idx in range(L):
        if l_idx == 0: # all zeros at the first row
            DO_NOTHING = True
        elif l_idx == 1: # for the second row
            V_np[l_idx,0],V_np[l_idx,1] = -1,+1
            A_np[l_idx,0],A_np[l_idx,1] = 0,0
        else: # for the rest of the rows
            V_np[l_idx,l_idx-1],V_np[l_idx,l_idx] = -1,+1
            A_np[l_idx,l_idx-2],A_np[l_idx,l_idx-1],A_np[l_idx,l_idx] = +1,-2,+1
    dt = 2e-1
    V_np,A_np = V_np/dt,A_np/(dt*dt) # [L x L]
    V,A = np2torch(V_np),np2torch(A_np)
    return V,A,V_np,A_np

def get_x_0_for_quadratic_DDPM(times, D=1, M=10, USE_SINGLE_TRAJECTORY=False, device='cpu'):
    V,A,V_np,A_np = get_V_and_A(times)
    if USE_SINGLE_TRAJECTORY:
        np.random.seed(seed=3)
        # x_0_np = gp_sampler(times=times, hyp_gain=1.0, hyp_len=0.2, meas_std=1e-8, n_traj=D).T
        x_0_np = get_quadratic_function(times, D).T
        x_0 = np2torch(x_0_np,device=device) # [D x L]
        plt.figure(figsize=(3*D,1))
        for d_idx in range(D):
            plt.subplot(1,D,d_idx+1)
            plt.plot(times[:,0],x_0_np[d_idx,:],ls='-',color='k',lw=1)
            plt.title('dim:[%d]'%(d_idx),fontsize=8)
        plt.show()
    else:
        L = times.shape[0]
        x_0_np = np.zeros(shape=(M,D,L)) # [M x D x L]
        for d_idx in range(D):
            # x_0_np_d = gp_sampler(times=times,hyp_gain=1.0,hyp_len=0.2,meas_std=1e-8,n_traj=M).T # [M x L]
            x_0_np_d = get_quadratic_function(times, M).T # [M x L]
            x_0_np[:,d_idx,:] = x_0_np_d
        x_0 = np2torch(x_0_np) # [M x D x L]
        # Plot
        for d_idx in range(D):
            plt.figure(figsize=(12,1))
            # Plot trajectory
            plt.subplot(1,3,1)
            for m_idx in range(M):
                plt.plot(times[:,0],x_0_np[m_idx,d_idx,:],ls='-',color='k',lw=1)
            plt.xlim(times[0],times[-1])
            plt.title('Trajectory (dim:[%d])'%(d_idx),fontsize=8)
            # Plot velocity
            plt.subplot(1,3,2)
            for m_idx in range(M):
                traj = x_0_np[m_idx,d_idx,:].reshape((-1,1)) # [L x 1]
                vel = V_np @ traj # [L x 1]
                plt.plot(times[:,0],vel[:,0],ls='-',color='k',lw=1)
            plt.xlim(times[0],times[-1])
            plt.title('Velocity (dim:[%d])'%(d_idx),fontsize=8)
            # Plot acceleration
            plt.subplot(1,3,3)
            for m_idx in range(M):
                traj = x_0_np[m_idx,d_idx,:].reshape((-1,1)) # [L x 1]
                acc = A_np @ traj # [L x 1]
                plt.plot(times[:,0],acc[:,0],ls='-',color='k',lw=1)
            plt.xlim(times[0],times[-1])
            plt.title('Acceleration (dim:[%d])'%(d_idx),fontsize=8)
            plt.show()
    return x_0,V,A

# Get x_0 for 1D trajectory
def get_x_0_for_1D_DDPM(times,D=1,M=10,USE_SINGLE_TRAJECTORY=False,device='cpu'):
    V,A,V_np,A_np = get_V_and_A(times)
    if USE_SINGLE_TRAJECTORY:
        np.random.seed(seed=3)
        x_0_np = gp_sampler(times=times,hyp_gain=1.0,hyp_len=0.2,meas_std=1e-8,n_traj=D).T # [D x L]
        x_0 = np2torch(x_0_np,device=device) # [D x L]
        plt.figure(figsize=(3*D,1))
        for d_idx in range(D):
            plt.subplot(1,D,d_idx+1)
            plt.plot(times[:,0],x_0_np[d_idx,:],ls='-',color='k',lw=1)
            plt.title('dim:[%d]'%(d_idx),fontsize=8)
        plt.show()
    else:
        L = times.shape[0]
        x_0_np = np.zeros(shape=(M,D,L)) # [M x D x L]
        for d_idx in range(D):
            x_0_np_d = gp_sampler(times=times,hyp_gain=1.0,hyp_len=0.2,meas_std=1e-8,n_traj=M).T # [M x L]
            x_0_np[:,d_idx,:] = x_0_np_d
        x_0 = np2torch(x_0_np) # [M x D x L]
        # Plot
        for d_idx in range(D):
            plt.figure(figsize=(12,1))
            # Plot trajectory
            plt.subplot(1,3,1)
            for m_idx in range(M):
                plt.plot(times[:,0],x_0_np[m_idx,d_idx,:],ls='-',color='k',lw=1)
            plt.xlim(times[0],times[-1])
            plt.title('Trajectory (dim:[%d])'%(d_idx),fontsize=8)
            # Plot velocity
            plt.subplot(1,3,2)
            for m_idx in range(M):
                traj = x_0_np[m_idx,d_idx,:].reshape((-1,1)) # [L x 1]
                vel = V_np @ traj # [L x 1]
                plt.plot(times[:,0],vel[:,0],ls='-',color='k',lw=1)
            plt.xlim(times[0],times[-1])
            plt.title('Velocity (dim:[%d])'%(d_idx),fontsize=8)
            # Plot acceleration
            plt.subplot(1,3,3)
            for m_idx in range(M):
                traj = x_0_np[m_idx,d_idx,:].reshape((-1,1)) # [L x 1]
                acc = A_np @ traj # [L x 1]
                plt.plot(times[:,0],acc[:,0],ls='-',color='k',lw=1)
            plt.xlim(times[0],times[-1])
            plt.title('Acceleration (dim:[%d])'%(d_idx),fontsize=8)
            plt.show()
    return x_0,V,A

