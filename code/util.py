import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance
from datasets import get_quadratic_function

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
    dt = (times[-1] - times[0]) / len(times)
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

from scipy.stats import percentileofscore 
from sklearn.metrics import pairwise_distances

def MMD_K(K, M, N):
    """
    Calculates the empirical MMD^{2} given a kernel matrix computed from the samples and the sample sizes of each distribution.
    
    Parameters:
    K - kernel matrix of all pairwise kernel values of the two distributions
    M - number of samples from first distribution
    N - number of samples from second distribution
    
    Returns:
    MMDsquared - empirical estimate of MMD^{2}
    """
    
    Kxx = K[:N,:N]
    Kyy = K[N:,N:]
    Kxy = K[:N,N:]
    
    t1 = (1./(M*(M-1)))*np.sum(Kxx - np.diag(np.diagonal(Kxx)))
    t2 = (2./(M*N)) * np.sum(Kxy)
    t3 = (1./(N*(N-1)))* np.sum(Kyy - np.diag(np.diagonal(Kyy)))
    
    MMDsquared = (t1-t2+t3)
    
    return MMDsquared

def K_ID(X, Y, gamma=1):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the identity operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    n_obs = X.shape[1]
    XY = np.vstack((X,Y))
    dist_mat = (1/np.sqrt(n_obs))*pairwise_distances(XY, metric='euclidean')
    if gamma == -1:
        gamma = np.median(dist_mat[dist_mat > 0])
   
    K = np.exp(-0.5*(1/gamma**2)*(dist_mat**2))
    return K

def two_sample_test(X, Y, hyp, n_perms, z_alpha = 0.05, make_K = K_ID, return_p = False):
    """
    Performs the two sample test and returns an accept or reject statement
    
    Parameters:
    X - (n_samples, n_obs) array of samples from the first distribution 
    Y - (n_samples, n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel
    n_perms - number of permutations performed when bootstrapping the null
    z_alpha - rejection threshold of the test
    return_p - option to return the p-value of the test
    make_K - function called to construct the kernel matrix used to compute the empirical MMD
    
    Returns:
    rej - 1 if null rejected, 0 if null accepted
    p-value - p_value of test
    
    """
    
    # Number of samples of each distribution is identified and kernel matrix formed
    M = X.shape[0]
    N = Y.shape[0]
    # X = np.expand_dims(X.flatten(), -1)
    # Y = np.expand_dims(Y.flatten(), -1)
    K = make_K(X, Y, hyp) # [M x N]

    print(K.shape)
    # Empirical MMD^{2} calculated
    MMD_test = MMD_K(K, M, N)
    
    # For n_perms repeats the kernel matrix is shuffled and empirical MMD^{2} recomputed
    # to simulate the null
    shuffled_tests = np.zeros(n_perms)
    for i in range(n_perms):
            idx = np.random.permutation(M+N)
            K = K[idx, idx[:, None]]
            shuffled_tests[i] = MMD_K(K,M,N)
    
    # Threshold of the null calculated and test is rejected if empirical MMD^{2} of the data
    # is larger than the threshold
    q = np.quantile(shuffled_tests, 1.0-z_alpha)
    rej = int(MMD_test > q)
    
    if return_p:
        p_value = 1-(percentileofscore(shuffled_tests,MMD_test)/100)
        return MMD_test, rej, p_value
    else:
        return MMD_test, rej

def power_test(X_samples, Y_samples, gamma, n_tests, n_perms, z_alpha = 0.05, make_K=K_ID, return_p=False):
    """
    Computes multiple two-sample tests and returns the rejection rate
    
    Parameters:
    X_samples - (n_samples*n_tests,n_obs) array of samples from the first distribution 
    Y_samples - (n_samples*n_tests,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel
    n_tests - number of tests to perform
    n_perms - number of permutations performed when bootstrapping the null
    z_alpha - rejection threshold of the test
    make_K - function called to construct the kernel matrix used to compute the empirical MMD
    return_p - option to return the p-value of the test
    
    Returns:
    power - the rate of rejection of the null
    """
    
    # Number of samples of each distribution is identified
    M = int(X_samples.shape[0]/n_tests)
    N = int(Y_samples.shape[0]/n_tests)
    rej = np.zeros(n_tests)
    
    # For each test, extract the data to use and then perform the two-sample test
    for t in range(n_tests):
        X_t = X_samples[t*M:(t+1)*M,:]
        Y_t = Y_samples[t*N:(t+1)*N,:]
        rej[t] = two_sample_test(X_t,Y_t,gamma,n_perms,z_alpha = z_alpha,make_K = make_K,return_p = return_p)
    
    # Compute average rate of rejection
    power = np.mean(rej)
    return power
