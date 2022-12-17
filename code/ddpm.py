import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from util import np2torch,torch2np,gather_and_reshape

# DDPM constants
def get_ddpm_constants(beta_start=1e-4,beta_end=0.02,diffusion_step=1000):
    """
        Denoising Diffusion Probabilistic Model Constants
    """
    dc = {}
    dc['T'] = diffusion_step
    dc['betas'] = torch.linspace(start=beta_start,end=beta_end,steps=dc['T']) # [T]
    dc['alphas'] = 1.0 - dc['betas'] # [T]
    dc['alphas_bar'] = torch.cumprod(dc['alphas'],axis=0) # [T]
    dc['alphas_bar_prev'] = torch.nn.functional.pad(dc['alphas_bar'][:-1],pad=(1,0),value=1.0) # [T]
    dc['sqrt_recip_alphas'] = torch.sqrt(1.0/dc['alphas']) # [T]
    dc['sqrt_alphas_bar'] = torch.sqrt(dc['alphas_bar']) # [T]
    dc['sqrt_one_minus_alphas_bar'] = torch.sqrt(1.0-dc['alphas_bar']) # [T]
    dc['posterior_variance'] = dc['betas']*(1.0-dc['alphas_bar_prev'])/(1.0-dc['alphas_bar']) # [T]
    return dc

# Plot DDPM constants
def plot_ddpm_constants(dc,figsize=(12,4)):
    ts = np.linspace(start=1,stop=dc['T'],num=dc['T'])
    cs = [plt.cm.gist_rainbow(x) for x in np.linspace(0,1,8)]
    lw = 2 # linewidth
    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.plot(ts,torch2np(dc['betas']),color=cs[0],label=r'$\beta_t$',lw=lw)
    plt.plot(ts,torch2np(dc['alphas']),color=cs[1],label=r'$\alpha_t$',lw=lw)
    plt.plot(ts,torch2np(dc['alphas_bar']),color=cs[2],label=r'$\bar{\alpha}_t$',lw=lw)
    plt.plot(ts,torch2np(dc['alphas_bar_prev']),color=cs[3],label=r'$\bar{\alpha}_{t-1}$',lw=lw)
    plt.plot(ts,torch2np(dc['sqrt_recip_alphas']),color=cs[4],label=r'$\frac{1}{\sqrt{\alpha_t}}$',lw=lw)
    plt.plot(ts,torch2np(dc['sqrt_alphas_bar']),color=cs[5],label=r'$\sqrt{\bar{\alpha}_t}$',lw=lw)
    plt.plot(ts,torch2np(dc['sqrt_one_minus_alphas_bar']),color=cs[6],label=r'$\sqrt{1-\bar{\alpha}_t}$',lw=lw)

    plt.plot(ts,torch2np(dc['betas'])/torch2np(dc['sqrt_one_minus_alphas_bar']),
        color=cs[7],label=r'$\frac{1-\alpha_t}{ \sqrt{ 1-\bar{\alpha}_t } }$',lw=lw)

    plt.plot(ts,torch2np(dc['posterior_variance']),'--',color='k',label=r'$ Var[x_{t-1}|x_t,x_0] $',lw=lw)
    plt.xlabel('Diffusion steps',fontsize=8)
    plt.legend(fontsize=10,loc='center left',bbox_to_anchor=(1,0.5))
    plt.title('DDPM Constants',fontsize=10); plt.grid(lw=0.5)

    plt.subplot(1,2,2)
    plt.plot(ts,torch2np(dc['betas']),color=cs[0],label=r'$\beta_t$',lw=lw)
    plt.plot(ts,torch2np(dc['posterior_variance']),'--',color='k',label=r'$ Var[x_{t-1}|x_t,x_0] $',lw=lw)

    plt.plot(ts,torch2np(dc['betas'])/torch2np(dc['sqrt_one_minus_alphas_bar']),
        color=cs[7],label=r'$\frac{1-\alpha_t}{ \sqrt{ 1-\bar{\alpha}_t } }$',lw=lw)
        
    plt.xlabel('Diffusion steps',fontsize=8)
    plt.legend(fontsize=10,loc='center left',bbox_to_anchor=(1,0.5))
    plt.title('DDPM Constants',fontsize=10); plt.grid(lw=0.5); 
    plt.subplots_adjust(wspace=0.6); plt.show()

# Forward Hilbert diffusion sampler
def forward_hilbert_diffusion_sample(x_0,K_chols,steps,dc,noise_rate=1.0,RKHS_projs=None,
                                     noise_type='Gaussian',device='cpu'):
    """
        x_0: torch.Tensor [B x D x L]
        K_chols: torch.Tensor [D x L x L]
        steps: torch.Tensor [B]
        dc: dictionary
        noise_rate: float (0.0~1.0)
        RKHS_projs: torch.Tensor [D x L x L]
    """
    sqrt_alphas_bar_t = gather_and_reshape(
        values=dc['sqrt_alphas_bar'],steps=steps,x_shape=x_0.shape) # [B x 1 x 1]
    sqrt_one_minus_alphas_bar_t = gather_and_reshape(
        values=dc['sqrt_one_minus_alphas_bar'],steps=steps,x_shape=x_0.shape) # [B x 1 x 1]
    x_t_mean = sqrt_alphas_bar_t * x_0 # [B x D x L]
    x_t_std = sqrt_one_minus_alphas_bar_t # [B x D x L]
    # Correlated noise sampling
    if noise_type == 'Gaussian':
        noise = torch.randn_like(input=x_0) # [B x D x L]
    elif noise_type == 'Uniform':
        noise = 6.0*torch.rand_like(input=x_0)-3.0 # [B x D x L]
    else:
        print ("[forward_hilbert_diffusion_sample] Unknown [%s]"%(noise_type))
        noise = torch.randn_like(input=x_0) # [B x D x L]
    noise_expand = noise[:,:,:,None] # [B x D x L x 1]
    K_chols_torch_tile = torch.tile(input=K_chols,dims=(x_0.shape[0],1,1,1)) # [B x D x L x L]
    correlated_noise_permuted = K_chols_torch_tile @ noise_expand # [B x D x L x 1]
    correlated_noise = correlated_noise_permuted.squeeze(dim=3) # [B x D x L]
    
    # RKHS projection of 'correlated_noise'
    if RKHS_projs is not None:
        RKHS_projs_exapnd = RKHS_projs[None,:,:,:] # [1 x D x L x L]
        RKHS_projs_tile = torch.tile(RKHS_projs_exapnd,dims=(x_0.shape[0],1,1,1)) # [B x D x L x L]
        correlated_noise_expand = correlated_noise[:,:,:,None] # [B x D x L x 1]
        temp = RKHS_projs_tile @ correlated_noise_expand # [B x D x L x L] x [B x D x L x 1] => [B x D x L x 1]
        correlated_noise = temp.squeeze(dim=3) # [B x D x L]
    
    # Sample with correlated noise
    x_t = x_t_mean + noise_rate*x_t_std*correlated_noise # [B x D x L]
        
    return x_t,correlated_noise # [B x D x L]
    
# DDPM loss
def get_ddpm_loss(model,x_batch,K_chols,A=None,V=None,
                  t=0,dc={},noise_rate=1.0,RKHS_projs=None,noise_type='Gaussian',
                  l1_w=0.0,l2_w=1.0,huber_w=0.0,smt_l1_w=0.0,
                  vel_w=1e-6,acc_w=1e-6):
    """
        x_batch: [B x D x L]
        K_chols: [D x L x L]
        A: [L x L]
        V: [L x L]
    """
    # Sample from forward diffusion
    x_noisy,noise = forward_hilbert_diffusion_sample(
        x_0=x_batch,K_chols=K_chols,steps=t,dc=dc,noise_rate=noise_rate,
        RKHS_projs=RKHS_projs,noise_type=noise_type
    ) # [B x D x L]

    # Predict noise
    x_noisy_flat = x_noisy.reshape(x_noisy.shape[0],-1) # [B x DL]
    noise_pred = model(x_noisy_flat, t) # [B x DL]
    noise_pred_unflat = noise_pred.reshape_as(x_batch) # [B x D x L]
    x_t_minus_one = x_batch - 0.01*noise_pred_unflat # [B x D x L]
    
    # Smoothing loss
    if V is not None:
        V_expand = V[None,None,:,:] # [1 x 1 x L x L]
        V_expand_tile = torch.tile(input=V_expand,dims=(x_t_minus_one.shape[0],x_t_minus_one.shape[1],1,1)) # [B x D x L x L]
        x_t_minus_one_expand = x_t_minus_one[:,:,:,None] # [B x D x L x 1]
        vel = V_expand_tile @ x_t_minus_one_expand # [B x D x L x 1]
        vel = vel.squeeze(dim=3) # [B x D x L]
        vel_squared = torch.square(vel) # [B x D x L]
        vel_loss = torch.mean(vel_squared) # [1]: average acceleration error
    else:
        vel_loss = 0.0
    if A is not None:
        A_expand = A[None,None,:,:] # [1 x 1 x L x L]
        A_expand_tile = torch.tile(input=A_expand,dims=(x_t_minus_one.shape[0],x_t_minus_one.shape[1],1,1)) # [B x D x L x L]
        x_t_minus_one_expand = x_t_minus_one[:,:,:,None] # [B x D x L x 1]
        acc = A_expand_tile @ x_t_minus_one_expand # [B x D x L x 1]
        acc = acc.squeeze(dim=3) # [B x D x L]
        acc_squared = torch.square(acc) # [B x D x L]
        acc_loss = torch.mean(acc_squared) # [1]: average acceleration error
    else:
        acc_loss = 0.0

    # Compute prediction loss
    l1_loss     = F.l1_loss(noise,noise_pred_unflat) # [1]
    l2_loss     = F.mse_loss(noise,noise_pred_unflat) # [1]
    huber_loss  = F.huber_loss(noise,noise_pred_unflat) # [1]
    smt_l1_loss = F.smooth_l1_loss(noise,noise_pred_unflat,beta=0.1) # [1]
    loss = l1_w*l1_loss+l2_w*l2_loss+huber_w*huber_loss+smt_l1_w*smt_l1_loss+ \
            vel_w*vel_loss+acc_w*acc_loss # [1]
    info = {'l1':l1_w*l1_loss,'l2':l2_w*l2_loss,'huber':huber_w*huber_loss,'smt_l1':smt_l1_w*smt_l1_loss,
            'vel':vel_w*vel_loss,'acc':acc_w*acc_loss}

    return loss,info

# Evaluate
def eval_hddpm_1D(
    model,dc,K_chols,RKHS_projs,times,x_0,
    B=3,M=8,device='cpu',
    RKHS_PROJECTION_EACH_X_T=False,
    PLOT_GENERATED_TRAJECTORIES_AT_ONCE=True,
    PLOT_ANCESTRAL_STEPS=True):
    """
        Evaluate Hilbert-space DDPM for 1D
    """
    model.eval()
    D = K_chols.shape[0]
    L = times.shape[0]
    # Sample x_T from prior
    x_0_dummy = np2torch(np.zeros((B,D,L)),device=device)
    steps = torch.zeros(B).type(torch.long).to(device) # [B]
    _,correlated_noise = forward_hilbert_diffusion_sample(
        x_0=x_0_dummy,K_chols=K_chols,steps=steps,dc=dc,noise_rate=1.0,
        RKHS_projs=RKHS_projs,noise_type='Gaussian')
    x_T = correlated_noise.clone() # [B x D x L]
    x_t = x_T.clone()
    # Generate
    x_ts = ['']*dc['T']
    for t in range(0,dc['T'])[::-1]:
        # Projection of 'x_t' (x_t: [B x D x L], RKHS_projs: [D x L x L])
        if RKHS_PROJECTION_EACH_X_T:
            RKHS_projs_exapnd = RKHS_projs[None,:,:,:] # [1 x D x L x L]
            RKHS_projs_tile = torch.tile(RKHS_projs_exapnd,dims=(B,1,1,1)) # [B x D x L x L]
            x_t_expand = x_t[:,:,:,None] # [B x D x L x 1]
            x_t = RKHS_projs_tile @ x_t_expand # [B x D x L x L] x [B x D x L x 1] => [B x D x L x 1]
            x_t = x_t.squeeze(dim=3) # [B x D x L]
        # Epsilon network
        step = torch.full((1,), t,device=device,dtype=torch.long)
        x_t_flat = x_t.reshape(x_t.shape[0],-1) # [B x DL]
        eps_t = model(x=x_t_flat,t=step) # [B x DL]
        eps_t_unflat = eps_t.reshape_as(x_t) # [B x D x L]
        # Diffusion constants
        betas_t = gather_and_reshape(
            values=dc['betas'],steps=step,x_shape=x_t.shape) # [B x 1 x 1]
        sqrt_one_minus_alphas_bar_t = gather_and_reshape(
            values=dc['sqrt_one_minus_alphas_bar'],steps=step,x_shape=x_t.shape)
        sqrt_recip_alphas_t = gather_and_reshape(
            values=dc['sqrt_recip_alphas'],steps=step,x_shape=x_t.shape)
        # Compute posterior mean
        model_mean_t = sqrt_recip_alphas_t * (
            x_t - betas_t*eps_t_unflat/sqrt_one_minus_alphas_bar_t
        ) # [B x D x L]
        # Compute correlated noise
        x_0_dummy = np2torch(np.zeros((B,D,L)),device=device)
        steps = torch.zeros(B).type(torch.long) # [B]
        _,noise_t = forward_hilbert_diffusion_sample(
            x_0=x_0_dummy,K_chols=K_chols,steps=steps,dc=dc,noise_rate=1.0,
            RKHS_projs=RKHS_projs,noise_type='Gaussian'
        )
        noise_t # [B x D x L]
        # Compute posterior variance
        posterior_variance_t = gather_and_reshape(
            values=dc['posterior_variance'],steps=step,x_shape=x_t.shape)
        # Sample
        if t == 0: # last sampling, use mean
            x_t = model_mean_t
        else:
            x_t = model_mean_t + torch.sqrt(posterior_variance_t) * noise_t
        # Append
        x_ts[t] = x_t # [B x D x L]
    
    # Plot generated trajectories
    if PLOT_GENERATED_TRAJECTORIES_AT_ONCE:
        for d_idx in range(D):
            plt.figure(figsize=(6,2))
            for b_idx in range(B):
                # Plot sampled trajectoires
                for b_idx in range(B):
                    plt.plot(times[:,0],torch2np(x_t)[b_idx,d_idx,:],ls='-',color='k')
                # Plot training data
                if len(x_0.shape) == 2: # [D x L]
                    plt.plot(times[:,0],torch2np(x_0)[d_idx,:],ls='-',color='b',lw=1/4); 
                elif len(x_0.shape) == 3: # [M x D x L]
                    M = x_0.shape[0]
                    for m_idx in range(M):
                        plt.plot(times[:,0],torch2np(x_0)[m_idx,d_idx,:],ls='-',color='b',lw=1/4)
            plt.grid('on')
            plt.title('Generated Samples (D:[%d])'%(d_idx),fontsize=8); 
            plt.xlim(0,+1); plt.ylim(-2.5,+2.5)
            plt.show()
    else:
        for d_idx in range(D):
            for b_idx in range(B):
                plt.figure(figsize=(6,1))
                plt.subplot(1,2,1)
                if len(x_0.shape) == 2:
                    plt.plot(times[:,0],torch2np(x_0)[d_idx,:],ls='-',color='b');
                elif len(x_0.shape) == 3: # [M x D x L]
                    M = x_0.shape[0]
                    for m_idx in range(M):
                        plt.plot(times[:,0],torch2np(x_0)[m_idx,d_idx,:],ls='-',color='b',lw=1/2); 
                plt.grid('on')
                plt.title('batch:[%d/%d] dim:[%d] Data'%(b_idx,B,d_idx),fontsize=8); 
                plt.xlim(0,+1); plt.ylim(-2.5,+2.5)
                plt.subplot(1,2,2)
                if len(x_0.shape) == 2:
                    plt.plot(times[:,0],torch2np(x_0)[d_idx,:],ls='-',color='b',lw=1/2); 
                elif len(x_0.shape) == 3: # [M x D x L]
                    M = x_0.shape[0]
                    for m_idx in range(M):
                        plt.plot(times[:,0],torch2np(x_0)[m_idx,d_idx,:],ls='-',color='b',lw=1/4); 
                plt.plot(times[:,0],torch2np(x_t)[b_idx,d_idx,:],ls='-',color='k'); plt.grid('on')
                plt.title('dim:[%d]'%(d_idx),fontsize=8); 
                plt.xlim(0,+1); plt.ylim(-2.5,+2.5)
                plt.show()
    
            
    # Plot how the trajectories are generated
    if PLOT_ANCESTRAL_STEPS:
        for d_idx in range(D):
            for b_idx in range(B):
                plt.figure(figsize=(12,1))
                for subplot_idx,t in enumerate(np.linspace(0,dc['T']-1,M).astype(np.int64)):
                    plt.subplot(1,M,subplot_idx+1)
                    if len(x_0.shape) == 2:
                        plt.plot(times[:,0],torch2np(x_0)[d_idx,:],ls='-',color='b',lw=1) # GT 
                    elif len(x_0.shape) == 3: # [M x D x L]
                        M = x_0.shape[0]
                        for m_idx in range(M):
                            plt.plot(times[:,0],torch2np(x_0)[m_idx,d_idx,:],ls='-',color='b',lw=1/2); 
                    plt.plot(times[:,0],torch2np(x_ts[t][b_idx,d_idx,:]),ls='-',color='k',lw=1) # generated
                    plt.xlim(0,+1); plt.ylim(-2.5,+2.5); plt.grid('on')
                    plt.title('dim:[%d] t:[%d]'%(d_idx,t),fontsize=8)
                plt.show()    

    # Back to train mode
    model.train()