import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Position embedding
class SinPositionEmbeddingsClass(nn.Module):
    def __init__(self,dim=128,T=1000):
        super().__init__()
        self.dim = dim
        self.T = T
    @torch.no_grad()
    def forward(self,steps=torch.arange(start=0,end=1000,step=1)):
        device = steps.device
        half_dim = self.dim // 2
        embeddings = np.log(self.T) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = steps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# DDPN networks
class DenseBlockClass(nn.Module):
    def __init__(self,in_dim=10,out_dim=5,pos_emb_dim=10,actv=nn.ReLU(),
                 USE_POS_EMB=True):
        """
            Initialize
        """
        super(DenseBlockClass,self).__init__()
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.pos_emb_dim = pos_emb_dim
        self.actv        = actv
        self.USE_POS_EMB = USE_POS_EMB
        self.dense1 = nn.Linear(self.in_dim,self.out_dim)
        self.bnorm1 = nn.BatchNorm1d(self.out_dim)
        self.dense2 = nn.Linear(self.out_dim,self.out_dim)
        self.bnorm2 = nn.BatchNorm1d(self.out_dim)
        self.pos_emb_mlp = nn.Linear(self.pos_emb_dim,self.out_dim)
        
    def forward(self,x,t):
        """
            Forward
        """
        h = self.bnorm1(self.actv(self.dense1(x))) # dense -> actv -> bnrom1 [B x out_dim]
        if self.USE_POS_EMB:
            h = h + self.pos_emb_mlp(t) # [B x out_dim]
        h = self.bnorm2(self.actv(self.dense2(h))) # [B x out_dim]
        return h

class DenoisingDenseUNetClass(nn.Module):
    def __init__(self,
                 name        = 'dense_unet',
                 D           = 3,
                 L           = 100,
                 pos_emb_dim = 128,
                 h_dims      = [128,64],
                 z_dim       = 32,
                 actv        = nn.ReLU(),
                 USE_POS_EMB = True,
                 RKHS_projs  = None
                ):
        """
            Initialize
        """
        super(DenoisingDenseUNetClass,self).__init__()
        self.name        = name
        self.D           = D
        self.L           = L
        self.x_dim       = self.D * self.L
        self.pos_emb_dim = pos_emb_dim
        self.h_dims      = h_dims
        self.z_dim       = z_dim
        self.actv        = actv
        self.USE_POS_EMB = USE_POS_EMB
        self.RKHS_projs  = RKHS_projs
        # Initialize layers
        self.init_layers()
        
    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = nn.ModuleDict()
        # Encoder
        h_prev = self.x_dim
        for h_idx,h_dim in enumerate(self.h_dims):
            self.layers['Enc_%02d'%(h_idx)] = DenseBlockClass(
                in_dim=h_prev,out_dim=h_dim,pos_emb_dim=self.pos_emb_dim,actv=self.actv,
                USE_POS_EMB = self.USE_POS_EMB)
            h_prev = h_dim
        self.layers['Enc_%02d'%(len(self.h_dims))] = DenseBlockClass(
            in_dim=self.h_dims[-1],out_dim=self.z_dim,pos_emb_dim=self.pos_emb_dim,actv=self.actv,
            USE_POS_EMB=self.USE_POS_EMB)
        # Map
        self.layers['Map'] = DenseBlockClass(
            in_dim=self.z_dim,out_dim=self.z_dim,pos_emb_dim=self.pos_emb_dim,actv=self.actv,
            USE_POS_EMB=self.USE_POS_EMB)
        # Decoder
        h_prev = self.z_dim
        for h_idx,h_dim in enumerate(self.h_dims[::-1]):
            self.layers['Dec_%02d'%(h_idx)] = DenseBlockClass(
                in_dim=h_prev,out_dim=h_dim,pos_emb_dim=self.pos_emb_dim,actv=self.actv,
                USE_POS_EMB = self.USE_POS_EMB)
            h_prev = 2*h_dim
        self.layers['Dec_%02d'%(len(self.h_dims))] = DenseBlockClass(
            in_dim=h_prev,out_dim=self.x_dim,
            pos_emb_dim=self.pos_emb_dim,actv=self.actv,USE_POS_EMB=self.USE_POS_EMB)
        # Out
        self.layers['Out'] = DenseBlockClass(
            in_dim=2*self.x_dim,out_dim=self.x_dim,pos_emb_dim=self.pos_emb_dim,actv=self.actv,
            USE_POS_EMB = self.USE_POS_EMB)
        # Time embedding
        self.layers['Pos_Emb'] = nn.Sequential(
                SinPositionEmbeddingsClass(dim=self.pos_emb_dim),
                nn.Linear(self.pos_emb_dim,self.pos_emb_dim),
                self.actv
            )
        
    def forward(self,x,t):
        """
            Forward
            x: [B x DL]
            t: [B x 1]
        """
        self.nets = {}
        net = x # [B x DL]
        # Positional Embedding
        pos_emb = self.layers['Pos_Emb'](t) 
        
        # Encoder 
        self.enc_paths = []
        self.enc_paths.append(net)
        self.nets['x'] = net
        for h_idx in range(len(self.h_dims)+1):
            net = self.layers['Enc_%02d'%(h_idx)](net,pos_emb)
            self.enc_paths.append(net)
            self.nets['Enc_%02d'%(h_idx)] = net
        
        # Map
        net = self.layers['Map'](net,pos_emb)
        self.nets['Map'] = net # [B x z_dim]
        
        # Decoder
        self.dec_paths = []
        for h_idx in range(len(self.h_dims)+1):
            net = self.layers['Dec_%02d'%(h_idx)](net,pos_emb)
            net = torch.cat([self.enc_paths[len(self.h_dims)-h_idx],net],dim=1)
            self.dec_paths.append(net)
            self.nets['Dec_%02d'%(h_idx)] = net
        net = self.layers['Out'](net,pos_emb) # [B x DL]
        
        # RKHS projection
        if self.RKHS_projs is not None:
            RKHS_projs_exapnd = self.RKHS_projs[None,:,:,:] # [1 x D x L x L]
            RKHS_projs_tile = torch.tile(RKHS_projs_exapnd,dims=(net.shape[0],1,1,1)) # [B x D x L x L]
            net = net.reshape(-1,self.D,self.L) # [B x D x L]
            net = net[:,:,:,None]               # [B x D x L x 1]
            net = RKHS_projs_tile @ net        # [B x D x L x L] x [B x D x L x 1] => [B x D x L x 1]
            net = net.squeeze(dim=3)            # [B x D x L]
            net = net.reshape(-1,self.D*self.L) # [B x DL]
        
        # Return
        self.nets['Out'] = net  # [B x DL]
        return net


