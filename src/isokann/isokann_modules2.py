import numpy as np
import torch as pt
from torch.nn.functional import normalize
import scipy
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import logging
import sys

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.func import hessian, vmap

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


class MLP(pt.nn.Module):

    def __init__(self, Nodes, enforce_positive=0, act_fun='sigmoid', LeakyReLU_par=0.01):

        super(MLP, self).__init__()

        self.input_size    = Nodes[0]
        self.output_size   = Nodes[-1]
        self.Nhiddenlayers = len(Nodes)-2
        self.Nodes         = Nodes

        dims_in = Nodes[:-1]
        dims_out = Nodes[1:]

        if act_fun == 'sigmoid':
            self.activation  = pt.nn.Sigmoid()  # #
        elif act_fun == 'relu':
            self.activation  = pt.nn.ReLU()
        elif act_fun == 'leakyrelu': 
            self.activation  = pt.nn.LeakyReLU(LeakyReLU_par)
        elif act_fun == 'gelu': 
            self.activation  = pt.nn.GELU()
        elif act_fun == 'tanh':
            self.activation = pt.nn.Tanh()
        elif act_fun == 'softplus':
            self.activation = pt.nn.Softplus()

            
        layers = []

        for i, (dim_in, dim_out) in enumerate(zip(dims_in, dims_out)):
            layers.append(pt.nn.Linear(dim_in, dim_out))

            if i < self.Nhiddenlayers:
                layers.append(self.activation)

        self._layers = pt.nn.Sequential(*layers)
    

    def forward(self, x):
        """
            MLP forward pass
        """
        return self._layers(x)





class ratesNN(nn.Module):
    def __init__(self, mlp_network):

        super().__init__()
        self.net = mlp_network

        self.c1_ = nn.Parameter(pt.tensor(-2.0))
        self.c2_ = nn.Parameter(pt.tensor(-2.0))

        self.softplus = nn.Softplus()

    
    @property
    def c1(self):

        return self.softplus(self.c1_)


    @property
    def c2(self):

        return self.softplus(self.c2_)

    
    def forward(self, x):

        return self.net(x)

        



def nabla_chi(model, x):
    """
    Returns d chi / d x.
    x:   (B,1) with requires_grad=True
    chi: (B,m)
    out: (B,m,1)  (Jacobian per sample)
    """
    x = x.requires_grad_(True)
    
    chi = model(x)

    grads = []
    m = chi.shape[1]
    for i in range(m):
        gi = pt.autograd.grad(
            outputs=chi[:, i].sum(), #Sum of the derivatives = derivative of the sum (independent terms=0)
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]  # (B,inp_dim)
        grads.append(gi)

    G = pt.stack(grads, dim=2)  # (B,inp_dim, m)
    return chi, G





def laplacian_operator(model, x, h_val):
    
    h = pt.zeros_like(x) + h_val

    chi_n = model(x)
    # print(chi_n.shape)
    chi_plus = model(x + h)
    # print(chi_plus.shape)
    chi_minus = model(x - h)

    lap_chi = (chi_minus + chi_plus - 2*chi_n)/h_val**2
    

    return lap_chi




def generator_action(model, x, forces_fn, D, h):  
   
    chi, grad_chi = nabla_chi(model, x)
    # print(f"Gradient shape: {grad_chi.shape}")
    # print(f"Chi function shape {chi.shape}")

    # None -> model, 0 -> batch dimensions of chi
    lap_chi = vmap(laplacian_operator, in_dims=(None, 0, None))(model, x, h)  # vmap over batch
    lap_chi = lap_chi.sum(dim=0)
    # print(lap_chi)


    # print(f"Laplacian chi shape: {lap_chi.shape}")
    return chi, (-0.4*forces_fn * grad_chi.squeeze(-1)).sum(dim=1) + D * lap_chi 




def trainNN(
            model,
            Nepochs,
            batch_size,
            coords,
            forces_fn,
            optimizer, 
            lam_bound,
            split=0.2,
            momentum=None,
            D=1,
            device=None
            ):

    
    model.to(device)
    MSE = pt.nn.MSELoss()

    
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum = momentum, nesterov=True)
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    train_ds = TensorDataset(coords, forces_fn)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    

    for epoch in range(Nepochs+1):


        epoch_loss = 0
        # permutation = pt.randperm(X_train.size()[0], device=device)

       
        
        for xb, fb in loader:

            # Clear gradients for next training
            optimizer.zero_grad()

            xb = xb.to(device).float().requires_grad_(True)

            chi_batch, L_chi = generator_action(model, xb, fb, D, h=1e-1)

            residual = L_chi + model.c1 * chi_batch.squeeze() - model.c2 * (1 - chi_batch.squeeze())  # (B,)
            # reg_loss = MSE(chi_batch.squeeze(), chi_batch.squeeze() * 0) + MSE(1-chi_batch.squeeze(), (1-chi_batch.squeeze()) * 0)  

            loss_pinn = pt.mean(residual**2)
            loss = loss_pinn 
            # + lam_bound * reg_loss

            loss.backward()
            # pt.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 125 == 0:

            print(f"epoch {epoch:3d} | loss {epoch_loss/len(loader):.6f} |")


             










    

