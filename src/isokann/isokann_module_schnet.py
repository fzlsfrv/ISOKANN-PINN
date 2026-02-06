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
from torch.func import vmap, hessian
# from torch.autograd.functional import hessian

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


class ratesNN(nn.Module):
    def __init__(self, mlp_network):

        super().__init__()
        self.net = mlp_network

        self.c1_ = nn.Parameter(pt.tensor(0.5))
        self.c2_ = nn.Parameter(pt.tensor(0.5))

        self.softplus = nn.Softplus()

    
    @property
    def c1(self):

        return self.softplus(self.c1_)


    @property
    def c2(self):

        return self.softplus(self.c2_)

    
    def forward(self, x, z):

        return self.net(x, z)

        



def nabla_chi(model, x, z):
    """
    Returns d chi / d x.
    """
    x = x.requires_grad_(True)
    
    chi = model(
        z.reshape(-1).long(),
        x.reshape(-1, 3)
    )

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




def laplacian_operator(grad_chi, chi, x):

    m = grad_chi.shape[-1]
    
    laplacians=[]


    for i in range(m):

        gi = pt.autograd.grad(
                outputs=grad_chi[:, :, i], 
                inputs=x,
                grad_outputs=pt.ones_like(grad_chi[:, :, i]),
                create_graph=True,
                retain_graph=True
            )[0]
            
        laplacians.append(gi)

    
    Delta = gi.sum(dim=-1)

    return Delta




# def laplacian_operator(model, x):

#     H = hessian(model)(x)
#     H = H.squeeze(0)
#     # print(H.shape)
#     return pt.trace(H)



# def laplacian_operator(model, x):
#     x_vec = x.squeeze(0).squeeze(-1) if x.dim() > 1 else x.squeeze(-1)  
    
#     def scalar_fn(inputs):
#         out = model(inputs.unsqueeze(0))  
#         return out.squeeze(-1)
    
#     H = hessian(scalar_fn, x_vec, create_graph=True) 
#     print(H.shape)  
#     return pt.trace(H)




def generator_action(model, x, z, forces_fn, gamma, k_B, T, S=1):  
   
    chi, grad_chi = nabla_chi(model, x, z)
    # print(f"Gradient shape: {grad_chi.shape}")
    # print(f"Chi function shape {chi.shape}")

    lap_chi = laplacian_operator(grad_chi, chi, x)

    drift_term = (-(1.0 / (gamma * S)) * forces_fn * grad_chi.squeeze(-1)).sum(dim=1)

    diffusion_term = (k_B * T / (gamma * S)) * lap_chi

    L_scaled = drift_term + diffusion_term

    # print(f"Laplacian chi shape: {lap_chi.shape}")
    return chi, L_scaled



def scale_and_shift(y):
    minarr = pt.min(y)
    maxarr = pt.max(y)
    hat_y =  (y - minarr) / (maxarr - minarr)

    return hat_y



def trainNN(
            model,
            Nepochs,
            batch_size,
            coords,
            forces_fn,
            atomic_numbers,
            optimizer, 
            lam_bound,
            split=0.2,
            momentum=None,
            D=1,
            device=None, 
            T=310.15,
            gamma=1000.0,
            k_B=0.008314
            ):

    
    model.to(device)
    MSE = pt.nn.MSELoss()

    
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum = momentum, nesterov=True)
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)



    # max_force = pt.max(forces_fn.abs())
    mean_force_mag = pt.abs(forces_fn).mean().item()
    S = (mean_force_mag / gamma)

    train_ds = TensorDataset(coords, forces_fn, atomic_numbers)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    c1_vals = []
    c2_vals = []


    for epoch in range(Nepochs+1):


        epoch_loss = 0
        # permutation = pt.randperm(X_train.size()[0], device=device)
        
        for xb, fb, zb in loader:

            # Clear gradients for next training
            optimizer.zero_grad()

            xb = xb.to(device).float().requires_grad_(True)

            # mean_force = fb.mean()
            # S = (mean_force / gamma).item()

            # S = 1

            
            chi_batch, L_chi = generator_action(model, xb, zb, fb, gamma, k_B, T, S)


            c1_s = model.c1/S
            c2_s = model.c2/S


            residual = L_chi + c1_s * chi_batch.squeeze() - c2_s * (1 - chi_batch.squeeze()) 

            # reg_loss = MSE(chi_batch.squeeze(), chi_batch.squeeze() * 0) + MSE(1-chi_batch.squeeze(), (1-chi_batch.squeeze()) * 0)  

            loss_pinn = MSE(residual, pt.zeros_like(residual))

            loss = loss_pinn 


            loss.backward()
            # pt.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 125 == 0:

            print(f"epoch {epoch:3d} | loss {epoch_loss/len(loader):.6f} |")
            
        if epoch % 25 == 0:
            c1_vals.append(model.c1.item())
            c2_vals.append(model.c2.item())
        
    return c1_vals, c2_vals

             










    

