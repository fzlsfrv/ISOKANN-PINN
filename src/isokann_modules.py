import numpy as np
import torch as pt
from torch.nn.functional import normalize
import scipy
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import logging
import sys
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

            
        layers = []

        for i, (dim_in, dim_out) in enumerate(zip(dims_in, dims_out)):
            layers.append(torch.nn.Linear(dim_in, dim_out))

            if i < self.Nhiddenlayers:
                layers.append(self.activation)

        self._layers = torch.nn.Sequential(*layers)
    

    def forward(self, x):
        """
            MLP forward pass
        """
        return self._layers(x)





class ratesNN(nn.Module):
    def __init__(self, mlp_network):

        super.__init__()
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

    
    def forward(self, f):

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
        gi = torch.autograd.grad(
            outputs=chi[:, i].sum(), #Sum of the derivatives = derivative of the sum (independent terms=0)
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]  # (B,inp_dim)
        grads.append(gi)

    G = torch.stack(grads, dim=2)  # (B,inp_dim, m)
    return chi, G




# def laplacian_chi(grad_chi, x):
    
#     grad_chi.unsqueeze(-1)
#     m = chi.shape[1]
#     D = x.shape[1]

#     laplacians = []

#     for i in range(m):

#         gi = torch.autograd.grad(
#             outputs=grad_chi[:,i].sum(),
#             inputs=x,
#             create_graph=True,
#             retain_graph=True
#         )[0]
#         laplacians.append(gi)
    
#     L = torch.stack(laplacians, dim=1)

#     return L




def laplacian_operator(model, x):

    H = hessian(model)(x)

    return torch.trace(H)



def generator_action(model, x, forces_fn, D):  
   
    chi, grad_chi = nabla_chi(model, x)

    # None -> model, 0 -> batch dimensions of chi
    lap_chi = vmap(laplacian_operator, in_dims=(None, 0))(model, x)  # vmap over batch
    return chi, (forces_fn * grad_chi).sum(-1) + D * lap_chi 






def trainNN(
            model,
            lr,
            wd, 
            Nepochs,
            batch_size,
            patience,
            dataset,
            forces_fn,
            optimizer, 
            potential_grad,
            kB, 
            T, 
            gamma,
            split=0.2,
            momentum=None
            ):
    
    best_loss = float('inf')
    patience_counter=0

    

    MSE = pt.nn.MSELoss()

    # optimizer = pt.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum = momentum, nesterov=True)
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    train_losses = []
    val_losses = []


    for epoch in range(Nepochs):

        permutation = pt.randperm(X_train.size()[0], device=device)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        for (xb,) in loader:

            # Clear gradients for next training
            optimizer.zero_grad()

            xb = xb.to(device).float().requires_grad_(True)

            chi_batch, L_chi = generator_action(model, xb, forces_fn, D)

            residual = L_chi + model.c1.item() * chi.squeeze() - model.c2.item() * (1 - chi.squeeze())  # (B,)
            reg_loss = mse_loss(chi.squeeze(), chi.squeeze() * 0) + mse_loss(1-chi.squeeze(), (1-chi.squeeze()) * 0)  

            
            loss = residual + 0.01 * reg_loss

            loss.backward()
            # pt.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()


             










    

