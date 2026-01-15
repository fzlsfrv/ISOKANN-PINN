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

            if i < len(n_units) - 2:
                layers.append(self.activation)

        self._layers = torch.nn.Sequential(*layers)
    

    def forward(self, x):
        """
            MLP forward pass
        """
        return self._layers(x)




def nabla_chi(model, x):
    """
    Returns d chi / d x.
    x:   (B,1) with requires_grad=True
    chi: (B,m)
    out: (B,m,1)  (Jacobian per sample)
    """
    x = x.requires_grad_(True)
    chi = model(x)  # (B,m)

    grads = []
    m = chi.shape[1]
    for i in range(m):
        gi = torch.autograd.grad(
            outputs=chi[:, i].sum(), #Sum of the derivatives = derivative of the sum (independent terms=0)
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]  # (B,1)
        grads.append(gi)

    G = torch.stack(grads, dim=1)  # (B,m,1)
    return G




def laplacian_chi(x):

    return x








def trainNN(
            model,
            lr,
            wd, 
            Nepochs,
            batch_size,
            momentum,
            patience,
            X,
            Y,
            potential_grad,
            split=0.2
            ):
    
    best_loss = float('inf')
    patience_counter=0

    # Split training and validation data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split, random_state=42)

    MSE = pt.nn.MSELoss()

    optimizer = pt.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum = momentum, nesterov=True)
    # optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    train_losses = []
    val_losses = []


    for epoch in range(Nepochs):

        permutation = pt.randperm(X_train.size()[0], device=device)

        for i in range(0, X_train.size()[0], batch_size):

            # Clear gradients for next training
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]

            batch_x, batch_y = X_train[indices], Y_train[indices]

            new_points  =  model(batch_x)

            new_points_grad = nabla_chi(model, new_points)

            inf_gen = -potential_grad*new_points_grad + laplacian_chi(new_points)








    

