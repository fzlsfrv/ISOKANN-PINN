import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor
from utils import MLP
from schnet_utils import RadiusInteractionGraph, InteractionBlock, ShiftedSoftplus


class RadialBasisExpansion(torch.nn.Module):
    def __init__(self, start, stop, num_gaussians, gamma: float):
        super().__init__()
        self.gamma = gamma 
        self.num_gaussians = num_gaussians

        self.register_buffer('mu', torch.linspace(start, stop, num_gaussians))

    def expand(self, dist):
        
        if dist.dim() == 0:
            dist = dist.unsqueeze(0)

        self.rbf = torch.exp(-self.gamma * (dist.view(-1, 1) - self.mu)**2)

        return self.rbf
        

class SchNet(torch.nn.Module):
    """
    Args:
        dim_embedding (int, optional): Embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
    """

    def __init__(
        self,
        dim_embedding: int = 32,
        num_filters: int = 32,
        num_interactions: int = 2,
        num_gaussians: int = 10,
        cutoff: float = 5.0,
    ):
        super().__init__()

        self.dim_embedding = dim_embedding
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = aggr_resolver("add")

        # Embedding Layer
        self.embedding = Linear(1, dim_embedding)

        # Interaction graph layer
        self.interaction_graph = RadiusInteractionGraph(self.cutoff)

        # Radial basis expansion layer
        self.distance_expansion = RadialBasisExpansion(0, self.cutoff,  self.num_gaussians, gamma=4.0)

        # Interaction
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                dim_embedding, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # Readout
        self.lin1 = Linear(dim_embedding, dim_embedding // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(dim_embedding // 2, 1)

    def forward(self, z: torch.tensor, pos: torch.tensor, batch_dimensions: torch.Tensor):
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_samples * num_atoms, 1]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_samples * num_atoms, 3]`.
            batch (torch.Tensor): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_samples * num_atoms]`.
        """

        # Embedding of each atomic charge (only one in our case)
        h = self.embedding(z)

        # Interaction graph from atomic positions
        edges, distances = self.interaction_graph(
                                                    pos.reshape(-1, 3), batch_dimensions
                                                 )


        # expand distances in radial basis functions
        radial_expansion = self.distance_expansion.expand(distances)

        for interaction in self.interactions:
            h = h + interaction(h, edges, distances, radial_expansion)

        # Readout
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        out = self.readout(h, batch_dimensions, dim=0)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dim_embedding={self.dim_embedding}, "
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})"
        )


