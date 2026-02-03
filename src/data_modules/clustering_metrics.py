import MDAnalysis as mda
import os
import mdtraj as md
import numpy as np
import sympy as sp
import torch as pt
from tqdm import tqdm
import scipy
import itertools
import matplotlib.pyplot as plt
import glob

from scipy.spatial.distance import pdist
from MDAnalysis.analysis.distances import*
from MDAnalysis.analysis.rms import RMSD

import mdtraj as md


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def calculate_distances(ligand, AA, inp_dir, pdb_file, traj_dir, traj_file):


    traj = mda.Universe(os.path.join(inp_dir, pdb_file), os.path.join(traj_dir, traj_file))
    distances = []


    for i in range(len(traj.trajectory)):
        traj.trajectory[i]
        box_ = traj.trajectory.ts.dimensions
        dist = distance_array(
                                traj.select_atoms(AA).positions, 
                                traj.select_atoms(ligand).positions,
                                box = box_
                                )
        distances.append(dist)


    distances = np.array(distances)

    D0 = np.squeeze(distances, -1)

    return D0




#==============================================================================================

def calculate_angles(
                                    ligand_C, 
                                    ligand_F,
                                    CA_280,
                                    CA_279, 
                                    inp_dir, 
                                    pdb_file, 
                                    traj_dir, 
                                    traj_file, 
                                    distances_dir, 
                                    out_file
                                  ):

    traj = md.load(os.path.join(traj_dir, traj_file), top=os.path.join(inp_dir, pdb_file))

    idx_C = traj.top.select(ligand_C)[0]

    idx_F = traj.top.select(ligand_F)[0]
    
    idx_CA = traj.top.select(CA_280)[0]

    idx_CA_2 = traj.top.select(CA_279)[0]

    angles = md.compute_dihedrals(traj, np.array([[idx_CA_2, idx_CA, idx_C, idx_F]]))

    angles_ = angles[:, 0]

    D0 = calculate_distances(ligand_C, CA_280, inp_dir, pdb_file, traj_dir, traj_file)

    X = np.column_stack([D0, angles_])

    X0 = pt.tensor(X, dtype=pt.float32, device=device)

    print("Feature tensor shape:", X0.shape)

    pt.save(X0, os.path.join(distances_dir, out_file))

    print("Saved to:", os.path.join(distances_dir, out_file))




#==============================================================================================


def get_coords(ligand_F, inp_dir, pdb_file, traj_dir, traj_file, distances_dir, out_file):

    traj = mda.Universe(os.path.join(inp_dir, pdb_file), os.path.join(traj_dir, traj_file))

    F_atoms = traj.select_atoms(ligand_F)

    all_coords = []

    for ts in traj.trajectory:
        all_coords.append(F_atoms.positions)

    all_coords = np.array(all_coords)

    X0 = pt.tensor(all_coords, dtype=pt.float32, device=device)

    print("Feature tensor shape:", X0.shape)

    pt.save(X0, os.path.join(distances_dir, out_file))

    print("Saved to:", os.path.join(distances_dir, out_file))







 



