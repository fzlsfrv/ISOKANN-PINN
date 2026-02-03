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
from openmm import *
from openmm.app import *
from openmm.unit import *
from scipy.spatial.distance import pdist
from MDAnalysis.analysis.distances import*

sys.path.append(os.path.abspath('../'))
from src.useful_functions import*



print(" ")
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


#pt.cuda.empty_cache()

def generate_pairs(N):
    #return np.c_[np.array(np.meshgrid(np.arange(N), np.arange(N))).T.reshape(-1,2)]
    t = np.arange(0,N,1)
    return np.array(list(set(itertools.combinations(t, 2))))

def generate_PWDistances_torch(
                        inp_dir  =  'input/',
                        dcd_dir  =  'output/',
                        out_dir  =  'ISOKANN_files/',
                        pdbfile_solute    = 'pdbfile_no_water.pdb', 
                        pdbfile_water     = 'pdbfile_water.pdb',  
                        file_traj_water   = "trajectory_water.dcd",
                        file_traj_solute  = "trajectory.dcd",
                        frames     = np.array([0,1,2,3,4,5,6,7,8,9]),
                        rel_coords = np.array([[0, 70]]),
                        periodic   = False,
                        BB=False):
    
    if BB==True:
        print(" ")
        print("This will create torch files containing PW distances using only BACKBONE atoms")
        print(" ")
    else:
        print(" ")
        print("This will create torch files containing PW distances using ALL atoms")
        print(" ")
    
    # Starting points (number of files in input/initial_states)
    traj      = mda.Universe(inp_dir + pdbfile_water, dcd_dir + file_traj_water) 
    
    
    trj = md.load(dcd_dir + file_traj_water, top= inp_dir + pdbfile_water)
    
    
    ref = mda.Universe(inp_dir + pdbfile_water)
    Npoints  = traj.trajectory.n_frames
   
    print("Number of initial states:", Npoints)
    
    # _, _, files = next(os.walk(out_final_states))
    # Nfinpoints = int(( len(files) - 1 ) / Npoints)
    # print("Number of final states:", Nfinpoints)
    
    #from MDAnalysis.analysis.bat import BAT
    #R = BAT(traj)
    
    Natoms = ref.atoms.n_atoms
    print("Number of atoms:", Natoms)
    
    
    # Calculate relevant coordinate
    print("I am generating the relevant coordinate...")
    if len(rel_coords[0])==2:
        print("len(rel_coords)==2, then the relevant coordinate is a DISTANCE between 2 atoms")
        r        = np.squeeze(md.compute_distances(trj, rel_coords, periodic=periodic))
    elif len(rel_coords[0])==3:
        print("len(rel_coords)==3, then the relevant coordinate is an ANGLE between 3 atoms")
        r        = np.squeeze(md.compute_angles(trj, rel_coords, periodic=True))
    elif len(rel_coords[0])==4:
        print("len(rel_coords)==4, then the relevant coordinate is a DIHEDRAL between 4 atoms")
        r        = np.squeeze(md.compute_dihedrals(trj, rel_coords, periodic=True))
        
    np.savetxt(out_dir + 'R0_1.txt', r)
    
    print(" ")
    
    if BB == True:
    
        u = mda.Universe(inp_dir + pdbfile_water)
        calphas = u.select_atoms("name CA")
        n_calpha = calphas.n_atoms
        bb_idx = calphas.indices
        
        lig = u.select_atoms("resname 7V7 and not name H*") 
        n_lig = lig.n_atoms
        kb8_idx = lig.indices
        sel_idx = np.unique(np.concatenate([bb_idx, kb8_idx]))
        
        
        
        #coor = pdb_numpy.Coor(inp_dir + pdbfile_water)
        #bb_idx = coor.get_index_select("protein and name CA")
        #kb8_idx = coor.get_index_select("resname KB8")        
        #sel_idx = np.unique(np.concatenate([bb_idx, kb8_idx]))

        
        pairs   =   generate_pairs(len(sel_idx))
        
        Ndims   =   len(pairs)
        print("Number of backbone atoms + ligand:", len(sel_idx))
        print("Number of pairwise distances between BACKBONE+LIGAND atoms:", Ndims)
    
        # Load initial states
        print("I am creating the tensor with the initial states...")
        #d0  =  md.compute_distances(traj.atom_slice(bb), pairs, periodic=False)

        d0 = np.zeros((Ndims,5000))
        for i in tqdm(range(5000)):
            traj.trajectory[i]
            atoms   =  traj.atoms[sel_idx]
            atom_coords = atoms.positions
            box_ = traj.trajectory.ts.dimensions
            d0[:, int(i)] = self_distance_array(atom_coords, box = box_)
        
        
        d0 = np.transpose(d0)
        D0  =  pt.tensor(d0, dtype=pt.float32, device=device)
      
        
    else:
        pairs   =   generate_pairs(Natoms)
        Ndims   =   len(pairs)
        print("Number of pairwise distances between ALL atoms:", Ndims)
            
        d0 = np.zeros((Ndims,int(Npoints)))
        for i in tqdm(range(Npoints)):
            traj.trajectory[i]
            atoms   =  traj.atoms[sel_idx]
            atom_coords = atoms.positions
            box_ = traj.trajectory.ts.dimensions
            d0[:, int(i)] = self_distance_array(atom_coords, box = box_)
        
        
        d0 = np.transpose(d0)
        D0  =  pt.tensor(d0, dtype=pt.float32, device=device)
    
    
    print(f"Tensor saved to: {os.path.join(out_dir)}")
    pt.save(D0, out_dir + 'PWDistances_0.pt')
    
    
    print('Shape of D0?')
    print('Npoints, Ndims')
    print(D0.shape)
    
    
