from openmm import*
from openmm.app import*
from openmm.unit import*
import mdtraj as md

from src.useful_functions import*
import numpy as np






def setup_system(
                    base, 
                    ligand_name = None,
                    nbcutoff = 1.0,
                    from_pdb=False,
                    pdb_file=None,
                    psf_file=None,
                    crd_file=None  
                ):

    if from_pdb:

        pdb = PDBFile(os.path.join(base, pdb_file))
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

        system = forcefield.createSystem(pdb.topology, 
                                         nonbondedMethod = PME, 
                                         nonbondedCutoff = nbcutoff*nanometer,
                                         constraints = HBonds,
                                         )

        return system

    
    else:
        psf = CharmmPsfFile(os.path.join(base, psf_file))
        crd = CharmmCrdFile(os.path.join(base, crd_file))

        param_files = (
            glob.glob(os.path.join(param_dir, '*.rtf')) +
            glob.glob(os.path.join(param_dir, '*.prm')) +
            glob.glob(os.path.join(param_dir, '*.str'))
        )
        
        if ligand_name != None:    
            ligand_param_dir = os.path.join(base, ligand_name)
            ligand_param_files = (
                glob.glob(os.path.join(ligand_param_dir, '*.rtf')) + 
                glob.glob(os.path.join(ligand_param_dir, '*.prm'))
            )
            param_files = param_files + ligand_param_files

        params = CharmmParameterSet(param_files)

        #set the box
        sysinfo_file = os.path.join(base, "openmm", "sysinfo.dat")
        with open(sysinfo_file) as f:
            line = f.readline().strip()
            lx, ly, lz = map(float, line.strip())

        psf.setBox(lx*unit.angstroms, ly*unit.angstroms, lz*unit.angstrom)

        #create_system
        system = psf.createSystem(params, nonbondedMethod=app.LJPME, nonbondedCutoff=1.0 * unit.nanometer, constraints = app.HBonds)


        # Centering the solute within the periodic box before running the simulation
        # This step is not strictly required for the simulation to run correctly,
        # but without it, the periodic box may appear misaligned with the structure,
        # causing the protein (or membrane) to drift outside the visible box in trajectory files.
        # Centering improves visualization and helps ensure proper PBC wrapping in trajectory output.
        positions_check = crd.positions
        center = np.mean(positions_check.value_in_unit(unit.nanometer), axis=0)
        box = psf.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
        box_center = np.array(box) / 2.0
        translation = box_center - center
        centered_positions = positions_check + translation * unit.nanometer
        centered_positions = centered_positions.value_in_unit(unit.nanometer)

        return system, psf, centered_positions


def get_parameters(
                        system,
                        base,
                        pdb_file,
                        dcd_file,
                        dcd_dir,
                        dt, #in femtoseconds
                        T,
                        gamma,
                        platform_name,
                        integrator_type='Langevin',
                        psf=None,
                        get_potential_grad=False,
                        get_coords=False
                      ):
    if integrator_type=='Langevin':
        integrator = LangevinIntegrator(T*kelvin, gamma/picoseconds, dt*femtoseconds)
    
    platform = Platform.getPlatformByName(platform_name)
    
    if psf is not None:
        simulation = Simulation(psf.topology, system, integrator, platform)
    else:
        top_dir = os.path.join(base, pdb_file)
        pdb = PDBFile(top_dir)
        simulation = Simulation(pdb.topology, system, integrator, platform)


    traj = md.load_dcd(os.path.join(dcd_dir, dcd_file), top=top_dir)


    if get_coords:

        coords = traj.xyz

        return coords


    elif get_potential_grad:

        all_forces = []

        for frame in range(traj.n_frames):

            simulation.context.setPositions(traj.xyz[frame])
            state = simulation.context.getState(getForces=True)
            forces = state.getForces(asNumpy=True)

            all_forces.append(forces.value_in_unit(kilojoules_per_mole/nanometer))
        
        all_forces = np.array(all_forces)

        return all_forces
    




    