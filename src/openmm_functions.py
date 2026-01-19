from openmm import*
from openmm.app import*
from useful_functions import*
import numpy as np



def setup_system(
                    base, 
                    ligand_name = None, 
                    nbmethod = 'PME', 
                    nbcutoff = 1.0,  
                ):
    
    psf = CharmmPsfFile(os.path.join(base, 'step5_assembly.psf'))
    crd = CharmmCrdFile(os.path.join(base, 'step5_assembly.crd'))

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


def get_potential_grad(
                        system,
                        psf,
                        dcd_file,
                        dcd_dir,
                        system,
                        dt, #in femtoseconds
                        T,
                        gamma,
                        platform,
                        integrator_type='Langevin',
                      ):
    if integrator_type=='Langevin':
        integrator = LangevinIntegrator(T*unit.kelvin, 1/unit.picoseconds, 2.0*unit.femtoseconds)
    

    simulation = Simulation(psf.topology, system, integrator, platform)

    dcd = DCDFile(os.path.join(dcd_dir, dcd_file))

    all_forces = []

    for i, coords in enumerate(dcd.getPositions()):

        simulation.context.setPositions(coords)

        state = simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)

        all_forces.append(forces.value_in_unit(kilojoules_per_mole/nanometer))
    
    all_forces = np.array(all_forces)

    return all_forces
    


    


    