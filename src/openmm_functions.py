from openmm import*
from openmm.app import*
from openmm.unit import*
import mdtraj as md
import json
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
        print(f"Number of atoms in the system: {psf.topology.getNumAtoms()}")

        param_dir = os.path.join(base, 'toppar')

        params = CharmmParameterSet(
            os.path.join(base, 'toppar/top_all36_prot.rtf'),
            os.path.join(base, 'toppar/par_all36m_prot.prm'),
            os.path.join(base, 'toppar/top_all36_na.rtf'),
            os.path.join(base, 'toppar/par_all36_na.prm'),
            os.path.join(base, 'toppar/top_all36_carb.rtf'),
            os.path.join(base, 'toppar/par_all36_carb.prm'),
            os.path.join(base, 'toppar/top_all36_lipid.rtf'),
            os.path.join(base, 'toppar/par_all36_lipid.prm'),
            os.path.join(base, 'toppar/top_all36_cgenff.rtf'),
            os.path.join(base, 'toppar/par_all36_cgenff.prm'),
            os.path.join(base, 'toppar/toppar_all36_moreions.str'),
            os.path.join(base, 'toppar/top_interface.rtf'),
            os.path.join(base, 'toppar/par_interface.prm'),
            os.path.join(base, 'toppar/toppar_all36_nano_lig.str'),
            os.path.join(base, 'toppar/toppar_all36_nano_lig_patch.str'),
            os.path.join(base, 'toppar/toppar_all36_synthetic_polymer.str'),
            os.path.join(base, 'toppar/toppar_all36_synthetic_polymer_patch.str'),
            os.path.join(base, 'toppar/toppar_all36_polymer_solvent.str'),
            os.path.join(base, 'toppar/toppar_water_ions.str'),
            os.path.join(base, 'toppar/toppar_dum_noble_gases.str'),
            os.path.join(base, 'toppar/toppar_ions_won.str'),
            os.path.join(base, 'toppar/cam.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_arg0.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_c36m_d_aminoacids.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_fluoro_alkanes.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_heme.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_na_combined.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_retinol.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_model.str'),
            os.path.join(base, 'toppar/toppar_all36_prot_modify_res.str'),
            os.path.join(base, 'toppar/toppar_all36_na_nad_ppi.str'),
            os.path.join(base, 'toppar/toppar_all36_na_rna_modified.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_sphingo.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_archaeal.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_bacterial.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_cardiolipin.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_cholesterol.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_dag.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_inositol.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_lnp.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_lps.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_mycobacterial.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_miscellaneous.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_model.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_prot.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_tag.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_yeast.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_hmmm.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_detergent.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_ether.str'),
            os.path.join(base, 'toppar/toppar_all36_lipid_oxidized.str'),
            os.path.join(base, 'toppar/toppar_all36_carb_glycolipid.str'),
            os.path.join(base, 'toppar/toppar_all36_carb_glycopeptide.str'),
            os.path.join(base, 'toppar/toppar_all36_carb_imlab.str'),
            os.path.join(base, 'toppar/toppar_all36_label_spin.str'),
            os.path.join(base, 'toppar/toppar_all36_label_fluorophore.str'),
            os.path.join(base, '7v7/7v7.rtf'),
            os.path.join(base, '7v7/7v7.prm')
        )

        # param_files = (
        #     glob.glob(os.path.join(param_dir, '*.rtf')) +
        #     glob.glob(os.path.join(param_dir, '*.prm')) +
        #     glob.glob(os.path.join(param_dir, '*.str'))
        # )
        
        # if ligand_name != None:    
        #     ligand_param_dir = os.path.join(base, ligand_name)
        #     ligand_param_files = (
        #         glob.glob(os.path.join(ligand_param_dir, '*.rtf')) + 
        #         glob.glob(os.path.join(ligand_param_dir, '*.prm'))
        #     )
        #     param_files = param_files + ligand_param_files

        # print(len(param_files))

        # params = CharmmParameterSet(*param_files)

        #set the box
        sysinfo_file = os.path.join(base, "openmm", "sysinfo.dat")
        with open(sysinfo_file, 'r') as f:
            content = f.read().strip()
        data = json.loads(content)
        lx, ly, lz = data["dimensions"][:3]

        psf.setBox(lx*angstroms, ly*angstroms, lz*angstrom)

        #create_system
        system = psf.createSystem(params, nonbondedMethod=app.LJPME, nonbondedCutoff=1.0 *nanometer, constraints = app.HBonds)


        # Centering the solute within the periodic box before running the simulation
        # This step is not strictly required for the simulation to run correctly,
        # but without it, the periodic box may appear misaligned with the structure,
        # causing the protein (or membrane) to drift outside the visible box in trajectory files.
        # Centering improves visualization and helps ensure proper PBC wrapping in trajectory output.
        positions_check = crd.positions
        center = np.mean(positions_check.value_in_unit(nanometer), axis=0)
        box = psf.topology.getUnitCellDimensions().value_in_unit(nanometer)
        box_center = np.array(box) / 2.0
        translation = box_center - center
        centered_positions = positions_check + translation * nanometer
        centered_positions = centered_positions.value_in_unit(nanometer)

        return system, psf, centered_positions


def get_parameters(
                        system,
                        base,
                        dcd_file,
                        dcd_dir,
                        dt, #in femtoseconds
                        T,
                        gamma,
                        platform_name,
                        pdb_file='pdbfile.pdb',
                        integrator_type='Langevin',
                        psf=None,
                        get_potential_grad=False,
                        get_coords=False,
                        selection=None
                      ):
    if integrator_type=='Langevin':
        integrator = LangevinIntegrator(T*kelvin, gamma/picoseconds, dt*femtoseconds)
    
    platform = Platform.getPlatformByName(platform_name)
    top_dir = os.path.join(base, pdb_file)
    
    if psf is not None:
        simulation = Simulation(psf.topology, system, integrator, platform)
    else:
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
    

def select(
            psf,
            selection=None,
            ):
        
    md_top = md.Topology.from_openmm(psf.topology)
    indices = md_top.select(selection)

    return indices

