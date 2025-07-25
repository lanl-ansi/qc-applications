# === Standard Library Imports ===
import os
from typing import List, Tuple
from argparse import ArgumentParser, Namespace

# === Third Party Imports ===
from openfermion.ops.representations import InteractionOperator

# === Project-Specific Imports ===
from qca.utils.chemistry_utils import load_pathway, gsee_molecular_hamiltonian, generate_molecular_hamiltonian, gen_double_factorization_qpe
from qca.utils.utils import CatalystMetaData, grab_circuit_resources

# === Script-Specific Structs / Data Classes ===
from dataclasses import dataclass
@dataclass
class pathway_info:
    pathway: list[int]
    filename: str

# === Helper Functions ===
#TODO: Rename to differentiate from qca.utils.chemistry_utils.generate_molecular_hamiltonian
def generate_molecular_hamiltonians(
        fname: str,
        basis:str,
        pathway: list[int],
        active_space_reduc: float | None,
):
    '''
    Loads the molecules in a reaction pathway from the specified .xyz file and generates an array of the 
    respective molecular Hamiltonians. 
    '''
    coords_pathways = load_pathway(fname, pathway)
    hams = []
    mol_metadata = []

    for coords in coords_pathways:
        _, charge, multi = [int(coords[0][j]) for j in range(3)]
        geometry = []

        for coord in coords[1:]:
            atom = (coord[0], tuple(coord[1]))
            geometry.append(atom)
        mol_ham, num_electrons, num_orbitals = generate_molecular_hamiltonian(
            basis=basis,
            geometry=geometry,
            multiplicity=multi,
            charge=charge,
            active_space_frac=active_space_reduc
        )
        hams.append(mol_ham)
        mol_metadata.append((num_electrons, num_orbitals))
    return hams, mol_metadata

def qubitization_subprocess(
        outdir: str,
        circuit_name: str,
        mol_hams: list[InteractionOperator],
        mol_metadata: List[Tuple[int, int]],
        active_space_frac:int,
        basis:str,
        rotation_bits_precision:int,
        double_factorization_error_threshold:float,
        sf_error_threshold:float,
        energy_error:float,
        double_factorization_precision:float | None,
        eps: float | None,
        gate_precision:float | None,
        use_analytical: bool = True
):
    """
    Generates and estimates the resources of a Qubitized Quantum Phase Estimation (QPE) circuit 
    using the double-factorized block-encoding to estimate the ground state energy (GSEE)
    """
    for idx, ham in enumerate(mol_hams):
        new_name = f'{circuit_name}_{idx}'
        # is_extrapolated is false as the circuit is already constructed w/ respect to the number of bits precision
        num_electrons, num_orbitals = mol_metadata[idx]

        metadata = CatalystMetaData(
            id = idx,
            name = new_name,
            category='scientific',
            size = 'see metadata',
            task='Qubitized Phase Estimation with a Double-Factorized Encoding',

            basis = basis,
            num_electrons=num_electrons,
            num_orbitals= num_orbitals,
            active_space_fraction=active_space_frac)
        
        #Construct the Qubitized QPE Circuit
        qpe_circuit = gen_double_factorization_qpe(
            mol_ham=ham,
            use_analytical=use_analytical,
            bits_rot=rotation_bits_precision,
            df_error_threshold=double_factorization_error_threshold,
            sf_error_threshold=sf_error_threshold,
            energy_error=energy_error,
            df_prec=double_factorization_precision,
            eps=eps
        )

        #Estimate the Resources
        grab_circuit_resources(
            circuit=qpe_circuit,
            outdir=outdir,
            algo_name='DF_QPE',
            fname=new_name,
            is_extrapolated=False,
            use_analytical=use_analytical,
            bits_precision=double_factorization_precision,
            metadata=metadata,
            include_nested_resources=False,
            gate_synth_accuracy=gate_precision,
            write_circuits=False
        )


def trotterization_subprocess(
        outdir:str,
        circuit_name:str,
        mol_hams: list[InteractionOperator],
        ev_time:float,
        trotter_order:int,
        trotter_steps:int,
        bits_precision:int
):
    '''
    (DEPRECIATED)
    Generates and estimates the resources of a Trotterized Quantum Phase Estimation (QPE) Circuit. 
    '''

    gsee_args = {
        'trotterize' : True,
        'ev_time'    : ev_time,
        'trot_ord'   : trotter_order,
        'trot_num'   : trotter_steps
    }


    gsee_molecular_hamiltonian(
        outdir=outdir,
        catalyst_name=circuit_name,
        gse_args=gsee_args,
        trotter_steps=trotter_steps,
        bits_precision=bits_precision,
        molecular_hamiltonians=mol_hams
    )

def grab_arguments() -> Namespace:
    parser = ArgumentParser('Perform a sweep over different pathways of varying active spaces')

    #File I/O Args
    parser.add_argument(
        '-F',
        '--filename',
        type=str,
        help='absolute filepath pointing to xyz file for extracting molecular hamiltonian along a reaction pathway',
        required=True
    )
    parser.add_argument(
        '-RE',
        '--outdir',
        type=str,
        default=f'{os.path.dirname(os.path.realpath(__file__))}/',
        help='directory to store generated resource estimates')
    
    parser.add_argument(
        '-Q',
        '--use_qubitization',
        action='store_true',
        help='Flag to use the qubitized double Factorization encoding your Hamiltonian'
    )
    parser.add_argument(
        '-A',
        '--active_space_reduction',
        type=int,
        help='Factor to reduce the active space',
        default=1
    )
    parser.add_argument(
        '-B',
        '--basis',
        type=str,
        help='basis working in',
        default='sto-3g'
    )
    parser.add_argument( #needed for trotterization
        '-T',
        '--evolution_time',
        type=float,
        help='Float representing total evolution time if approximating exp^{iHt} for phase estimation',
        default=1
    )
    parser.add_argument( #needed for trotterization
        '-O',
        '--trotter_order',
        type=int,
        help='specify trotter order if using a trotter subprocess for phase estimation',
        default=2
    )
    parser.add_argument(
        '-S',
        '--trotter_steps',
        type=int,
        help='Number of trotter steps if using a trotter subprocess for phase estimation',
        default=1
    )
    parser.add_argument(
        '-P',
        '--pathway',
        metavar='N',
        type=int,
        nargs='*',
        help='reaction pathway of interest'
    )
    parser.add_argument(
        '-BP',
        '--bits_precision',
        type=int,
        default=10,
        help='Number of bits to estimate phase to'
    )
    parser.add_argument(
        '-GP',
        '--gate_synth',
        type=float,
        help='Accuracy used when decomposing circuits',
        default=1e-10
    )
    parser.add_argument(
        '-BR',
        '--rotation_bits_precision',
        type=int,
        help='The number of precision bits to use for the rotation angles output by the QROM',
        default=7
    )
    parser.add_argument(
        '-DFE',
        '--double_factorization_error',
        type=float,
        help='The threshold used to throw out factors from the double factorization.',
        default=1e-3
    )
    parser.add_argument( #NOTE IS SF SINGLE FACTORIZATION?
        '-SFE',
        '--sf_error',
        type=float,
        help='The threshold used to throw out factors from the first eigendecomposition',
        default=1e-8
    )
    parser.add_argument(
        '-EE',
        '--energy_error',
        type=float,
        help='The allowable error in phase estimation energy',
        default=1e-3
    )

    args = parser.parse_args()
    return args

# === main() ===
def main(args:Namespace):
    pid = os.getpid()
    
    args = grab_arguments()

    pathway = args.pathway
    if not pathway:
        raise ValueError('Pathway not specified')
    
    is_single_instance:bool = pathway is None

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #Extract the arguments
    fname = args.filename
    use_qubitization = args.use_qubitization
    bits_precision = args.bits_precision
    active_space_reduction = args.active_space_reduction
    basis = args.basis
    evolution_time = args.evolution_time
    trotter_order = args.trotter_order
    trotter_steps = args.trotter_steps
    
    rotation_bits_precision = args.rotation_bits_precision
    sf_error = args.sf_error
    df_error = args.double_factorization_error
    energy_error = args.energy_error
    gate_synth_accuracy = args.gate_synth
    fname = os.path.abspath(fname)

    #Generate the Molecular Hamiltonians
    mol_hams, mol_metadata = generate_molecular_hamiltonians(fname, basis, pathway, active_space_reduction)

    circuit_name = os.path.basename(fname).split('.xyz')[0]

    #Determine which algorithm to use and then generate the respective circuit and resource estimate
    if use_qubitization:

        qubitization_subprocess(
            outdir=outdir,
            circuit_name=circuit_name,
            mol_hams=mol_hams,
            mol_metadata=mol_metadata,
            active_space_frac=active_space_reduction,
            basis=basis,
            rotation_bits_precision=rotation_bits_precision,
            double_factorization_error_threshold=df_error,
            sf_error_threshold=sf_error,
            energy_error=energy_error,
            double_factorization_precision=bits_precision,
            gate_precision=gate_synth_accuracy,
            eps=None
        )
    else:
        trotterization_subprocess(
            outdir=outdir,
            circuit_name=circuit_name,
            mol_hams=mol_hams,
            ev_time=evolution_time,
            trotter_order=trotter_order,
            trotter_steps=trotter_steps,
            bits_precision=bits_precision
        )

# === Entrypoint ===
if __name__ == '__main__':
    args = grab_arguments()
    main(args)
