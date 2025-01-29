import os
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
from qca.utils.chemistry_utils import load_pathway, gsee_molecular_hamiltonian, generate_molecular_hamiltonian, gen_df_qpe

from openfermion.ops.representations import InteractionOperator
@dataclass
class pathway_info:
    pathway: list[int]
    fname: str

def grab_arguments() -> Namespace:
    parser = ArgumentParser('Perform a sweep over different pathways of varying active spaces')
    parser.add_argument(
        '-DF',
        '--use_df',
        action='store_true',
        help='Flag to double Factorize encode your hamiltonian'
    )
    parser.add_argument(
        '-A',
        '--active_space_reduction',
        type=int,
        help='Factor to reduce the active space',
        default=1
    )
    parser.add_argument(
        '-F',
        '--fname',
        type=str,
        help='absolute filepath pointing to xyz file for extracting molecular hamiltonian along a reaction pathway',
        required=True
    )
    parser.add_argument(
        '-B',
        '--basis',
        type=str,
        help='basis working in',
        default='sto-3g'
    )
    parser.add_argument(
        '-T',
        '--evolution_time',
        type=float,
        help='Float representing total evolution time if approximating exp^{iHt} for phase estimation',
        default=1
    )
    parser.add_argument(
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
        '-RE',
        '--re_dir',
        type=str,
        default=f'{os.path.dirname(os.path.realpath(__file__))}/',
        help='directory to store generated resource estimates')
    parser.add_argument(
        '-BP',
        '--bits_prec',
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
        '--bits_rot',
        type=int,
        help='The number of precision bits to use for the rotation angles output by the QROM',
        default=7
    )
    parser.add_argument(
        '-DFE',
        '--df_error',
        type=float,
        help='The threshold used to throw out factors from the double factorization.',
        default=1e-3
    )
    parser.add_argument(
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

def gen_mol_hams(
        fname: str,
        basis:str,
        pathway: list[int],
        active_space_reduc: float | None,
):
    coords_pathways = load_pathway(fname, pathway)
    hams = []
    for coords in coords_pathways:
        _, charge, multi = [int(coords[0][j]) for j in range(3)]
        geometry = []
        for coord in coords[1:]:
            atom = (coord[0], tuple(coord[1]))
            geometry.append(atom)
        mol_ham = generate_molecular_hamiltonian(
            basis=basis,
            geometry=geometry,
            multiplicity=multi,
            charge=charge,
            active_space_frac=active_space_reduc
        )
        hams.append(mol_ham)
    return hams

def df_subprocess(
        outdir: str,
        circuit_name: str,
        mol_hams: list[InteractionOperator],
        bits_rot:int,
        df_error_threshold:float,
        sf_error_threshold:float,
        energy_error:float,
        df_prec:float | None,
        eps: float | None,
        gate_precision:float | None,
        use_analytical: bool = True
):
    for idx, ham in enumerate(mol_hams):
        new_name = f'{circuit_name}_{idx}'
        # is_extrapolated is false as the circuit is already constructed w/ respect to the number of bits precision
        gen_df_qpe(
            mol_ham=ham,
            use_analytical=use_analytical,
            outdir=outdir,
            fname=new_name,
            bits_rot=bits_rot,
            df_error_threshold=df_error_threshold,
            sf_error_threshold=sf_error_threshold,
            energy_error=energy_error,
            df_prec=df_prec,
            eps=eps,
            gate_precision=gate_precision,
            is_extrapolated=False
        )

def trotter_subprocess(
        outdir:str,
        circuit_name:str,
        mol_hams: list[InteractionOperator],
        ev_time:float,
        trotter_order:int,
        trotter_steps:int,
        bits_precision:int
):
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


if __name__ == '__main__':
    pid = os.getpid()
    args = grab_arguments()
    pathway = args.pathway
    if not pathway:
        raise ValueError('Pathway not specified')
    is_single_instance:bool = pathway is None

    outdir = args.re_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fname = args.fname
    use_df = args.use_df
    bits_precision = args.bits_prec
    active_space_reduc = args.active_space_reduction
    basis = args.basis
    evolution_time = args.evolution_time
    trotter_order = args.trotter_order
    trotter_steps = args.trotter_steps
    
    bits_rot = args.bits_rot
    sf_error = args.sf_error
    df_error = args.df_error
    energy_error = args.energy_error
    gate_synth_accuracy = args.gate_synth
    fname = os.path.abspath(fname)
    mol_hams = gen_mol_hams(fname, basis, pathway, active_space_reduc)
    circuit_name = os.path.basename(fname).split('.xyz')[0]

    if not use_df:
        trotter_subprocess(
            outdir=outdir,
            circuit_name=circuit_name,
            mol_hams=mol_hams,
            ev_time=evolution_time,
            trotter_order=trotter_order,
            trotter_steps=trotter_steps,
            bits_precision=bits_precision
        )
    else:
        df_subprocess(
            outdir=outdir,
            circuit_name=circuit_name,
            mol_hams=mol_hams,
            bits_rot=bits_rot,
            df_error_threshold=df_error,
            sf_error_threshold=sf_error,
            energy_error=energy_error,
            df_prec=bits_precision,
            gate_precision=gate_synth_accuracy,
            eps=None
        )
