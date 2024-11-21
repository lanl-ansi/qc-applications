import os
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
from qca.utils.chemistry_utils import load_pathway, generate_electronic_hamiltonians, gsee_molecular_hamiltonian


@dataclass
class pathway_info:
    pathway: list[int]
    fname: str

def grab_arguments() -> Namespace:
    parser = ArgumentParser('Perform a sweep over different pathways of varying active spaces')
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
        '-D',
        '--dir',
        type=str,
        default=os.path.dirname(os.path.realpath(__file__)),
        help='directory to store generated resource estimates')
    parser.add_argument(
        '-BP',
        '--bits_prec',
        type=int,
        default=10,
        help='Number of bits to estimate phase to'
    )
    parser.add_argument(
        '-d', 
        '--directory', 
        type=str, 
        help='Directoty with pathway datafiles.',
        default='./data/'
    )
    args = parser.parse_args()
    return args

def trotter_subprocess(
        outdir:str,
        fname:str,
        pathway: list[int],
        basis:str,
        ev_time:float,
        active_space_reduc:float,
        trotter_order:int,
        trotter_steps:int,
        bits_precision:int
):
    coords_pathways = load_pathway(fname, pathway)
    molecular_hamiltonians = generate_electronic_hamiltonians(
        basis=basis,
        active_space_frac=active_space_reduc,
        coordinates_pathway=coords_pathways,
        run_scf=1
    )
    catalyst_name = fname.split('.xyz')[0]
    gsee_args = {
        'trotterize' : True,
        'ev_time'    : ev_time,
        'trot_ord'   : trotter_order,
        'trot_num'   : trotter_steps
    }
    gsee_molecular_hamiltonian(
        outdir=outdir,
        catalyst_name=catalyst_name,
        gse_args=gsee_args,
        trotter_steps=trotter_steps,
        bits_precision=bits_precision,
        molecular_hamiltonians=molecular_hamiltonians
    )


if __name__ == '__main__':
    pid = os.getpid()
    args = grab_arguments()
    pathway = args.pathway
    if not pathway:
        raise LookupError('Unspecified reaction pathway')
    
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = args.fname
    bits_precision = args.bits_prec
    active_space_reduc = args.active_space_reduction
    basis = args.basis
    evolution_time = args.evolution_time
    trotter_order = args.trotter_order
    trotter_steps = args.trotter_steps 
    
    trotter_subprocess(
        outdir=outdir,
        fname=fname,
        pathway=pathway,
        basis=basis,
        ev_time=evolution_time,
        active_space_reduc=active_space_reduc,
        trotter_order=trotter_order,
        trotter_steps=trotter_steps,
        bits_precision=bits_precision
    )    
