import os
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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
        default=10
    )
    args = parser.parse_args()
    return args

def generate_ap_re(
        catalyst_name:str,
        num_processes:int,
        hamiltonians: list,
        gsee_args:dict,
        trotter_steps:int,
        bits_precision: int
    ):
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for idx, hamiltonian in enumerate(hamiltonians):
            future = executor.submit(
                gsee_molecular_hamiltonian,
                f'pathway({idx})_{catalyst_name}',
                gsee_args,
                trotter_steps,
                bits_precision,
                hamiltonian
            )
            results.append(future)
        for future in as_completed(results):
            print(f'completed')

def grab_molecular_hamiltonians_pool(
        active_space_reduc:float,
        num_processes:int,
        pathways: list,
        basis:str
    ) -> list:
    hamiltonians = []
    results = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        for coords in pathways:
            future = executor.submit(
                generate_electronic_hamiltonians,
                basis, active_space_reduc, coords, 1
            )
            results.append(future)
        for future in as_completed(results):
            hamiltonians.append(future.result())
    return hamiltonians


if __name__ == '__main__':
    pid = os.getpid()
    args = grab_arguments()
    active_space_reduc = args.active_space_reduction
    pathways = [
        pathway_info(
            pathway=[27, 1, 14, 15, 16, 24, 25, 26],
            fname='water_oxidation_Co2O9H12.xyz'
        ),
        pathway_info(
            pathway=[3, 1, 14, 15, 16, 20, 21, 22, 23],
            fname='water_oxidation_Co2O9H12.xyz'
        ),
        pathway_info(
            pathway=[2, 1, 14, 15, 16, 17, 18, 19],
            fname='water_oxidation_Co2O9H12.xyz'
        ),
        pathway_info(
            pathway=[5, 10, 28, 29, 30, 31, 32, 33],
            fname='water_oxidation_Co2O9H12.xyz'
        )
    ]
    coords_pathways = [
        load_pathway(pathway.fname, pathway.pathway) for pathway in pathways
    ]
    molecular_hamiltonians = grab_molecular_hamiltonians_pool(
        active_space_reduc=active_space_reduc,
        num_processes=len(pathways),
        pathways=coords_pathways,
        basis='sto-3g'
    )
    gsee_args = {
        'trotterize' : True,
        'ev_time'    : 1,
        'trot_ord'   : 2,
        'trot_num'   : 1
    }
    generate_ap_re(
        catalyst_name='Co2O9H12',
        num_processes=len(pathways),
        hamiltonians=molecular_hamiltonians,
        gsee_args=gsee_args,
        trotter_steps=1,
        bits_precision=10
    )
