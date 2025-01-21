
import time
from qca.utils.algo_utils import estimate_trotter, gsee_resource_estimation
from qca.utils.hamiltonian_utils import (
    flatten_nx_graph,
    nx_heisenberg_terms,
    nx_triangle_lattice,
    pyliqtr_hamiltonian_to_openfermion_qubit_operator
)
from qca.utils.utils import GSEEMetaData, TrotterMetaData

from pyLIQTR.utils.Hamiltonian import Hamiltonian as pyH
from argparse import ArgumentParser

import numpy as np

from networkx import (
    Graph,
)

def gen_king_graph(lattice_size: int, j1: str, j2: str) -> Graph:
    '''
    Given some lattice size, nearest-neighbor interaction, and next-nearest-neighbor interaction,
    we define a graph that corresponds to the J1-J2 Heisenberg model.
    '''
    graph = Graph()

    for i in range(lattice_size):
        for j in range(lattice_size):
            graph.add_node((i,j))

    for i in range(lattice_size):
        for j in range(lattice_size):
            directions = [(0,1), (1,1), (1,0), (1,-1)]
            for di, dj in directions:
                neighbor_i = i + di
                neighbor_j = j + dj
                if 0 <= neighbor_i < lattice_size and 0 <= neighbor_j < lattice_size:
                    if abs(di) + abs(dj) == 1:
                        weight = j1
                    if abs(di) + abs(dj) == 2:
                        weight = j2
                    graph.add_edge((i,j), (neighbor_i, neighbor_j), weight=weight)
    
    return graph

def generate_square_heisenberg_hamiltonian(lattice_size:int, j1: float, j2: float):
    '''
    Given some lattice size and j1 and j2 arguments, we will generate its corresponding
    Hamiltonian. 
    '''
    graph = gen_king_graph(lattice_size, j1, j2)
    flat_graph = flatten_nx_graph(graph)
    H = nx_heisenberg_terms(flat_graph)
    return H

def assign_heisenberg_triangular_labels(graph: Graph, lattice_size:int, j1:float, j2:float) -> None:
    for i in range(lattice_size - 1):
        for j in range(lattice_size - 1):
            graph[(i,j)][(i+1,j)]['weight'] = j1
            graph[(i,j)][(i,j+1)]['weight'] = j1
            graph[(i,j)][i+1,j+1]['weight'] = j2
        graph[(i,lattice_size-1)][(i+1,lattice_size-1)]['weight'] = j1
    for j in range(lattice_size - 1):
        graph[(lattice_size-1,j)][(lattice_size-1,j+1)]['weight'] = j1

def generate_heisenberg_triangular_hamiltonian(lattice_size:int, j1:float, j2:float):
    graph = nx_triangle_lattice(lattice_size)
    assign_heisenberg_triangular_labels(graph, lattice_size, j1, j2)
    flat_graph = flatten_nx_graph(graph)
    H = nx_heisenberg_terms(flat_graph)
    return H

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-l', '--lattice_size', type=int, help='Lattice size')
    parser.add_argument('-BP', '--bits_prec', type=int, help='Number of bits to estimate phase to')
    parser.add_argument('-TO', '--trotter_order', type=int, default=2, help='Trotter order for trotterized subprocess QPE')
    parser.add_argument('-TS', '--trotter_steps', type=int, default=1, help='Number of trotter steps for trotterized subprocess QPE')
    parser.add_argument('-P', '--energy_precision', type=float, default=1e-3)
    parser.add_argument('-J1', type=float, default=1)
    parser.add_argument('-J2', type=float, default=1/2)
    parser.add_argument('-O', '--outdir', default='./', type=str)
    parser.add_argument('-N', '--name', default='ExoticPhases', type=str)
    parser.add_argument('-e', '--evolution_time', type=float, default=100)
    parser.add_argument('-R', '--reps', type=int)
    return parser.parse_args()

def gsee_exotic_phases(args, use_square:bool, system_val:float):
    lattice_size = args.lattice_size
    j1 = args.J1
    j2 = args.J2
    trotter_order = args.trotter_order
    trotter_steps = args.trotter_steps
    shape = 'square' if use_square else 'triangular'
    name = f'{args.name}_{shape}_{lattice_size}'
    bits_prec = args.bits_prec
    outdir = args.outdir

    
    
    heisenberg = generate_square_heisenberg_hamiltonian(lattice_size, j1, j2) if use_square else generate_heisenberg_triangular_hamiltonian(lattice_size, j1, j2)
    pyliqtr_heisenberg = pyH(heisenberg)
    qubit_op_heisenberg = pyliqtr_hamiltonian_to_openfermion_qubit_operator(
        pyliqtr_heisenberg
    )

    E_min = -len(qubit_op_heisenberg.terms)
    E_max = 0
    omega = E_max-E_min
    t = 2*np.pi/omega
    phase_offset = E_max*t
    bits_prec = args.bits_prec
    init_state = [0] * lattice_size *lattice_size * 2

    metadata = GSEEMetaData(
        id = time.time_ns(),
        name=f'{shape} Heisenberg - Exotic Phases GSEE',
        task='ground_state_energy_estimation',
        category='scientific',
        size=f'{lattice_size}x{lattice_size}',
        evolution_time=t,
        bits_precision=bits_prec,
        trotter_order=trotter_order,
        nsteps=trotter_steps,
        is_extrapolated=True,
        gate_synth_accuracy=1e-10,
        value=system_val,
        repetitions_per_application=100
    )

    gsee_args = {
        'trotterize' : True,
        'mol_ham'    : qubit_op_heisenberg,
        'ev_time'    : t,
        'trot_ord'   : trotter_order,
        'trot_num'   : trotter_steps
    }
    gsee_resource_estimation(
        outdir=outdir,
        nsteps=trotter_steps,
        gsee_args=gsee_args,
        init_state=init_state,
        precision_order=1,
        bits_precision=bits_prec,
        phase_offset=phase_offset,
        is_extrapolated=True,
        use_analytical=True,
        metadata=metadata,
        circuit_name=name,
    )

def dynamics(args, system_val:float):
    lattice_size = args.lattice_size
    j1 = args.J1
    j2 = args.J2
    evolution_time = args.evolution_time
    energy_precision = args.energy_precision
    trotter_steps = args.trotter_steps
    trotter_order = args.trotter_order
    name = f'{args.name}_triangular_dynamics_{lattice_size}'
    outdir = args.outdir

    triangular_heisenberg = generate_heisenberg_triangular_hamiltonian(lattice_size, j1, j2)
    pyliqtr_heisenberg = pyH(triangular_heisenberg)
    qubit_op_heisenberg = pyliqtr_hamiltonian_to_openfermion_qubit_operator(
        pyliqtr_heisenberg
    )

    metadata = TrotterMetaData(
        id=time.time_ns(),
        name='Triangular Heisenberg - Exotic Phases Dynamics',
        category='scientific',
        size=f'{lattice_size}x{lattice_size}',
        task='time_dependent_dynamics',
        evolution_time=evolution_time,
        trotter_order=trotter_order,
        energy_precision=energy_precision,
        nsteps=trotter_steps,
        is_extrapolated=True,
        gate_synth_accuracy=1e-10,
        value=system_val,
        repetitions_per_application=200
    )

    _ = estimate_trotter(
        openfermion_hamiltonian=qubit_op_heisenberg,
        evolution_time=evolution_time,
        energy_precision=energy_precision,
        outdir=outdir,
        hamiltonian_name=name,
        nsteps=trotter_steps,
        is_extrapolated=True,
        use_analytical=True,
        metadata=metadata
    )

if __name__ == '__main__':
    args = get_args()
    app_val = 750000
    target_size = 100*100
    reps = args.reps

    qb_cost = (app_val/reps)/target_size
    system_val = qb_cost*(args.lattice_size*args.lattice_size)

    gsee_exotic_phases(args, use_square=True, system_val=system_val)
    gsee_exotic_phases(args, use_square=False, system_val=system_val)
    dynamics(args, system_val)