#!/usr/bin/env python

import os
import argparse
import time
import openfermion as of
import numpy as np
import math
from pyLIQTR.PhaseEstimation.pe import PhaseEstimation
from networkx import get_node_attributes, draw, draw_networkx_edge_labels
from qca.utils.algo_utils import gsee_resource_estimation
from qca.utils.utils import circuit_estimate, GSEEMetaData
from qca.utils.hamiltonian_utils import generate_three_orbital_nx, nx_to_three_orbital_hamiltonian

## Three band

def main(args):
    t1 = args.param_t1
    t2 = args.param_t2
    t3 = args.param_t3
    t4 = args.param_t4
    t5 = args.param_t5
    t6 = args.param_t6
    t7 = args.param_t7
    t8 = args.param_t8
    mu = args.param_mu
    delta = args.param_delta

    trotter_steps = args.trotter_steps
    trotter_order = args.trotter_order

    lattice_size = args.lattice_size

    value = args.value
    repetitions = args.repetitions
    directory = args.directory
    name = args.name
    is_extrapolated = args.extrapolate

    bits_precision = estimate_bits_precision(args.error_precision)
    g = generate_three_orbital_nx(lattice_size,lattice_size)
    n_qubits = len(g)

    ham = nx_to_three_orbital_hamiltonian(g,t1,t2,t3,t4,t5,t6,t7,t8,mu,delta)

    E_min = -len(ham.terms) * max(abs(t1), abs(t2), abs(t3), abs(t4), abs(mu))
    E_max = 0
    omega = E_max-E_min
    evolution_time = 2*np.pi/omega
    phase_offset = E_max*evolution_time

    init_state = [0] * n_qubits

    gsee_args = {
        'trotterize' : True,
        'mol_ham'    : ham,
        'ev_time'    : evolution_time,
        'trot_ord'   : trotter_order,
        'trot_num'   : 1 #Accounted for in a scaling argument later
    }


    print('starting')

    metadata = GSEEMetaData(
        id=time.time_ns(),
        name=name,
        category='scientific',
        size=f'{lattice_size}x{lattice_size}',
        task='Ground State Energy Estimation',
        value_per_circuit=value,
        repetitions_per_application=repetitions,

        evolution_time=evolution_time,
        trotter_order=trotter_order,
        is_extrapolated=is_extrapolated,
        bits_precision=bits_precision,
        trotter_layers=trotter_steps,
        implementation="GSEE"
    )

    print('Estimating Circuit Resources')
    t0 = time.perf_counter()
    estimate = gsee_resource_estimation(
            outdir=directory,
            numsteps=trotter_steps,
            gsee_args=gsee_args,
            init_state=init_state,
            precision_order=1,
            phase_offset=phase_offset,
            bits_precision=bits_precision,
            circuit_name=name,
            metadata=metadata,
            write_circuits=args.circuit_write)
    t1 = time.perf_counter()
    return estimate

def estimate_bits_precision(epsilon):
    return math.ceil(math.log2(1.0/epsilon))

def parse_arguments():
    parser = argparse.ArgumentParser(prog='HTSC-three-band-RE')
    parser.add_argument('-l', '--lattice-size', type=int, default=10)
    parser.add_argument('-e', '--error-precision', type=float, default=1e-5)
    parser.add_argument('-t', '--trotter-steps', type=int, default=1)
    parser.add_argument('-to', '--trotter-order', type=int, default=2)

    parser.add_argument('-t1', '--param-t1', type=float, default=0.02)
    parser.add_argument('-t2', '--param-t2', type=float, default=0.06)
    parser.add_argument('-t3', '--param-t3', type=float, default=0.03)
    parser.add_argument('-t4', '--param-t4', type=float, default=-0.01)
    parser.add_argument('-t5', '--param-t5', type=float, default=0.2)
    parser.add_argument('-t6', '--param-t6', type=float, default=0.3)
    parser.add_argument('-t7', '--param-t7', type=float, default=-0.2)
    parser.add_argument('-t8', '--param-t8', type=float, default=0.1)
    parser.add_argument('-mu', '--param-mu', type=float, default=1)
    parser.add_argument('-delta', '--param-delta', type=float, default=0.4)

    parser.add_argument('-n', '--name', type=str, default=f'FermiHubbard-ThreeBand', help='name of this circuit instance, becomes prefix for output file')
    parser.add_argument('-d', '--directory', type=str, default='./', help='output file directory')
    parser.add_argument('-v', '--value', type=float, default=0, help='value of the total application')
    parser.add_argument('-r', '--repetitions', type=int, default=1, help='repetitions needed to achieve value of computatation (not runs of this script)')
    parser.add_argument('-c', '--circuit_write', default=False, action='store_true')
    parser.add_argument('-x', '--extrapolate', default=False, action='store_true')
    return parser

if __name__ == "__main__":
    parser = parse_arguments()
    main(parser.parse_args())
