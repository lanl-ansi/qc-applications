#!/usr/bin/env python

import os
import argparse
import time
import openfermion as of
import numpy as np
import math
from pyLIQTR.PhaseEstimation.pe import PhaseEstimation
from networkx import get_node_attributes, draw, draw_networkx_edge_labels
from qca.utils.utils import circuit_estimate, EstimateMetaData
from qca.utils.hamiltonian_utils import generate_two_orbital_nx, nx_to_two_orbital_hamiltonian

## Two band

def main(args):
    t1 = args.param_t1
    t2 = args.param_t2
    t3 = args.param_t3
    t4 = args.param_t4
    mu = args.param_mu

    trotter_steps = args.trotter_steps
    trotter_order = args.trotter_order

    lattice_size = args.lattice_size

    value = args.value
    repetitions = args.repetitions
    directory = args.directory
    name = args.name

    bits_precision = estimate_bits_precision(args.error_precision)
    g = generate_two_orbital_nx(lattice_size,lattice_size)
    n_qubits = len(g)

    ham = nx_to_two_orbital_hamiltonian(g,t1,t2,t3,t4,mu)

    E_min = -len(ham.terms) * max(abs(t1), abs(t2), abs(t3), abs(t4), abs(mu))
    E_max = 0
    omega = E_max-E_min
    t = 2*np.pi/omega
    phase_offset = E_max*t

    init_state = [0] * n_qubits

    pe_args = {
        'trotterize' : True,
        'mol_ham'    : ham,
        'ev_time'    : t,
        'trot_ord'   : trotter_order,
        'trot_num'   : 1 #Accounted for in a scaling argument later
    }


    print('starting')

    metadata = EstimateMetaData(
        id=time.time_ns(),
        name=name,
        category='scientific',
        size=f'{lattice_size}x{lattice_size}',
        task='Ground State Energy Estimation',
        implementations=f'GSEE, evolution_time={t}, bits_precision={bits_precision}, trotter_order={trotter_order}',
    )

    t0 = time.perf_counter()
    gse_inst = PhaseEstimation(
        precision_order=1,
        init_state=init_state,
        phase_offset=phase_offset,
        include_classical_bits=False, # Do this so print to openqasm works
        kwargs=pe_args)
    gse_inst.generate_circuit()
    t1 = time.perf_counter()
    print(f'time to generate high level: {t1 - t0}')

    gse_circuit = gse_inst.pe_circuit

    print('Estimating Circuit Resources')
    t0 = time.perf_counter()
    estimate = circuit_estimate(
                     circuit=gse_circuit,
                     metadata = metadata,
                     outdir=directory,
                     circuit_name=name,
                     write_circuits=args.circuit_write,
                     bits_precision=bits_precision,
                     numsteps=trotter_steps)
    t1 = time.perf_counter()
    return estimate

def estimate_bits_precision(epsilon):
    return math.ceil(math.log2(1.0/epsilon))

def parse_arguments():
    parser = argparse.ArgumentParser(prog='HTSC-two-band-RE')
    parser.add_argument('-l', '--lattice-size', type=int, default=10)
    parser.add_argument('-e', '--error-precision', type=float, default=1e-3)
    parser.add_argument('-t', '--trotter-steps', type=int, default=1)
    parser.add_argument('-to', '--trotter-order', type=int, default=2)
    parser.add_argument('-t1', '--param-t1', type=float, default=-1)
    parser.add_argument('-t2', '--param-t2', type=float, default=1.3)
    parser.add_argument('-t3', '--param-t3', type=float, default=0.85)
    parser.add_argument('-t4', '--param-t4', type=float, default=0.85)
    parser.add_argument('-mu', '--param-mu', type=float, default=1)
    parser.add_argument('-n', '--name', type=str, default=f'FermiHubbard-TwoBand', help='name of this circuit instance, becomes prefix for output file')
    parser.add_argument('-d', '--directory', type=str, default='./', help='output file directory')
    parser.add_argument('-v', '--value', type=float, default=300000, help='value of the total application')
    parser.add_argument('-r', '--repetitions', type=int, default=400, help='repetitions needed to achieve value of computatation (not runs of this script)')
    parser.add_argument('-c', '--circuit_write', default=False, action='store_true')
    return parser

if __name__ == "__main__":
    parser = parse_arguments()
    main(parser.parse_args())
