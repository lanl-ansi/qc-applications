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
from qca.utils.hamiltonian_utils import generate_two_orbital_nx, nx_to_two_orbital_hamiltonian

def main(args):
    lattice_size = args.lattice_size
    tunneling = args.param_t1
    coulomb = args.param_mu

    error_precision = args.error_precision
    trotter_steps = args.trotter_steps
    trotter_order = args.trotter_order
    name = args.name
    directory = args.directory
    value = args.value
    repetitions = args.repetitions
    circuit_write = args.circuit_write
    is_extrapolated= args.extrapolate

    ham = of.fermi_hubbard(lattice_size, lattice_size, tunneling=tunneling, coulomb=coulomb, periodic=False) #returns an aperiodic fermionic hamiltonian

    #TODO: Fix this Hardcoding
    trotter_order = 2
    trotter_steps = 1 #Using one trotter step for a strict lower bound with this method

    #this scales the circuit depth proportional to 2 ^ bits_precision
    bits_precision = estimate_bits_precision(error_precision)

    E_min = -len(ham.terms) * max(abs(tunneling), abs(coulomb))
    E_max = 0
    omega = E_max-E_min
    evolution_time = 2*np.pi/omega
    phase_offset = E_max*evolution_time

    gsee_args = {
        'trotterize' : True,
        'mol_ham'    : ham,
        'ev_time'    : evolution_time,
        'trot_ord'   : trotter_order,
        'trot_num'   : 1 #handling adjustment in resource estimate to save time - scales circuit depth linearly.
    }


    init_state = [0] * lattice_size * lattice_size * 2 #TODO: use Fock state from Hartree-Fock as initial state

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

    print('Estimating one_band')
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
            is_extrapolated=is_extrapolated,
            write_circuits=args.circuit_write)
    t1 = time.perf_counter()
    print(f'Time to estimate one_band: {t1-t0}')
    return estimate

def estimate_bits_precision(epsilon):
    return math.ceil(math.log2(1.0/epsilon))

def parse_arguments():
    parser = argparse.ArgumentParser(prog='HTSC-two-band-RE')
    parser.add_argument('-l', '--lattice-size', type=int, default=20)
    parser.add_argument('-e', '--error-precision', type=float, default=1e-5)
    parser.add_argument('-t', '--trotter-steps', type=int, default=1)
    parser.add_argument('-to', '--trotter-order', type=int, default=2)
    parser.add_argument('-t1', '--param-t1', type=float, default=-1)
    parser.add_argument('-mu', '--param-mu', type=float, default=1)
    parser.add_argument('-n', '--name', type=str, default=f'FermiHubbard-OneBand', help='name of this circuit instance, becomes prefix for output file')
    parser.add_argument('-d', '--directory', type=str, default='./', help='output file directory')
    parser.add_argument('-v', '--value', type=float, default=0, help='value of the total application')
    parser.add_argument('-r', '--repetitions', type=int, default=1, help='repetitions needed to achieve value of computatation (not runs of this script)')
    parser.add_argument('-c', '--circuit_write', default=False, action='store_true')
    parser.add_argument('-x', '--extrapolate', default=False, action='store_true')
    return parser

if __name__ == "__main__":
    parser = parse_arguments()
    main(parser.parse_args())
