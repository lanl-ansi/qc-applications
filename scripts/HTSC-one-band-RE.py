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

    ham = of.fermi_hubbard(lattice_size, lattice_size, tunneling=tunneling, coulomb=coulomb, periodic=False) #returns an aperiodic fermionic hamiltonian

    trotter_order = 2
    trotter_steps = 1 #Using one trotter step for a strict lower bound with this method

    #this scales the circuit depth proportional to 2 ^ bits_precision
    bits_precision = estimate_bits_precision(error_precision)

    E_min = -len(ham.terms) * max(abs(tunneling), abs(coulomb))
    E_max = 0
    omega = E_max-E_min
    t = 2*np.pi/omega
    phase_offset = E_max*t

    args = {
        'trotterize' : True,
        'mol_ham'    : ham,
        'ev_time'    : t,
        'trot_ord'   : trotter_order,
        'trot_num'   : 1 #handling adjustment in resource estimate to save time - scales circuit depth linearly.
    }


    init_state = [0] * lattice_size * lattice_size * 2 #TODO: use Fock state from Hartree-Fock as initial state

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
        precision_order=1, #actual precision bits accounted as scaling factors in the resource estimate
        init_state=init_state,
        phase_offset=phase_offset,
        include_classical_bits=False, # Do this so print to openqasm works
        kwargs=args)
    gse_inst.generate_circuit()
    t1 = time.perf_counter()
    print(f'One band GSEE time to generate high level Circuit: {t1 - t0}')

    gse_circuit = gse_inst.pe_circuit

    print('Estimating one_band')
    t0 = time.perf_counter()
    estimate = circuit_estimate(
        circuit=gse_circuit,
        metadata = metadata,
        outdir=directory,
        circuit_name=name,
        write_circuits=circuit_write,
        bits_precision=bits_precision,
        numsteps=trotter_steps
    )
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
    return parser

if __name__ == "__main__":
    parser = parse_arguments()
    main(parser.parse_args())
