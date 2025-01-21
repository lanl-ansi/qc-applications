#!/usr/bin/env python
import argparse
import math
import time
import numpy as np

from qca.utils.algo_utils import gsee_resource_estimation
from qca.utils.utils import GSEEMetaData
from qca.utils.hamiltonian_utils import tavis_cummings_model_qubit_hamiltonian

def main(args):
    n_s = args.param_n_s
    n_b = args.param_n_b
    omega_c = args.param_omega_c
    omega_o = args.param_omega_o
    lam = args.param_lambda 

    bits_precision_tavis_cummings = estimate_bits_precision(args.error_precision)
    trotter_order_tavis_cummings = args.trotter_order
    trotter_steps_tavis_cummings = args.trotter_steps

    name = args.name
    directory = args.directory
    value = args.value
    repetitions = args.repetitions
    circuit_write = args.circuit_write

    ham_tavis_cummings = tavis_cummings_model_qubit_hamiltonian(n_s = n_s, n_b = n_b, omega_c = omega_c, omega_o = omega_o, lam = lam)

    #this scales the circuit depth proportional to 2 ^ bits_precision

    E_min_tavis_cummings = -len(ham_tavis_cummings.terms)
    E_max_tavis_cummings = 0
    tavis_cummings_omega = E_max_tavis_cummings-E_min_tavis_cummings
    t_tavis_cummings = 2*np.pi/tavis_cummings_omega
    tavis_cummings_phase_offset = E_max_tavis_cummings*t_tavis_cummings

    args_tavis_cummings = {
        'trotterize' : True,
        'mol_ham'    : ham_tavis_cummings,
        'ev_time'    : t_tavis_cummings,
        'trot_ord'   : trotter_order_tavis_cummings,
        'trot_num'   : 1 #handling adjustment in resource estimate to save time - scales circuit depth linearly.
    }


    init_state_tavis_cummings = [0] * (n_b + n_s + 1) #TODO: use Fock state from Hartree-Fock as initial state

    print('starting')
    value = value/repetitions
    tavis_cummings_metadata = GSEEMetaData(
        id=time.time_ns(),
        name=name,
        category='scientific',
        size=f'{n_b} + 1 + {n_s}',
        task='Ground State Energy Estimation',
        value=value,
        repetitions_per_application=repetitions,

        
        evolution_time=t_tavis_cummings,
        trotter_order = trotter_order_tavis_cummings,
        bits_precision = bits_precision_tavis_cummings,
        nsteps=trotter_steps_tavis_cummings, 
    )

    print('Estimating tavis_cummings')
    t0 = time.perf_counter()
    estimate = gsee_resource_estimation(
        outdir=directory,
        nsteps=trotter_steps_tavis_cummings,
        gsee_args=args_tavis_cummings,
        init_state=init_state_tavis_cummings,
        precision_order=1, #actual precision bits accounted as scaling factors in the resource estimate
        phase_offset=tavis_cummings_phase_offset,
        bits_precision=bits_precision_tavis_cummings,
        circuit_name=name,
        metadata = tavis_cummings_metadata,
        write_circuits=circuit_write
    )
    t1 = time.perf_counter()
    print(f'Time to estimate tavis_cummings: {t1-t0}')
    return estimate

def estimate_bits_precision(epsilon):
    return math.ceil(math.log2(1.0/epsilon))

def parse_arguments():
    parser = argparse.ArgumentParser(prog='TavisCummingsModel')

    parser.add_argument('-ns','--param-n_s', type=int, default=100)
    parser.add_argument('-nb','--param-n_b', type=int, default=100)
    parser.add_argument('-oc','--param-omega_c', type=float, default=1.3)
    parser.add_argument('-oo','--param-omega_o', type=float, default=1)
    parser.add_argument('-lam','--param-lambda', type=float, default=1.5)
    parser.add_argument('-hb','--param-h_bar', type=float, default=1)

    parser.add_argument('-e', '--error-precision', type=float, default=1e-3)
    parser.add_argument('-t', '--trotter-steps', type=int, default=1)
    parser.add_argument('-to', '--trotter-order', type=int, default=2)
    parser.add_argument('-n', '--name', type=str, default=f'TavisCummingsModel', help='name of this circuit instance, becomes prefix for output file')
    parser.add_argument('-d', '--directory', type=str, default='./', help='output file directory')
    parser.add_argument('-v', '--value', type=float, default=0, help='value of the total application')
    parser.add_argument('-r', '--repetitions', type=int, default=1, help='repetitions needed to achieve value of computatation (not runs of this script)')
    parser.add_argument('-c', '--circuit_write', default=False, action='store_true')
    parser.add_argument('-x', '--extrapolate', default=False, action='store_true')
    return parser

if __name__ == "__main__":
    parser = parse_arguments()
    main(parser.parse_args())
