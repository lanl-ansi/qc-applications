import os
from argparse import ArgumentParser
import numpy as np
from math import sqrt
import pandas as pd
import networkx as nx
from networkx import Graph
from pandas import DataFrame
from networkx.generators.lattice import hexagonal_lattice_graph
from qca.utils.utils import TrotterMetaData, QSPMetaData, GSEEMetaData
from qca.utils.algo_utils import estimate_trotter, estimate_qsp, gsee_resource_estimation
from qca.utils.hamiltonian_utils import (
    flatten_nx_graph,
    assign_hexagon_labels,
    pyliqtr_hamiltonian_to_openfermion_qubit_operator,
)
from pyLIQTR.utils.Hamiltonian import Hamiltonian as pyH
import time

def nx_rucl_terms(graph, data_series):
    H = []
    n = len(graph.nodes)
    for (n1,n2,edge_data) in graph.edges(data=True):
        label = edge_data['label'][0]
        distance = int(edge_data['label'][1])
        
        #Heisenberg and Kitaev terms
        if distance == 1:
            weight_J = data_series.J1
            weight_K = data_series.K1
        elif distance == 2:
            weight_J = data_series.J2
            weight_K = data_series.K2
        elif distance == 3:
            weight_J = data_series.J3
            weight_K = data_series.K3
        else:
            raise ValueError(f'Invalid value for distance: {distance}. Expected 1, 2, or 3')
            
        
        if not (weight_J == 0 and weight_K == 0):
            string_x = n*'I' 
            string_y = n*'I' 
            string_z = n*'I'
            
            weight_x = weight_J
            weight_y = weight_J
            weight_z = weight_J
            for i in [n1,n2]:
                string_x = string_x[:i] + 'X' + string_x[i+1:]
                string_y = string_y[:i] + 'Y' + string_y[i+1:]
                string_z = string_z[:i] + 'Z' + string_z[i+1:]
            if label == 'X':
                weight_x += weight_K
            elif label == 'Y':
                weight_y += weight_K
            else:
                weight_z += weight_K
            if weight_x != 0:
                H.append((string_x, weight_x))
            if weight_y != 0:
                H.append((string_y, weight_y))
            if weight_z != 0:
                H.append((string_z, weight_z))
            
        #Gamma Terms
        if distance == 1 and data_series.Gam1 != 0:
            string_gam1_1 = n*'I'
            string_gam1_2 = n*'I'
            #unwrapping loop since there is no ordering guarantee
            labels=['X', 'Y', 'Z']
            labels.remove(label)
            l1,l2 = labels
            string_gam1_1 = string_gam1_1[:n1] + l1 + string_gam1_1[n1+1:]
            string_gam1_1 = string_gam1_1[:n2] + l2 + string_gam1_1[n2+1:]
            
            string_gam1_2 = string_gam1_2[:n1] + l2 + string_gam1_2[n1+1:]
            string_gam1_2 = string_gam1_2[:n2] + l1 + string_gam1_2[n2+1:]
            
            H.append((string_gam1_1, data_series.Gam1))
            H.append((string_gam1_2, data_series.Gam1))
            
        #Gamma' Terms
        if distance == 1 and data_series.Gam_prime1 != 0:
            #unwrapping inner loop since there is no ordering guarantee
            labels=['X', 'Y', 'Z']
            labels.remove(label)
            for label_offset in labels:
                string_gam1_1 = n*'I'
                string_gam1_2 = n*'I'
                l1 = label
                l2 = label_offset
                
                string_gam1_1 = string_gam1_1[:n1] + l1 + string_gam1_1[n1+1:]
                string_gam1_1 = string_gam1_1[:n2] + l2 + string_gam1_1[n2+1:]
                string_gam1_2 = string_gam1_2[:n1] + l2 + string_gam1_2[n1+1:]
                string_gam1_2 = string_gam1_2[:n2] + l1 + string_gam1_2[n2+1:]
                H.append((string_gam1_1, data_series.Gam_prime1))
                H.append((string_gam1_2, data_series.Gam_prime1))
    return H


def generate_time_varying_terms(graph, s, x = lambda s: 0, y = lambda s: 0, z = lambda s: 0):
    assert callable(x)
    assert callable(y)
    assert callable(z)
    
    weight_x, weight_y, weight_z = x(s), y(s), z(s)
    n = len(graph)
    H = []
    if not (weight_x == 0):
        for node in graph.nodes:
            string_x = n*'I'
            string_x = string_x[:node] + 'X' + string_x[node+1:]
            H.append((string_x, weight_x))
    if not (weight_y == 0):
        for node in graph.nodes:
            string_y = n*'I'
            string_y = string_y[:node] + 'Y' + string_y[node+1:]
            H.append((string_y, weight_y))
    if not (weight_z == 0):
        for node in graph.nodes:
            string_z = n*'I'
            string_z = string_z[:node] + 'Z' + string_z[node+1:]
            H.append((string_z, weight_z))
    return H


def assign_hexagon_labels_rucl(graph):
    assign_hexagon_labels(graph, 'X1', 'Y1', 'Z1')
       
    #Adding next nearest and next-next nearest neighbor edges and labels
    for n in graph.nodes:
        r,c = n
        
        #next nearest neighbors
        if (r, c+2) in graph:
            graph.add_edge(n, (r, c+2), label = 'Z2')
        if (r+1, c+1) in graph:
            graph.add_edge(n, (r+1, c+1), label = 'Y2')
        if (r-1, c+1) in graph:
            graph.add_edge(n, (r-1, c+1), label = 'X2')
       
        #next-next nearest neighbors
        if (r+1, c) in graph and not ((n, (r+1, c)) in graph.edges):
            graph.add_edge(n, (r+1,c), label = 'Z3')
        if (r+1, c+2) in graph and (r + c)%2 == 0:
            graph.add_edge(n, (r+1, c+2), label= 'X3')
        if (r-1, c+2) in graph and (r + c)%2 == 1:
            graph.add_edge(n, (r-1, c+2), label= 'Y3')

def generate_rucl_hamiltonian(lattice_size, data_series, s=0, field_x=lambda s: 0, field_y=lambda s: 0, field_z=lambda s: 0):
    graph = hexagonal_lattice_graph(lattice_size,lattice_size)
    assign_hexagon_labels_rucl(graph)
    graph = flatten_nx_graph(graph)
    H_constant = nx_rucl_terms(graph, data_series)
    H_time_varied = generate_time_varying_terms(graph, s, x=field_x, y = field_y, z = field_z)
    H = H_constant + H_time_varied
    return H

def assign_spin_labels_rucl(lattice_size)->Graph:
    graph = hexagonal_lattice_graph(lattice_size, lattice_size)
    spin_labels = dict([(node, pow(-1, node[0])) for node in graph])
    nx.set_node_attributes(graph, spin_labels, name= 'spin')
    return graph

def gen_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-L',
        '--lattice_size',
        type=int,
        required=True,
        help='integer denoting the lattice size'
    )

    parser.add_argument(
        '-M',
        '--mode',
        type=str,
        required=True,
        choices=['dynamics', 'gsee'],
        help ='Mode of operation: "dynamics" or "gsee'
    )

    parser.add_argument(
        '-N',
        '--nsteps',
        type=int,
        required=True,
        help="Number of steps for the algorithms"
    )

    parser.add_argument(
        '-T',
        '--evolution_time',
        type=float,
        default=1000.0,
        help='The evolution time (default: 1000)'
    )

    parser.add_argument(
        '-D', 
        '--directory', 
        type=str, 
        help='Directoty with pathway datafiles.',
        default='../data/'
    )
    return parser.parse_args()

def generate_rucl_dynamics_re(
    energy_precision:float,
    lattice_size: int,
    evolution_time:float,
    nsteps:int,
    df_rucl:DataFrame,
    outdir:str) -> None:

    gate_synth_accuracy = 10
    trotter_order = 2
    is_extrapolated=True

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for rucl_idx in range(len(df_rucl)):
        H_rucl = generate_rucl_hamiltonian(
            lattice_size,
            df_rucl.iloc[rucl_idx],
            field_x = lambda s: 1/sqrt(6)/2,
            field_y = lambda s: 1/sqrt(6)/2,
            field_z = lambda s: -2/sqrt(6)/2
        )
        H_rucl_pyliqtr = pyH(H_rucl)
        openfermion_hamiltonian_rucl = pyliqtr_hamiltonian_to_openfermion_qubit_operator(H_rucl_pyliqtr)


        trotter_metadata = TrotterMetaData(
            id=f'{time.time_ns()}',
            name=f'RuCl_row_{rucl_idx}',
            category='scientific',
            size=f'lattice_size: {lattice_size}',
            task='Time-Dependent Dynamics',

            gate_synth_accuracy=gate_synth_accuracy,
            evolution_time = evolution_time, 
            nsteps = nsteps,
            trotter_order = trotter_order,
            energy_precision=energy_precision,
            is_extrapolated=is_extrapolated,
        )
        estimate_trotter(
            openfermion_hamiltonian=openfermion_hamiltonian_rucl,
            evolution_time=evolution_time,
            energy_precision=energy_precision,
            metadata=trotter_metadata,
            outdir=outdir,
            trotter_order=trotter_order,
            hamiltonian_name=f'trotter_rucl_size_{lattice_size}_row_{rucl_idx}',
            nsteps=nsteps,
            is_extrapolated=is_extrapolated
        )

        qsp_metadata = QSPMetaData(
            id =f'{time.time_ns()}',
            name=f'RuCl_row_{rucl_idx}',
            category='scientific',
            size=f'lattice_size: {lattice_size}',
            task='Time-Dependent Dynamics',

            evolution_time = evolution_time, 
            nsteps = nsteps,
            energy_precision=energy_precision,
        )

        estimate_qsp(
            pyliqtr_hamiltonian=H_rucl_pyliqtr,
            evolution_time=evolution_time,
            nsteps=nsteps,
            energy_precision=energy_precision,
            metadata=qsp_metadata,
            outdir=outdir,
            hamiltonian_name=f'qsp_rucl_size_{lattice_size}_row_{rucl_idx}',
            write_circuits=False
        )

def generate_rucl_gsee_re(
    bits_precision:float,
    lattice_size: int,
    evolution_time:float,
    nsteps:int,
    df_rucl:DataFrame,
    outdir:str) -> None:

    gate_synth_accuracy = 10
    trotter_order = 2
    is_extrapolated=True

    graph = hexagonal_lattice_graph(lattice_size,lattice_size)
    assign_hexagon_labels(graph)
    graph = flatten_nx_graph(graph)
    n_qubits = len(graph.nodes)
    init_state = [0] * n_qubits

    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for rucl_idx in range(len(df_rucl)):
        H_rucl_no_fields = generate_rucl_hamiltonian(
            lattice_size,
            df_rucl.iloc[rucl_idx],
            field_x = lambda s: 0,
            field_y = lambda s: 0,
            field_z = lambda s: 0
        )
        H_rucl__no_fields_pyliqtr = pyH(H_rucl_no_fields)
        openfermion_hamiltonian_rucl_no_fields = pyliqtr_hamiltonian_to_openfermion_qubit_operator(H_rucl__no_fields_pyliqtr)


        E_min = -len(openfermion_hamiltonian_rucl_no_fields.terms)
        E_max = 0
        omega = E_max-E_min
        t = 2*np.pi/omega
        phase_offset = E_max*t

        gsee_metadata = GSEEMetaData(
            id = f'{time.time_ns()}',
            name=f'RuCl_row_{rucl_idx}',
            category='scientific',
            size=f'lattice_size: {lattice_size}',
            task='Ground State Energy Estimation',

            gate_synth_accuracy=gate_synth_accuracy,
            evolution_time = evolution_time, 
            nsteps = nsteps,
            trotter_order = trotter_order,
            bits_precision = bits_precision,
            is_extrapolated=is_extrapolated,

        )

        gsee_args = {
        'trotterize' : True,
        'mol_ham'    : openfermion_hamiltonian_rucl_no_fields,
        'ev_time'    : evolution_time,
        'trot_ord'   : trotter_order,
        'trot_num'   : nsteps
    }
        gsee_resource_estimation(
            outdir=outdir,
            nsteps=nsteps,
            gsee_args=gsee_args,
            init_state=init_state,
            precision_order=1,
            bits_precision=bits_precision,
            phase_offset=phase_offset,
            metadata = gsee_metadata,
            circuit_name = f'qsee_rucl_size_{lattice_size}_row_{rucl_idx}'
    )


def rucl_estimate():
    args = gen_args()

    RuCl_csv_directory=args.directory
    df_rucl = pd.read_csv(f"{RuCl_csv_directory}RuCl_test_input.csv")

    if args.mode == 'dynamics':
        re_dir = 'RuCl_RE/Dynamics/'
        generate_rucl_dynamics_re(
            energy_precision=1e-3,
            lattice_size=args.lattice_size,
            evolution_time=args.evolution_time,
            nsteps=args.nsteps,
            df_rucl=df_rucl,
            outdir=re_dir
        )
    elif args.mode == 'gsee':
        re_dir = 'RuCl_RE/GSEE/'
        generate_rucl_gsee_re(
            bits_precision = 10,
            lattice_size=args.lattice_size,
            evolution_time=args.evolution_time,
            nsteps=args.nsteps,
            df_rucl=df_rucl,
            outdir=re_dir
        )


if __name__ == '__main__':
    rucl_estimate()
