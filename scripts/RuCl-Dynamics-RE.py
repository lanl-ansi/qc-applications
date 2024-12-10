import os
from argparse import ArgumentParser
import numpy as np
from math import sqrt
import networkx as nx
from networkx import Graph
from pandas import DataFrame
from networkx.generators.lattice import hexagonal_lattice_graph
from qca.utils.utils import TrotterMetaData, QSPMetaData
from qca.utils.algo_utils import estimate_trotter, estimate_qsp
from qca.utils.hamiltonian_utils import (
    flatten_nx_graph,
    assign_hexagon_labels,
    pyliqtr_hamiltonian_to_openfermion_qubit_operator,
)
from pyLIQTR.utils.Hamiltonian import Hamiltonian as pyH
import time

def nx_rucl_terms(g, data_series):
    H = []
    n = len(g.nodes)
    for (n1,n2,d) in g.edges(data=True):
        label = d['label'][0]
        distance = int(d['label'][1])

        #Heisenberg and Kitaev terms
        if distance == 1:
            weight_J = data_series.J1
        elif distance == 2:
            weight_J = data_series.J2
        else:
            weight_J = data_series.J3

        if distance == 1:
            weight_K = data_series.K1
        elif distance == 2:
            weight_K = data_series.K2
        else:
            weight_K = data_series.K3

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
    return H


def generate_time_varying_terms(g, s, x = lambda s: 0, y = lambda s: 0, z = lambda s: 0):
    assert callable(x)
    assert callable(y)
    assert callable(z)

    weight_x, weight_y, weight_z = x(s), y(s), z(s)
    n = len(g)
    H = []
    if not (weight_x == 0):
        for node in g.nodes:
            string_x = n*'I'
            string_x = string_x[:node] + 'X' + string_x[node+1:]
            H.append((string_x, weight_x))
    if not (weight_y == 0):
        for node in g.nodes:
            string_y = n*'I'
            string_y = string_y[:node] + 'Y' + string_y[node+1:]
            H.append((string_y, weight_y))
    if not (weight_z == 0):
        for node in g.nodes:
            string_z = n*'I'
            string_z = string_z[:node] + 'Z' + string_z[node+1:]
            H.append((string_z, weight_z))
    return H


def assign_hexagon_labels_rucl(g):
    assign_hexagon_labels(g, 'X1', 'Y1', 'Z1')

    # Adding next nearest and next-next nearest neighbor edges and labels
    for n in g.nodes:
        r,c = n

        #next nearest neighbors
        if (r, c+2) in g:
            g.add_edge(n, (r, c+2), label = 'Z2')
        if (r+1, c+1) in g:
            g.add_edge(n, (r+1, c+1), label = "Y2")
        if (r-1, c+1) in g:
            g.add_edge(n, (r-1, c+1), label = "X2")

        #next-next nearest neighbors
        if (r+1, c) in g and not ((n, (r+1, c)) in g.edges):
            g.add_edge(n, (r+1,c), label = "Z3")
        if (r+1, c+2) in g and (r + c)%2 == 0:
            g.add_edge(n, (r+1, c+2), label="X3")
        if (r-1, c+2) in g and (r + c)%2 == 1:
            g.add_edge(n, (r-1, c+2), label="Y3")

def generate_rucl_hamiltonian(lattice_size, data_series, s=0, field_x=lambda s: 0, field_y=lambda s: 0, field_z=lambda s: 0):
    g = hexagonal_lattice_graph(lattice_size,lattice_size)
    assign_hexagon_labels_rucl(g)
    g = flatten_nx_graph(g)
    H_constant = nx_rucl_terms(g, data_series)
    H_time_varied = generate_time_varying_terms(g, s, x=field_x, y = field_y, z = field_z)
    H = H_constant + H_time_varied
    return H

def assign_spin_labels_rucl(lattice_size:int) -> Graph:
    g = hexagonal_lattice_graph(lattice_size, lattice_size)
    spin_labels = dict([(node, pow(-1, node[0])) for node in g])
    nx.set_node_attributes(g, spin_labels, name='spin')
    return g

def gen_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-L',
        '--lattice_size',
        type=int,
        required=True,
        help='integer denoting the lattice size'
    )
    return parser.parse_args()

def generate_rucl_re(
    energy_precision:float,
    lattice_size: int,
    evolution_time:float,
    df_rucl:DataFrame,
    outdir:str) -> None:

    nsteps = 1500000
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

def rucl_estimate():
    args = gen_args()
    rucl_references = ["Winter et al. PRB", "Winter et al. NC", "Wu et al.", "Cookmeyer and Moore", "Kim and Kee", "Suzuki and Suga",
              "Yadav et al.", "Ran et al.", "Hou et al.", "Wang et al.", "Eichstaedt et al.", "Eichstaedt et al.",
              "Eichstaedt et al.", "Banerjee et al.", "Kim et al.", "Kim and Kee", "Winter et al.", "Ozel et al.", "Ozel et al."]

    rucl_methods = ["Ab initio (DFT + exact diag.)", "Ab initio-inspired (INS fit)", "THz spectroscopy fit",
                    "Magnon thermal Hall (sign)", "DFT + t=U expansion", "Magnetic specific heat", "Quantum chemistry (MRCI)",
                    "Spin wave fit to INS gap", "Constrained DFT + U", "DFT + t=U expansion", "Fully ab initio (DFT + cRPA + t=U)",
                    "Neglecting non-local Coulomb", "Neglecting non-local SOC", "Spin wave fit", "DFT + t=U expansion",
                    "DFT + t=U expansion", "Ab initio (DFT + exact diag.)", "Spin wave fit/THz spectroscopy", "Spin wave fit/THz spectroscopy"]

    rucl_J1 = [-1.7, -0.5, -0.35, -0.5, -1.53, -1.53, 1.2, 0, -1.87, -0.3, -1.4, -0.2, -1.3, -4.6, -12, -3.5, -5.5, -0.95, 0.46]
    rucl_K1 = [-6.7, -5.0, -2.8, -5.0, -6.55, -24.4, -5.6, -6.8, -10.7, -10.9, -14.3, -4.5, -13.3, 7.0, 17., 4.6, 7.6, 1.15, -3.5]
    rucl_Gam1 = [6.6, 2.5, 2.4, 2.5, 5.25, 5.25, 1.2, 9.5, 3.8, 6.1, 9.8, 3.0, 9.4, 0, 12., 6.42, 8.4, 3.8, 2.35]
    rucl_Gam_prime1 = [-0.9, 0, 0, 0, -0.95, -0.95, -0.7, 0, 0, 0, -2.23, -0.73, -2.3, 0, 0, -0.04, 0.2, 0, 0]
    rucl_J2 = [0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rucl_K2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.63, -0.33, -0.67, 0, 0, 0, 0, 0, 0]
    rucl_J3 = [2.7, 0.5, 0.34, 0.1125, 0, 0, 0.25, 0, 1.27, 0.03, 1.0, 0.7, 1.0, 0, 0, 0, 2.3, 0, 0]
    rucl_K3 = [0, 0, 0, 0, 0, 0, 0, 0, 0.63, 0, 0.03, 0.1, 0.1, 0, 0, 0, 0, 0, 0]

    d_rucl = {
        'reference': rucl_references,
        'method': rucl_methods,
        'J1': rucl_J1,
        'K1': rucl_K1,
        'Gam1': rucl_Gam1,
        'Gam_prime1': rucl_Gam_prime1,
        'J2': rucl_J2,
        'K2': rucl_K2,
        'J3': rucl_J3,
        'K3': rucl_K3}

    df_rucl = DataFrame(d_rucl)
    re_dir = 'temp_RE/Dynamics/'
    generate_rucl_re(
        energy_precision=1e-3,
        lattice_size=args.lattice_size,
        evolution_time=1000,
        df_rucl=df_rucl,
        outdir=re_dir
    )

if __name__ == '__main__':
    rucl_estimate()
