import os
import time
import random
import numpy as np
import networkx as nx
from cirq import Circuit, QasmOutput
from openfermion import count_qubits
from cirq.contrib import qasm_import
from pyLIQTR.utils.Hamiltonian import Hamiltonian
from openfermion.ops.operators import QubitOperator
from pyLIQTR.utils.utils import open_fermion_to_qasm
from pyLIQTR.circuits.qsp import generate_QSP_circuit
from pyLIQTR.utils.qsp_helpers import print_to_openqasm
from openfermion.circuits import trotter_steps_required, error_bound
from qca.utils.utils import circuit_estimate, estimate_cpt_resources
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from openfermion.circuits.trotter_exp_to_qgates import trotterize_exp_qubop_to_qasm
from pyLIQTR.phase_factors.fourier_response.fourier_response import Angler_fourier_response

def estimate_qsp(
    pyliqtr_hamiltonian: Hamiltonian,
    evolution_time:float,
    numsteps:int,
    energy_precision:float,
    outdir:str,
    hamiltonian_name:str='hamiltonian',
    write_circuits:bool=False,
) -> Circuit:
    timestep_of_interest=evolution_time/numsteps
    random.seed(0)
    np.random.seed(0)
    t0 = time.perf_counter()
    angles_response = Angler_fourier_response(tau=timestep_of_interest*pyliqtr_hamiltonian.alpha,
                                              eps=energy_precision,
                                              random=True,
                                              silent=True)
    angles_response.generate()
    angles = angles_response.phases
    qsp_circuit = generate_QSP_circuit(pyliqtr_hamiltonian, angles, pyliqtr_hamiltonian.problem_size)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f'Time to generate high level QSP circuit: {elapsed} seconds')
    circuit_estimate(
        circuit=qsp_circuit,
        outdir=outdir,
        numsteps=numsteps,
        circuit_name=hamiltonian_name,
        algo_name='QSP_Step',
        write_circuits=write_circuits
    )
    return qsp_circuit

def find_hamiltonian_ordering(of_hamiltonian: QubitOperator) -> list:
    """
    Function to generate a near optimal term ordering for trotterization of transverse field Ising Models.
    This would need to be modified if there were multi-qubit interactions that were not just ZZ
    """
    #ordering hamiltonian terms by performing edge coloring to make optimal trotter ordering
    #assuming that any 2 body interactions are ZZ
    sorted_terms = sorted(list(of_hamiltonian.terms.keys()))
    sorted_terms.sort(key=lambda x: len(x) * 100 + ord(x[0][1])) #Z and X get translated to 90 and 88 respectively, multiplying by 100 ensures interacting term weight is considered
    one_body_terms_ordered = list(filter(lambda x: len(x) == 1, sorted_terms))
    two_body_terms = list(filter(lambda x: len(x) == 2, sorted_terms))
    
    #assigning edge colorings to order two body terms
    g = nx.Graph()
    for term in two_body_terms:
        edge = (term[0][0], term[1][0])
        g.add_edge(*edge)
    edge_coloring = nx.greedy_color(nx.line_graph(g))
    nx.set_edge_attributes(g, edge_coloring, "color")

    for (i,term) in enumerate(two_body_terms):
        n1,n2 = (term[0][0], term[1][0])
        color = g.edges[n1,n2]['color']
        term = (*term, color)
        two_body_terms[i] = term
    
    two_body_terms.sort(key=lambda x: x[2])
    two_body_terms_ordered = list()
    for (i,term) in enumerate(two_body_terms):
        new_item = (term[0],term[1])
        two_body_terms_ordered.append(new_item)
    return one_body_terms_ordered + two_body_terms_ordered


def estimate_trotter(
    openfermion_hamiltonian: QubitOperator,
    evolution_time: float,
    energy_precision: float,
    outdir:str,
    hamiltonian_name:str='hamiltonian',
    write_circuits:bool=False
) -> Circuit:

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    t0 = time.perf_counter()
    bounded_error = error_bound(list(openfermion_hamiltonian.get_operators()),tight=False)
    nsteps = trotter_steps_required(trotter_error_bound = bounded_error,
                                    time = evolution_time, 
                                    energy_precision = energy_precision)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f'Time to estimate number of trotter steps required ({nsteps}): {elapsed} seconds')

    t0 = time.perf_counter()
    term_ordering = find_hamiltonian_ordering(openfermion_hamiltonian)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f'Time to find term ordering: {elapsed} seconds')
    
    t0 = time.perf_counter()
    trotter_circuit_of = trotterize_exp_qubop_to_qasm(openfermion_hamiltonian,
                                                      trotter_order=2,
                                                      evolution_time=evolution_time/nsteps,
                                                      term_ordering=term_ordering)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f'Time to generate trotter circuit from openfermion: {elapsed} seconds')

    qasm_str_trotter = open_fermion_to_qasm(count_qubits(openfermion_hamiltonian), trotter_circuit_of)
    trotter_circuit_qasm = qasm_import.circuit_from_qasm(qasm_str_trotter)
    t0 = time.perf_counter()
    cpt_trotter = clifford_plus_t_direct_transform(trotter_circuit_qasm)
    t1 = time.perf_counter()
    elapsed = t1-t0
    print(f'Time to generate a clifford + T circuit from trotter circuit: {elapsed} seconds')

    if write_circuits:
        outfile_qasm_trotter = f'{outdir}Trotter_Unitary.qasm'
        outfile_qasm_cpt = f'{outdir}Trotter_Unitary.cpt.qasm'
        with open(outfile_qasm_trotter, 'w', encoding='utf-8') as f:
            print_to_openqasm(f, trotter_circuit_qasm, trotter_circuit_qasm.all_qubits())
        with open(outfile_qasm_cpt, 'w', encoding='utf-8') as f:
            print_to_openqasm(f, cpt_trotter, qubits=cpt_trotter.all_qubits())

    estimate_cpt_resources(
        cpt_trotter,
        outdir,
        is_extrapolated=True,
        circuit_name=hamiltonian_name,
        algo_name='Trotter_Step',
        trotter_steps=nsteps
    )
    return cpt_trotter