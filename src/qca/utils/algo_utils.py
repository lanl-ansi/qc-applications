import os
import time
import random
import numpy as np
import networkx as nx

import cirq
from cirq import Circuit
from cirq.contrib import qasm_import

from openfermion import count_qubits
from openfermion.ops.operators import QubitOperator
from openfermion.circuits import trotter_steps_required, error_bound
from openfermion.circuits.trotter_exp_to_qgates import trotterize_exp_qubop_to_qasm

from pyLIQTR.utils.Hamiltonian import Hamiltonian
from pyLIQTR.utils.utils import open_fermion_to_qasm
from pyLIQTR.circuits.qsp import generate_QSP_circuit
from pyLIQTR.PhaseEstimation.pe import PhaseEstimation
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from pyLIQTR.phase_factors.fourier_response.fourier_response import Angler_fourier_response

from qca.utils.utils import (
    gen_json,
    write_qasm,
    grab_circuit_resources,
    estimate_cpt_resources,
    GSEEMetaData,
    TrotterMetaData,
    QSPMetaData
)

def find_hamiltonian_ordering(of_hamiltonian: QubitOperator) -> list:
    """
    Function to generate a near optimal term ordering for trotterization of transverse field Ising Models.
    This would need to be modified if there were multi-qubit interactions that were not just ZZ
    """
    #ordering hamiltonian terms by performing edge coloring to make optimal trotter ordering
    #assuming that any 2 body interactions are ZZ
    sorted_terms = sorted(list(of_hamiltonian.terms.keys()))

    #Z and X get translated to 90 and 88 respectively, multiplying by 100 ensures interacting term weight is considered
    sorted_terms.sort(key=lambda x: len(x) * 100 + ord(x[0][1]))
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
    two_body_terms_ordered = []
    for (i,term) in enumerate(two_body_terms):
        new_item = (term[0],term[1])
        two_body_terms_ordered.append(new_item)
    return one_body_terms_ordered + two_body_terms_ordered

def estimate_qsp(
    pyliqtr_hamiltonian: Hamiltonian,
    evolution_time:float,
    nsteps:int,
    energy_precision:float,
    outdir:str,
    is_extrapolated:bool=False,
    metadata: QSPMetaData | None=None,
    hamiltonian_name:str='hamiltonian',
    write_circuits:bool=False,
    include_nested_resources:bool=True
) -> Circuit:
    timestep_of_interest=evolution_time/nsteps
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

    if nsteps and not is_extrapolated:
        is_extrapolated=True

    gate_synth_accuracy=metadata.gate_synth_accuracy
    grab_circuit_resources(
        circuit=qsp_circuit,
        outdir=outdir,
        algo_name='QSP',
        fname=hamiltonian_name,
        is_extrapolated=is_extrapolated,
        numsteps=nsteps,
        metadata=metadata,
        write_circuits=write_circuits,
        include_nested_resources=include_nested_resources,
        gate_synth_accuracy=gate_synth_accuracy
    )

    return qsp_circuit


def estimate_trotter(
    openfermion_hamiltonian: QubitOperator,
    evolution_time: float,
    energy_precision: float,
    outdir:str,
    is_extrapolated: bool=True,
    trotter_order: int = 2,
    metadata: TrotterMetaData | None=None,
    hamiltonian_name:str='hamiltonian',
    write_circuits:bool=False,
    nsteps:int|None=None,
    include_nested_resources:bool=True
) -> Circuit:
      
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not nsteps:
        t0 = time.perf_counter()
        bounded_error = error_bound(list(openfermion_hamiltonian.get_operators()),tight=False)
        nsteps = trotter_steps_required(trotter_error_bound = bounded_error,
                                        time = evolution_time,
                                        energy_precision = energy_precision)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f'Time to estimate number of trotter steps required ({nsteps}): {elapsed} seconds')

    metadata.nsteps=nsteps

    t0 = time.perf_counter()
    term_ordering = find_hamiltonian_ordering(openfermion_hamiltonian)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f'Time to find term ordering: {elapsed} seconds')
    t0 = time.perf_counter()

    #generates the circuit for a single trotter step and extrapolates the rest
    trotter_circuit_of = trotterize_exp_qubop_to_qasm(openfermion_hamiltonian,
                                                      trotter_order=trotter_order,
                                                      evolution_time=evolution_time/nsteps,
                                                      term_ordering=term_ordering)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f'Time to generate trotter circuit from openfermion: {elapsed} seconds')

    gate_synth_accuracy = metadata.gate_synth_accuracy
    qasm_str_trotter = open_fermion_to_qasm(count_qubits(openfermion_hamiltonian), trotter_circuit_of)
    trotter_circuit_qasm = qasm_import.circuit_from_qasm(qasm_str_trotter)

    t0 = time.perf_counter()
    cpt_trotter = clifford_plus_t_direct_transform(circuit=trotter_circuit_qasm, gate_precision=gate_synth_accuracy)
    t1 = time.perf_counter()
    elapsed = t1-t0
    print(f'Time to generate a clifford + T circuit from trotter circuit: {elapsed} seconds')

    if write_circuits:
        outfile_qasm_trotter = f'{outdir}Trotter_Unitary.qasm'
        write_qasm(
            circuit=trotter_circuit_qasm,
            fname=outfile_qasm_trotter
        )
        outfile_qasm_cpt = f'{outdir}Trotter_Unitary.cpt.qasm'
        write_qasm(
            circuit=cpt_trotter,
            fname=outfile_qasm_cpt
        )

    logical_re = estimate_cpt_resources(
        cpt_circuit=cpt_trotter,
        is_extrapolated=is_extrapolated,
        algo_name= 'TrotterStep',
        total_steps=nsteps,
        include_nested_resources=include_nested_resources
    )
    outfile = f'{outdir}{hamiltonian_name}_re.json'

    gen_json(logical_re, outfile, metadata )
    return cpt_trotter

def gsee_resource_estimation(
        outdir:str,
        nsteps:int,
        gsee_args:dict,
        init_state:list,
        precision_order:int,
        bits_precision:int,
        phase_offset:float,
        is_extrapolated:bool=False,
        metadata:GSEEMetaData | None =None,
        circuit_name:str='Hamiltonian',
        include_nested_resources:bool=False,
        include_classical_bits:bool=False,
        write_circuits:bool=False,
) -> Circuit:
    t0 = time.perf_counter()
    gse_circuit = PhaseEstimation(
        precision_order=precision_order,
        init_state=init_state,
        include_classical_bits=include_classical_bits,
        phase_offset=phase_offset,
        kwargs=gsee_args
    )
    t1 = time.perf_counter()
    elapsed = t1-t0
    print(f'Time to generate circuit for GSEE: {elapsed} seconds')

    gse_circuit.generate_circuit()
    pe_circuit = gse_circuit.pe_circuit
    gate_synth_accuracy=metadata.gate_synth_accuracy

    if (nsteps or bits_precision) and not is_extrapolated:
        is_extrapolated = True

    grab_circuit_resources(
        circuit=pe_circuit,
        outdir=outdir,
        algo_name='GSEE',
        fname=circuit_name,
        is_extrapolated=is_extrapolated,
        numsteps=nsteps,
        bits_precision=bits_precision,
        metadata=metadata,
        write_circuits=write_circuits,
        include_nested_resources=include_nested_resources,
        gate_synth_accuracy=gate_synth_accuracy
    )

    return pe_circuit