import os
import re
import json
import time
from statistics import median
from dataclasses import dataclass, asdict

import pandas as pd

import matplotlib.pyplot as plt

import cirq

from pyLIQTR.utils.utils import count_T_gates
from pyLIQTR.utils.qsp_helpers import print_to_openqasm
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform

@dataclass
class EstimateMetaData:
    id: str
    name: str
    category: str
    size: str
    task: str
    implementations: str
    value_per_circuit: float
    repetitions_per_application: int

def count_gates(cpt_circuit: cirq.AbstractCircuit) -> int:
    count = 0
    for moment in cpt_circuit:
        count += len(moment)
    return count

def extract_number(string):
    number = re.findall(r'\d+', string)
    return int(number[0]) if number else None

def get_T_depth_wire(cpt_circuit: cirq.AbstractCircuit):
    # maximum number of T-gates on a wire.  This may be more optimistic than
    # number of layers with T-gates.  Perhaps good to treat as lower bound
    # for an implementation
    count_dict = {}
    for moment in cpt_circuit:
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                reg_label = opstr[opstr.find("(")+1:opstr.find(")")]
                if reg_label not in count_dict:
                    count_dict[reg_label] = 1
                else:
                    count_dict[reg_label] += 1
    max_depth=0
    for register in count_dict:
        if count_dict[register] > max_depth:
            max_depth = count_dict[register]
    return max_depth

def get_T_width(cpt_circuit: cirq.AbstractCircuit) -> list[int]:
    t_widths = [0] * len(cpt_circuit)
    for i, moment in enumerate(cpt_circuit):
        width = 0
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                width += 1
        t_widths[i] = width
    return t_widths

def plot_T_step_histogram(cpt_circuit: cirq.AbstractCircuit, kwargs, lowest_ind:int=0) -> plt.hist:
    t_widths = get_T_width(cpt_circuit)
    bins = range(max(t_widths))
    y_label = bins[lowest_ind:-1]
    histogram = plt.hist(t_widths, y_label, **kwargs)
    return histogram, median(t_widths)

def plot_histogram(
        cpt_circuit: cirq.AbstractCircuit,
        histogram_title:str,
        figdir:str,
        widthdir:str,
        lowest_ind:int=0,
        **kwargs
    ) -> None:
    circuit_histogram, median_t_width = plot_T_step_histogram(cpt_circuit, kwargs=kwargs, lowest_ind=lowest_ind)
    median_t_width_text = f'median T width: {median_t_width}'
    plt.title(f'{histogram_title} - {median_t_width_text}')
    plt.xlabel('T Width')
    plt.ylabel('Count')
    plt.savefig(f'{figdir}_width_histogram_square.pdf')
    df_histogram_trotter_square = pd.DataFrame({'bin': circuit_histogram[1][:-1], \
                                            'count': circuit_histogram[0]})
    df_histogram_trotter_square.to_csv(f'{widthdir}widths_square.csv', sep=',',index=False)

def get_T_depth(cpt_circuit: cirq.AbstractCircuit):
    t_depth = 0
    for moment in cpt_circuit:
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                t_depth += 1
                break
    return t_depth

def gen_resource_estimate(
        cpt_circuit: cirq.AbstractCircuit,
        is_extrapolated: bool,
        total_steps:int = -1,
        bits_precision:int=1
    ) -> dict:
    '''
    Given some clifford + T circuit and a given filename, we grab the logical resource estimates
    from the circuit and then write it to disk. The function also returns the resource dictionary
    if the user needs it.

    trotter_steps is a flag denoting if the circuit was estimated through trotterization. If so, the
    user should specify the number of steps required.  
    '''
    num_qubits = len(cpt_circuit.all_qubits())
    gate_count = count_gates(cpt_circuit)
    t_count = count_T_gates(cpt_circuit)
    t_depth = get_T_depth(cpt_circuit)
    clifford_count = gate_count - t_count
    circuit_depth = len(cpt_circuit)

    resource_estimate = {
        'num_qubits': num_qubits,
        't_count': t_count,
        'circuit_depth': circuit_depth,
        'gate_count': gate_count,
        't_depth': t_depth,
        'clifford_count': clifford_count,
    }

    if total_steps > 0 and is_extrapolated:
        scaling_factor = total_steps
    elif bits_precision > 0:
        scaling_factor = pow(2, bits_precision - 1)
    else:
        scaling_factor = None
    
    if scaling_factor:
        resource_estimate['t_depth'] = resource_estimate['t_depth'] * scaling_factor
        resource_estimate['t_count'] = resource_estimate['t_count'] * scaling_factor
        resource_estimate['gate_count'] = resource_estimate['gate_count'] * scaling_factor
        resource_estimate['circuit_depth'] = resource_estimate['circuit_depth'] * scaling_factor
        resource_estimate['clifford_count'] = resource_estimate['clifford_count'] * scaling_factor

    return resource_estimate


def estimate_cpt_resources(
        cpt_circuit: cirq.AbstractCircuit,
        metadata: EstimateMetaData,
        outdir: str,
        is_extrapolated:bool,
        circuit_name:str,
        magnus_steps:int=1,
        trotter_steps:int=1
    ):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    total_steps = trotter_steps * magnus_steps
    re_metadata = asdict(metadata)
    re_metadata['Logical_Abstract'] = gen_resource_estimate(
        cpt_circuit=cpt_circuit,
        is_extrapolated=is_extrapolated,
        total_steps=total_steps
    )
    resource_estimate = re_metadata

    outfile_data = f'{outdir}{circuit_name}_re.json'
    re_as_json(resource_estimate, outfile_data)

def circuit_estimate(
        circuit: cirq.AbstractCircuit,
        metadata: EstimateMetaData,
        outdir: str,
        numsteps: int,
        circuit_name: str,
        bits_precision:int=1,
        write_circuits:bool = False
    ) -> cirq.AbstractCircuit:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    subcircuit_counts = {}
    for moment in circuit:
        for operation in moment:
            gate_type = type(operation.gate)
            gate_type_name = f'{str(gate_type).split("/")[-1][:-2]}'
            if gate_type in subcircuit_counts:
                subcircuit_counts[gate_type][0] += 1
            else:
                t0 = time.perf_counter()
                decomposed_circuit = cirq.Circuit(cirq.decompose(operation))
                t1 = time.perf_counter()
                decomposed_elapsed = t1-t0
                print(f'   Time to decompose high level {gate_type_name} circuit: {decomposed_elapsed} seconds ')
                t0 = time.perf_counter()
                cpt_circuit = clifford_plus_t_direct_transform(decomposed_circuit)
                t1 = time.perf_counter()
                cpt_elapsed = t1-t0
                print(f'   Time to transform decomposed {gate_type_name} circuit to Clifford+T: {cpt_elapsed} seconds')
                if write_circuits:
                    outfile_qasm_decomposed = f'{outdir}{gate_type_name}.decomposed.qasm'
                    outfile_qasm_cpt = f'{outdir}{gate_type_name}.cpt.qasm'
                    with open(outfile_qasm_decomposed, 'w', encoding='utf-8') as f:
                        print_to_openqasm(f, decomposed_circuit, qubits=decomposed_circuit.all_qubits())
                    with open(outfile_qasm_cpt, 'w', encoding='utf-8') as f:
                        print_to_openqasm(f, cpt_circuit, qubits=cpt_circuit.all_qubits())
                    
                subcircuit_counts[gate_type] = [1, cpt_circuit, gate_type_name]

    total_gate_count = 0
    total_gate_depth = 0
    total_T_count = 0
    total_T_depth = 0
    total_clifford_count = 0
    subcircuit_re = []
    for gate in subcircuit_counts:
        subcircuit = subcircuit_counts[gate][1]
        subcircuit_name = subcircuit_counts[gate][2]
        resource_estimate = gen_resource_estimate(
            subcircuit,
            is_extrapolated=False,
            bits_precision=bits_precision
        )
        subcircuit_info = {subcircuit_name:resource_estimate}
        subcircuit_re.append(subcircuit_info)

        gate_count = resource_estimate['gate_count']
        gate_depth = resource_estimate['circuit_depth']
        t_depth = resource_estimate['t_depth']
        t_count = resource_estimate['t_count']
        clifford_count = resource_estimate['clifford_count']
        
        total_gate_count += subcircuit_counts[gate][0] * gate_count * numsteps
        total_gate_depth += subcircuit_counts[gate][0] * gate_depth * numsteps
        total_T_depth += subcircuit_counts[gate][0] * t_depth * numsteps
        total_T_count += subcircuit_counts[gate][0] * t_count * numsteps
        total_clifford_count += subcircuit_counts[gate][0] * clifford_count * numsteps
    re_metadata = asdict(metadata)
    re_metadata['Logical_Abstract'] = {
        'num_qubits': len(circuit.all_qubits()),
        't_count': total_T_count,
        'circuit_depth': total_gate_depth,
        'gate_count': total_gate_count,
        't_depth': total_T_depth,
        'clifford_count': total_clifford_count
    }
    outfile_data = f'{outdir}{circuit_name}_re.json'
    re_as_json(re_metadata, outfile_data)

def re_as_json(main_estimate:dict, outdir:str) -> None:
    with open(outdir, 'w') as f:
            json.dump(main_estimate, f,
                    indent=4,
                    separators=(',', ': '))
