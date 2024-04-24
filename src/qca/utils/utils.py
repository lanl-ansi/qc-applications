import os
import re
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from statistics import median
from pyLIQTR.utils.utils import count_T_gates
from cirq import Circuit, QasmOutput, AbstractCircuit
from pyLIQTR.utils.qsp_helpers import circuit_decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform

def count_gates(cpt_circuit: AbstractCircuit) -> int:
    count = 0
    for moment in cpt_circuit:
        count += len(moment)
    return count

def extract_number(string):
    number = re.findall(r'\d+', string)
    return int(number[0]) if number else None

def get_T_depth_wire(cpt_circuit: AbstractCircuit):
    # maximum number of T-gates on a wire.  This may be more optimistic than
    # number of layers with T-gates.  Perhaps good to treat as lower bound
    # for an implementation
    count_dict = {}
    for moment in cpt_circuit:
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                reg_label = opstr[opstr.find("(")+1:opstr.find(")")]
                if not reg_label in count_dict:
                    count_dict[reg_label] = 1
                else:
                    count_dict[reg_label] += 1
    max_depth=0
    for register in count_dict:
        if count_dict[register] > max_depth:
            max_depth = count_dict[register]
    return max_depth

def get_T_width(cpt_circuit: AbstractCircuit) -> list[int]:
    t_widths = [0] * len(cpt_circuit)
    for i, moment in enumerate(cpt_circuit):
        width = 0
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                width += 1
        t_widths[i] = width
    return t_widths

def plot_T_step_histogram(cpt_circuit:AbstractCircuit, kwargs, lowest_ind:int=0) -> plt.hist:
    t_widths = get_T_width(cpt_circuit)
    bins = range(max(t_widths))
    y_label = bins[lowest_ind:-1]
    histogram = plt.hist(t_widths, y_label, **kwargs)
    return histogram, median(t_widths)

def plot_histogram(cpt_circuit: AbstractCircuit,
                   histogram_title:str,
                   figdir:str,
                   widthdir:str,
                   lowest_ind:int=0,
                   **kwargs) -> None:
    circuit_histogram, median_t_width = plot_T_step_histogram(cpt_circuit, kwargs=kwargs, lowest_ind=lowest_ind)
    median_t_width_text = f'median T width: {median_t_width}'
    plt.title(f'{histogram_title} - {median_t_width_text}')
    plt.xlabel('T Width')
    plt.ylabel('Count')
    plt.savefig(f'{figdir}_width_histogram_square.pdf')
    df_histogram_trotter_square = pd.DataFrame({'bin': circuit_histogram[1][:-1], \
                                            'count': circuit_histogram[0]})
    df_histogram_trotter_square.to_csv(f'{widthdir}widths_square.csv', sep=',',index=False)

def get_T_depth(cpt_circuit: AbstractCircuit):
    t_depth = 0
    for moment in cpt_circuit:
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                t_depth += 1
                break
    return t_depth

def gen_resource_estimate(cpt_circuit: AbstractCircuit,
                          is_extrapolated: bool,
                          total_steps:int = -1,
                          circ_occurences:int = -1) -> dict:
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
    t_depth_single_wire = get_T_depth_wire(cpt_circuit)
    clifford_count = gate_count - t_count
    circuit_depth = len(cpt_circuit)

    resource_estimate = {
        'num_qubits': num_qubits,
        'gate_count': gate_count,
        'circuit_depth': circuit_depth,
        't_count': t_count,
        't_depth': t_depth,
        'max_t_depth_wire': t_depth_single_wire,
        'clifford_count': clifford_count,
    }

    if total_steps > 0 and is_extrapolated:
        resource_estimate['t_depth'] = resource_estimate['t_depth'] * total_steps
        resource_estimate['t_count'] = resource_estimate['t_count'] * total_steps
        resource_estimate['max_t_depth_wire'] = resource_estimate['max_t_depth_wire'] * total_steps
        resource_estimate['gate_count'] = resource_estimate['gate_count'] * total_steps
        resource_estimate['circuit_depth'] = resource_estimate['circuit_depth'] * total_steps
        resource_estimate['clifford_count'] = resource_estimate['clifford_count'] * total_steps

    if circ_occurences > 0:
        resource_estimate['subcicruit_occurrences'] = circ_occurences

    return resource_estimate


def estimate_cpt_resources(
        cpt_circuit: AbstractCircuit,
        outdir: str,
        is_extrapolated:bool,
        circuit_name:str,
        algo_name:str,
        magnus_steps:int=1,
        trotter_steps:int=1
    ):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    total_steps = trotter_steps * magnus_steps
    resource_estimate = gen_resource_estimate(cpt_circuit=cpt_circuit,
                                              is_extrapolated=is_extrapolated,
                                              total_steps=total_steps)
    resource_estimate['circuit_occurences'] = 1
    resource_estimate['subcircuit_info'] = {
        f'{algo_name}': {
            'num_qubits': resource_estimate['num_qubits'],
            'gate_count': resource_estimate['gate_count']//total_steps,
            'clifford_count': resource_estimate['clifford_count']//total_steps,
            't_count': resource_estimate['t_count']//total_steps,
            't_depth': resource_estimate['t_depth']//total_steps,
            'max_t_depth_wire': resource_estimate['max_t_depth_wire']//total_steps,
            'circuit_depth': resource_estimate['circuit_depth']//total_steps,
            'subcicruit_occurrences': total_steps,
            'subcircuit_info': {}
        }
    }
    
    outfile_data = f'{outdir}{circuit_name}_high_level.json'
    re_as_json(resource_estimate, [], outfile_data, '')

def circuit_estimate(
        circuit:AbstractCircuit,
        outdir: str,
        numsteps: int,
        circuit_name: str,
        algo_name:str,
        write_circuits:bool = False
    ) -> AbstractCircuit:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    subcircuit_counts = dict()
    for moment in circuit:
        for operation in moment:
            gate_type = type(operation.gate)
            gate_type_name = f'{str(gate_type).split("/")[-1][:-2]}'
            if gate_type in subcircuit_counts:
                subcircuit_counts[gate_type][0] += 1
            else:
                t0 = time.perf_counter()
                decomposed_circuit = circuit_decompose_once(circuit_decompose_once(Circuit(operation)))
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
                    QasmOutput(
                        decomposed_circuit,
                        decomposed_circuit.all_qubits()).save(outfile_qasm_decomposed)

                    QasmOutput(cpt_circuit,
                               cpt_circuit.all_qubits()).save(outfile_qasm_cpt)
    
                subcircuit_counts[gate_type] = [1, cpt_circuit, gate_type_name]
    total_gate_count = 0
    total_gate_depth = 0
    total_T_count = 0
    total_T_depth = 0
    total_T_depth_wire = 0
    total_clifford_count = 0
    subcircuit_re = []
    for gate in subcircuit_counts:
        subcircuit = subcircuit_counts[gate][1]
        subcircuit_name = subcircuit_counts[gate][2]
        resource_estimate = gen_resource_estimate(subcircuit,
                                                  is_extrapolated=False,
                                                  circ_occurences=subcircuit_counts[gate][0])
        subcircuit_info = {subcircuit_name:resource_estimate}
        subcircuit_re.append(subcircuit_info)

        gate_count = resource_estimate['gate_count']
        gate_depth = resource_estimate['circuit_depth']
        t_depth = resource_estimate['t_depth']
        t_depth_wire = resource_estimate['max_t_depth_wire']
        t_count = resource_estimate['t_count']
        clifford_count = resource_estimate['clifford_count']
        
        total_gate_count += subcircuit_counts[gate][0] * gate_count * numsteps
        total_gate_depth += subcircuit_counts[gate][0] * gate_depth * numsteps
        total_T_depth += subcircuit_counts[gate][0] * t_depth * numsteps
        total_T_depth_wire += subcircuit_counts[gate][0] * t_depth_wire * numsteps
        total_T_count += subcircuit_counts[gate][0] * t_count * numsteps
        total_clifford_count += subcircuit_counts[gate][0] * clifford_count * numsteps

    outfile_data = f'{outdir}{circuit_name}_high_level.json'
    total_resources = {
        'num_qubits': len(circuit.all_qubits()),
        'gate_count': total_gate_count,
        'circuit_depth': total_gate_depth,
        't_count': total_T_count,
        't_depth': total_T_depth,
        'max_t_depth_wire': total_T_depth_wire,
        'clifford_count': total_clifford_count,
        'circuit_occurences': 1,
        'subcircuit_info': {
            f'{algo_name}': {
                'num_qubits': len(circuit.all_qubits()),
                'gate_count': total_gate_count//numsteps,
                'circuit_depth': total_gate_depth//numsteps,
                't_count': total_T_count//numsteps,
                't_depth': total_T_depth//numsteps,
                'max_t_depth_wire': total_T_depth_wire//numsteps,
                'clifford_count': total_clifford_count//numsteps,
                'circuit_occurences': numsteps,
            }
        }
    }
    re_as_json(algo_name=algo_name,
               main_estimate=total_resources,
               estimates=subcircuit_re,
               file_name=outfile_data)

def re_as_json(main_estimate:dict, estimates:list[dict], file_name:str, algo_name:str) -> None:
    if estimates:
        main_estimate['subcircuit_info'][algo_name]['subcircuit_info'] = {}
        for op in estimates:
            for op_key in op.keys():
                main_estimate['subcircuit_info'][algo_name]['subcircuit_info'][op_key] = op[op_key]
    with open(file_name, 'w') as f:
            json.dump(main_estimate, f,
                    indent=4,
                    separators=(',', ': '))