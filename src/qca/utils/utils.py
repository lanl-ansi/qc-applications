import os
import re
import json
import time
from cirq import Circuit, QasmOutput, AbstractCircuit
from pyLIQTR.utils.qsp_helpers import circuit_decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from pyLIQTR.utils.utils import count_T_gates
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_T_step_histogram(cpt_circuit:AbstractCircuit, kwargs, lowest_ind:int=0) -> plt.hist:
    t_widths = [0] * len(cpt_circuit)
    for i, moment in enumerate(cpt_circuit):
        width = 0
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                width += 1
        t_widths[i] = width
    bins = range(max(t_widths))
    histogram = plt.hist(t_widths, bins[lowest_ind:-1], **kwargs)
    return histogram

def plot_histogram(cpt_circuit: AbstractCircuit,
                   histogram_title:str,
                   figdir:str,
                   widthdir:str,
                   lowest_ind:int=0,
                   **kwargs) -> None:
    circuit_histogram = plot_T_step_histogram(cpt_circuit, kwargs=kwargs, lowest_ind=lowest_ind)
    plt.title(histogram_title)
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
                          trotter_steps:int = -1,
                          circ_occurences:int = -1) -> dict:
    '''
    Given some clifford + T circuit and a given filename, we grab the logical resource estimates
    from the circuit and then write it to disk. The function also returns the resource dictionary
    if the user needs it.

    trotter_steps is a flag denoting if the circuit was estimated through trotterization. If so, the
    user should specify the number of steps required.  
    '''
    t_count = count_T_gates(cpt_circuit)
    t_depth = get_T_depth(cpt_circuit)
    t_depth_single_wire = get_T_depth_wire(cpt_circuit)
    gate_count = count_gates(cpt_circuit)

    resource_estimate = {'num_qubits': len(cpt_circuit.all_qubits()),
                        't_count': t_count,
                        't_depth': t_depth,
                        'max_t_depth_wire': t_depth_single_wire,
                        'gate_count': gate_count,
                        'clifford_count': gate_count - t_count,
                        'circuit_depth': len(cpt_circuit)
                        }
    if trotter_steps > 0:
        resource_estimate['total_t_depth'] = t_depth * trotter_steps
        resource_estimate['max_t_count_single_wire'] = t_depth_single_wire * trotter_steps
        resource_estimate['trotter_steps_required'] = trotter_steps
    if circ_occurences > 0:
        resource_estimate['circuit occurrences'] = circ_occurences

    return resource_estimate

def re_as_json(main_estimate:dict, estimates:list[dict], file_name:str) -> None:
    if estimates:
        main_estimate['subcircuit_info'] = estimates
    with open(file_name, 'w') as f:
            json.dump(main_estimate, f,
                    indent=4,
                    separators=(',', ': '))
    
def estimate_gsee(
        circuit: Circuit,
        outdir: str,
        circuit_name: str = 'gse_circuit',
        write_circuits: bool = False) -> None:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    subcircuit_counts = dict()

    for moment in circuit:
        for operation in moment:
            gate_type = type(operation.gate)
            if gate_type in subcircuit_counts:
                subcircuit_counts[gate_type][0] += 1
            else:
                gate_type_name = f'{str(gate_type)[8:-2]}'
                t0 = time.perf_counter()
                decomposed_circuit = circuit_decompose_once(circuit_decompose_once(Circuit(operation)))
                t1 = time.perf_counter()
                elapsed = t1-t0
                print(f'   Time to decompose high level {gate_type_name} circuit: {elapsed} seconds ')

                t0 = time.perf_counter()
                cpt_circuit = clifford_plus_t_direct_transform(decomposed_circuit)
                t1 = time.perf_counter()
                elapsed = t1-t0
                print(f'   Time to transform decomposed {gate_type_name} circuit to Clifford+T: {elapsed} seconds')
                
                if write_circuits:
                    outfile_qasm_decomposed = f'{outdir}{gate_type_name}.decomposed.qasm'
                    outfile_qasm_cpt = f'{outdir}{gate_type_name}.cpt.qasm'
                    QasmOutput(
                        decomposed_circuit,
                        decomposed_circuit.all_qubits()).save(outfile_qasm_decomposed)

                    QasmOutput(cpt_circuit,
                               cpt_circuit.all_qubits()).save(outfile_qasm_cpt)
    
                subcircuit_counts[gate_type] = [1, cpt_circuit]
                
    total_gate_count = 0
    total_gate_depth = 0
    total_T_count = 0
    total_T_depth = 0
    total_T_depth_wire = 0
    total_clifford_count = 0
    for gate in subcircuit_counts:
        subcircuit = subcircuit_counts[gate][1]
        resource_estimate = gen_resource_estimate(subcircuit,
                                                  f'{outdir}{circuit_name}_{gate}.json',
                                                  circ_occurences=subcircuit_counts[gate][0])
        gate_count = resource_estimate['gate_count']
        gate_depth = resource_estimate['circuit_depth']
        t_depth = resource_estimate['t_depth']
        t_depth_wire = resource_estimate['t_depth_wire']
        t_count = resource_estimate['t_count']
        clifford_count = resource_estimate['clifford_count']
        
        total_gate_count += subcircuit_counts[gate][0] * gate_count
        total_gate_depth += subcircuit_counts[gate][0] * gate_depth
        total_T_depth += subcircuit_counts[gate][0] * t_depth
        total_T_depth_wire += subcircuit_counts[gate][0] * t_depth_wire
        total_T_count += subcircuit_counts[gate][0] * t_count
        total_clifford_count += subcircuit_counts[gate][0] * clifford_count

    outfile_data = f'{outdir}{circuit_name}_high_level.dat.json'
    total_resources = {
        'num_qubits': len(subcircuit.all_qubits()),
        'gate_count': total_gate_count,
        'circuit_depth': total_gate_depth,
        't_count': total_T_count,
        't_depth': total_T_depth,
        't_depth_wire': total_T_depth_wire,
        'clifford_count': total_clifford_count
    }
    with open(outfile_data, 'w') as f:
        json.dump(total_resources, f,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ': '))