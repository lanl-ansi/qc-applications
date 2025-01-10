import os
import re
import json
import time
from statistics import median
from dataclasses import dataclass, field, asdict

import pandas as pd
import logging as log
import matplotlib.pyplot as plt

import cirq

from pyLIQTR.utils.printing import openqasm
from pyLIQTR.utils.utils import count_T_gates
import pyLIQTR.utils.resource_analysis as pyLRA
import pyLIQTR.utils.global_ancilla_manager as gam
from pyLIQTR.utils.circuit_decomposition import circuit_decompose_multi
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform


@dataclass
class EstimateMetaData:
    id: str
    name: str
    category: str
    size: str
    task: str
    is_extrapolated: bool=field(default=False, kw_only=True)
    gate_synth_accuracy: int | float = field(default=10,kw_only=True)
    value_per_circuit: float | None=field(default=None, kw_only=True)
    value_per_t_gate: float | None=field(default=None,kw_only=True)
    repetitions_per_application: int | None=field(default=None, kw_only=True)

@dataclass
class GSEEMetaData(EstimateMetaData):
    evolution_time: float
    bits_precision: int
    trotter_order: int
    nsteps: int

@dataclass
class CatalystMetaData(GSEEMetaData):
    basis: str
    occupied_orbitals:int
    unoccupied_orbitals:int

@dataclass
class TrotterMetaData(EstimateMetaData):
    evolution_time: float #NOTE: This is JT in the current implementation
    trotter_order: int
    energy_precision: float
    nsteps: int=None 

@dataclass
class QSPMetaData(EstimateMetaData):
    evolution_time: float #NOTE: This is JT in the current implementation
    nsteps: int
    energy_precision: float

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
            if opstr.startswith('T'):
                t_depth += 1
                break
    return t_depth

def complete_decomposition(circuit: cirq.AbstractCircuit) -> cirq.AbstractCircuit:
    prev_decomp = None
    context = cirq.DecompositionContext(gam.gam)
    while circuit != prev_decomp:
        prev_decomp = circuit
        circuit = circuit_decompose_multi(circuit, 1, context=context)
    return circuit

def gen_resource_estimate(
        cpt_circuit: cirq.AbstractCircuit,
        circ_occurences:int | None=None,
    ) -> dict:
    '''
    Given some clifford + T circuit and a given filename, we grab the logical resource estimates
    from the circuit and returns the resource dictionary 
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

    if circ_occurences:
        resource_estimate['subcircuit_occurences'] = circ_occurences

    return resource_estimate

def scale_resource(resource: int, nsteps: int | None=None, bits_precision: float | None = None) -> int:
    scaling_factor = 0
    if resource != 0:
        scaled_steps = 0
        scaled_bits = 0
        if nsteps and nsteps > 0:
            scaled_steps = nsteps
        if bits_precision and bits_precision > 0:
            scaled_bits = pow(2, bits_precision - 1)
        
        if scaled_steps and scaled_bits:
            scaling_factor = scaled_steps * scaled_bits
        elif scaled_steps and not scaled_bits:
            scaling_factor = scaled_steps
        elif scaled_bits and not scaled_steps:
            scaling_factor = scaled_bits
        if scaling_factor:
            return int(resource * scaling_factor)
    return resource


def estimate_cpt_resources(
        cpt_circuit: cirq.AbstractCircuit,
        algo_name:str,
        is_extrapolated:bool,
        include_nested_resources:bool,
        nsteps: int|None=None
    ):
    logical_re = {
        'Logical_Abstract':  gen_resource_estimate(
            cpt_circuit=cpt_circuit,
        )
    }
    if nsteps and is_extrapolated:
        highest_scope = logical_re['Logical_Abstract']
        for key in highest_scope:
            highest_scope[key] = scale_resource(highest_scope[key], nsteps)

    logical_re['Logical_Abstract']['subcircuit_occurences'] = 1
    if include_nested_resources and nsteps:
        logical_re['Logical_Abstract']['subcircuit_info'] = grab_single_step_estimates(
            len(cpt_circuit.all_qubits()),
            logical_re['Logical_Abstract'],
            algo_name,
            nsteps
        )
    else:
        logical_re['Logical_Abstract']['subcircuit_info'] = {}
    return logical_re

def grab_single_step_estimates(num_qubits: int, main_estimates: dict, algo_name:str, nsteps: int) -> dict:
    return {
        f'{algo_name}': {
            'num_qubits': num_qubits,
            't_count': main_estimates['t_count']//nsteps,
            'circuit_depth': main_estimates['circuit_depth']//nsteps,
            'gate_count': main_estimates['gate_count']//nsteps,
            't_depth': main_estimates['t_depth']//nsteps,
            'clifford_count': main_estimates['clifford_count']//nsteps,
            'subcircuit_occurences': nsteps,
            'subcircuit_info': {}
        }
    }

def write_qasm(
        circuit: cirq.AbstractCircuit,
        fname: str
    ):
    qasm_iter = openqasm(
        circuit=circuit,
        rotation_allowed=True,
    )
    qasm_str = ' \n'.join([qasm for qasm in qasm_iter])
    with open(fname, 'w', encoding='utf-8') as file:
        file.write(qasm_str)

def circuit_estimate(
        circuit: cirq.AbstractCircuit,
        outdir: str,
        nsteps: int,
        algo_name: str,
        include_nested_resources:bool,
        is_extrapolated:bool,
        gate_synth_accuracy: int | float = 10,
        bits_precision:int=1,
        write_circuits:bool = False
    ) -> dict:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    subcircuit_counts = {}
    for moment in circuit:
        for operation in moment:
            gate_type = type(operation.gate)
            gate_type_name = gate_type.__name__
            if gate_type in subcircuit_counts:
                subcircuit_counts[gate_type][0] += 1
            else:
                t0 = time.perf_counter()
                # decomposed_circuit = cirq.Circuit(cirq.decompose(operation))
                decomposed_circuit = complete_decomposition(cirq.Circuit(operation))
                t1 = time.perf_counter()
                decomposed_elapsed = t1-t0
                print(f'   Time to decompose high level {gate_type_name} circuit: {decomposed_elapsed} seconds ')
                t0 = time.perf_counter()
                cpt_circuit = clifford_plus_t_direct_transform(circuit = decomposed_circuit, gate_precision=gate_synth_accuracy)
                t1 = time.perf_counter()
                cpt_elapsed = t1-t0
                print(f'   Time to transform decomposed {gate_type_name} circuit to Clifford+T: {cpt_elapsed} seconds')
                if write_circuits:
                    outfile_qasm_decomposed = f'{outdir}{gate_type_name}.decomposed.qasm'
                    write_qasm(
                        circuit=decomposed_circuit,
                        fname=outfile_qasm_decomposed
                    )
                    outfile_qasm_cpt = f'{outdir}{gate_type_name}.cpt.qasm'
                    write_qasm(
                        circuit=cpt_circuit,
                        fname=outfile_qasm_cpt
                    )

                subcircuit_counts[gate_type] = [1, cpt_circuit, gate_type_name]

    total_gate_count = 0
    total_gate_depth = 0
    total_T_count = 0
    total_T_depth = 0
    total_clifford_count = 0
    subcircuit_re = []
    for gate in subcircuit_counts:
        subcircuit_occurences = subcircuit_counts[gate][0]
        subcircuit = subcircuit_counts[gate][1]
        subcircuit_name = subcircuit_counts[gate][2]
        resource_estimate = gen_resource_estimate(
            subcircuit,
            circ_occurences=subcircuit_occurences,
        )
        subcircuit_info = {subcircuit_name:resource_estimate}
        subcircuit_re.append(subcircuit_info)

        gate_count = resource_estimate['gate_count']
        gate_depth = resource_estimate['circuit_depth']
        t_depth = resource_estimate['t_depth']
        t_count = resource_estimate['t_count']
        clifford_count = resource_estimate['clifford_count']
        
        curr_gate_count = subcircuit_occurences * gate_count
        curr_gate_depth = subcircuit_occurences * gate_depth 
        curr_t_depth = subcircuit_occurences * t_depth
        curr_t_count = subcircuit_occurences * t_count
        curr_clifford_count = subcircuit_occurences * clifford_count

        if (nsteps or bits_precision) and is_extrapolated:
            total_gate_count += scale_resource(curr_gate_count, nsteps, bits_precision)
            total_gate_depth += scale_resource(curr_gate_depth, nsteps, bits_precision)
            total_T_depth += scale_resource(curr_t_depth, nsteps, bits_precision)
            total_T_count += scale_resource(curr_t_count, nsteps, bits_precision)
            total_clifford_count += scale_resource(curr_clifford_count, nsteps, bits_precision)
 
    main_estimates = {
        'Logical_Abstract': {
            'num_qubits': len(circuit.all_qubits()),
            't_count': total_T_count,
            'circuit_depth': total_gate_depth,
            'gate_count': total_gate_count,
            't_depth': total_T_depth,
            'clifford_count': total_clifford_count,
            'subcircuit_occurences': 1,
            'subcircuit_info': {}
        }
    }
    if include_nested_resources and subcircuit_re and nsteps:
        main_estimates['Logical_Abstract']['subcircuit_info'] = grab_single_step_estimates(
            len(circuit.all_qubits()),
            main_estimates['Logical_Abstract'],
            algo_name,
            nsteps
        )
        for op in subcircuit_re:
            for op_key in op.keys():
                main_estimates['Logical_Abstract']['subcircuit_info'][algo_name]['subcircuit_info'][op_key] = op[op_key]
    return main_estimates

#TODO: Implement method to properly format gate_synth_accuracy
def re_as_json(main_estimate:dict, outdir:str) -> None:
    with open(outdir, 'w') as f:
        json.dump(main_estimate, f,
            indent=4,
            separators=(',', ': '))

def gen_json(main_estimate: dict, outfile:str, metadatata: EstimateMetaData|None=None):
    if metadatata:
        re_metadata = asdict(metadatata)
        main_estimate = re_metadata | main_estimate
    re_as_json(main_estimate, outfile)

def grab_circuit_resources(circuit: cirq.AbstractCircuit,
                           outdir: str,
                           algo_name: str,
                           fname: str,
                           is_extrapolated:bool,
                           use_analytical: bool = False,
                           nsteps: int|None=None,
                           bits_precision:int|None=None,
                           metadata: EstimateMetaData|None=None,
                           write_circuits:bool=False,
                           include_nested_resources:bool=False,
                           gate_synth_accuracy: int | float=10) -> None:
    if not use_analytical:
        estimates = circuit_estimate(
            circuit=circuit,
            outdir=outdir,
            nsteps=nsteps,
            algo_name=algo_name,
            is_extrapolated=is_extrapolated,
            include_nested_resources=include_nested_resources,
            gate_synth_accuracy=gate_synth_accuracy,
            bits_precision=bits_precision,
            write_circuits=write_circuits,
        )
    else:
        if gate_synth_accuracy > 1:
            log.warning('gate_synth_accuracy is greater than 1. Converting it to 1e-{gate_synth_accuracy}')
            gate_synth_accuracy = float(f'1e-{gate_synth_accuracy}')

        logical_estimates = pyLRA.estimate_resources(
            circuit,
            rotation_gate_precision=gate_synth_accuracy,
            profile=False
        )
        estimates = {'Logical_Abstract':{
            'num_qubits': logical_estimates['LogicalQubits'],
            't_count': logical_estimates['T'],
            'clifford_count': logical_estimates['Clifford'],
            'gate_count': logical_estimates['T'] + logical_estimates['Clifford'],
        }}
        header = estimates['Logical_Abstract']
        if is_extrapolated:
            for resource in header:
                header[resource] = scale_resource(header[resource], nsteps, bits_precision)
        
        header['subcircuit_occurences'] = 1
        header['subcircuit_info'] = {}

    #calculate and insert value_per_t_gate
    if metadata != None:
        header = estimates['Logical_Abstract']
        if (metadata.value_per_circuit != None) and (metadata.repetitions_per_application != None):
            metadata.value_per_t_gate = metadata.value_per_circuit/header['t_count']
    
    outfile = f'{outdir}{fname}_re.json'
    gen_json(estimates, outfile, metadata)