import os
import re
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from pyLIQTR.utils.utils import count_T_gates
from cirq import Circuit, QasmOutput, AbstractCircuit
from pyLIQTR.utils.qsp_helpers import circuit_decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform


def extract_number(string: str) -> Union[int, None]:
    """
    For some string, return a number from it if it exists

    :param string: the string to evaluate
    :type string: str type
    :return: Either the number found or None
    :rtype: Union[int, None]

    """
    number = re.findall(r'\d+', string)
    return int(number[0]) if number else None


def count_gates(cpt_circuit: AbstractCircuit) -> int:
    """
    For some clifford + T circuit, iterate through it and sum up the number
    of gates

    :param cpt_circuit: A clifford + T circuit
    :type cpt_circuit: cirq.AbstractCircuit
    :return: The number of gates in the clifford + T circuit
    :rtype: int

    """
    count = 0
    for moment in cpt_circuit:
        count += len(moment)
    return count


def get_T_depth_wire(cpt_circuit: AbstractCircuit) -> int:
    """
    Grab the maximum number of T-gates on a wire.  This may be more optimistic than
    number of layers with T-gates.  Perhaps good to treat as lower bound
    for an implementation

    :param cpt_circuit: A clifford + T circuit
    :type cpt_circuit: cirq.AbstractCircuit
    :return: The maximum T depth on a wire
    :rtype: int
    """
    count_dict = {}
    for moment in cpt_circuit:
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                reg_label = opstr[opstr.find("(") + 1:opstr.find(")")]
                if reg_label not in count_dict:
                    count_dict[reg_label] = 1
                else:
                    count_dict[reg_label] += 1
    max_depth = 0
    for register in count_dict:
        if count_dict[register] > max_depth:
            max_depth = count_dict[register]
    return max_depth


def plot_T_step_histogram(
        cpt_circuit: AbstractCircuit,
        kwargs,
        lowest_ind: int = 0) -> plt.hist:
    """
    For some clifford + T circuit, grab its corresponding histogram

    :param cpt_circuit: A clifford + T circuit
    :type cpt_circuit: cirq.AbstractCircuit
    :param kwargs: Additional arguments for plotting the histogram
    :param lowest_ind: Lowest index to plot
    :type lowest_ind: int
    :return: The corresponding histogram
    :rtype: plt.hist

    """
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
                   histogram_title: str,
                   figdir: str,
                   widthdir: str,
                   lowest_ind: int = 0,
                   **kwargs) -> None:
    """
    For some clifford + T circuit, plot its histogram

    :param cpt_circuit: A clifford + T circuit
    :type cpt_circuit: cirq.AbstractCircuit
    :param histogram_title: The title to give your histogram
    :type histogram_title: str
    :param widthdir: The T width
    :type widthdir: str
    :param lowest_ind: Lowest index to plot
    :type lowest_ind: int
    :param kwargs: Additional arguments for plotting the histogram
    :return: None

    """
    circuit_histogram = plot_T_step_histogram(
        cpt_circuit, kwargs=kwargs, lowest_ind=lowest_ind)
    plt.title(histogram_title)
    plt.xlabel('T Width')
    plt.ylabel('Count')
    plt.savefig(f'{figdir}_width_histogram_square.pdf')
    df_histogram_trotter_square = pd.DataFrame(
        {'bin': circuit_histogram[1][:-1], 'count': circuit_histogram[0]})
    df_histogram_trotter_square.to_csv(
        f'{widthdir}widths_square.csv', sep=',', index=False)


def get_T_depth(cpt_circuit: AbstractCircuit) -> int:
    """
    For some clifford + T circuit, grab the total T depth

    :param cpt_circuit: A clifford + T circuit
    :type cpt_circuit: cirq.AbstractCirquit
    :return: The T depth of the circuit
    :rtype: int

    """
    t_depth = 0
    for moment in cpt_circuit:
        for operator in moment:
            opstr = str(operator)
            if opstr[0] == 'T':
                t_depth += 1
                break
    return t_depth


def estimate_gsee(
        circuit: Circuit,
        outdir: str,
        circuit_name: str = 'gse_circuit',
        write_circuits: bool = False) -> None:
    """
    Perform ground state energy estimation (gsee) on some circuit and write out the
    results if desired

    :param circuit: The circuit to perform gsee on
    :type circuit: cirq.Circuit
    :param outdir: The directory to save the results to
    :type outdir: str
    :param circuit_name: The name of the circuit, defaults to gse_circuit
    :type circuit_name: str
    :param write_circuits: Flag to denote whether or not to save the results to disk
    :type write_circuits: bool
    :return: None

   """

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    subcircuit_counts = dict()
    t_counts = dict()
    clifford_counts = dict()
    gate_counts = dict()
    subcircuit_depths = dict()
    
    outfile_data = f'{outdir}{circuit_name}_high_level.dat'
    for moment in circuit:
        for operation in moment:
            gate_type = type(operation.gate)
            if gate_type in subcircuit_counts:
                subcircuit_counts[gate_type] += 1
            else:
                decomposed_circuit = circuit_decompose_once(circuit_decompose_once(Circuit(operation)))
                cpt_circuit = clifford_plus_t_direct_transform(decomposed_circuit)
                
                outfile_qasm_decomposed = f'{outdir}{str(gate_type)[8:-2]}.decomposed.qasm'
                outfile_qasm_cpt = f'{outdir}{str(gate_type)[8:-2]}.cpt.qasm'
                
                if write_circuits:
                    QasmOutput(
                        decomposed_circuit,
                        decomposed_circuit.all_qubits()).save(outfile_qasm_decomposed)

                    QasmOutput(cpt_circuit,
                               cpt_circuit.all_qubits()).save(outfile_qasm_cpt)
    
                subcircuit_counts[gate_type] = 1
                subcircuit_depths[gate_type] = len(cpt_circuit)
                t_counts[gate_type] = count_T_gates(cpt_circuit)
                gate_counts[gate_type] = count_gates(cpt_circuit)
                clifford_counts[gate_type] = gate_counts[gate_type] - t_counts[gate_type]
                
                
    total_gate_count = 0
    total_gate_depth = 0
    total_T_count = 0
    total_clifford_count = 0
    for gate in subcircuit_counts:
        total_gate_count += subcircuit_counts[gate] * gate_counts[gate]
        total_gate_depth += subcircuit_counts[gate] * subcircuit_depths[gate]
        total_T_count += subcircuit_counts[gate] * t_counts[gate]
        total_clifford_count += subcircuit_counts[gate] * clifford_counts[gate]
    with open(outfile_data, 'w') as f:
        f.write(str("Logical Qubit Count:"+str(len(circuit.all_qubits()))+"\n"))
        f.write(str("Total Gate Count:"+str(total_gate_count)+"\n"))
        f.write(str("Total Gate Depth:"+str(total_gate_depth)+"\n"))
        f.write(str("Total T Count:"+str(total_T_count)+"\n"))
        f.write(str("Total Clifford Count:"+str(total_clifford_count)+"\n"))
        f.write("Subcircuit Info:\n")
        for gate in subcircuit_counts:
            f.write(str(str(gate)+"\n"))
            f.write(str("Subcircuit Occurrences:"+str(subcircuit_counts[gate])+"\n"))
            f.write(str("Gate Count:"+str(gate_counts[gate])+"\n"))
            f.write(str("Gate Depth:"+str(subcircuit_depths[gate])+"\n"))
            f.write(str("T Count:"+str(t_counts[gate])+"\n"))
            f.write(str("Clifford Count:"+str(clifford_counts[gate])+"\n"))