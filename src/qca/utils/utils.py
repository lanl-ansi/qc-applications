import os
from re import findall
from typing import Union
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
    number = findall(r'\d+', string)
    return int(number[0]) if number else None


def count_gates(cpt_circuit: AbstractCircuit) -> int:
    """
    For some clifford + T circuit, iterate through it and sum up the number
    of gates
    
    :param cpt_circuit: A clifford + T circuit
    :type cpt_circuit: cirq.AbstractCircuit
    :return: The number of gates in the clifford + T circuit
    :rtype: integer

    """

    count = 0
    for moment in cpt_circuit:
        count += len(moment)
    return count


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

    t_counts = {}
    gate_counts = {}
    clifford_counts = {}
    subcircuit_counts = {}
    subcircuit_depths = {}

    outfile_data = f'{outdir}{circuit_name}_high_level.dat'

    for moment in circuit:
        for operation in moment:
            gate_type = type(operation.gate)
            if gate_type in subcircuit_counts:
                subcircuit_counts[gate_type] += 1
            else:
                decomposed_circuit = circuit_decompose_once(
                    circuit_decompose_once(Circuit(operation)))
                cpt_circuit = clifford_plus_t_direct_transform(
                    decomposed_circuit)

                if write_circuits:
                    outfile_qasm_decomposed = f'{outdir}{str(gate_type)[8:-2]}.decomposed.qasm'
                    outfile_qasm_cpt = f'{outdir}{str(gate_type)[8:-2]}.cpt.qasm'
                    QasmOutput(
                        decomposed_circuit,
                        decomposed_circuit.all_qubits()).save(outfile_qasm_decomposed)
                    QasmOutput(cpt_circuit,
                               cpt_circuit.all_qubits()).save(outfile_qasm_cpt)

                subcircuit_counts[gate_type] = 1
                subcircuit_depths[gate_type] = len(cpt_circuit)
                t_counts[gate_type] = count_T_gates(cpt_circuit)
                gate_counts[gate_type] = count_gates(cpt_circuit)
                clifford_counts[gate_type] = gate_counts[gate_type] - \
                    t_counts[gate_type]

    total_gate_count = 0
    total_gate_depth = 0
    total_T_count = 0
    total_clifford_count = 0
    for gate in subcircuit_counts:
        total_gate_count += subcircuit_counts[gate] * gate_counts[gate]
        total_gate_depth += subcircuit_counts[gate] * subcircuit_depths[gate]
        total_T_count += subcircuit_counts[gate] * t_counts[gate]
        total_clifford_count += subcircuit_counts[gate] * clifford_counts[gate]
    with open(outfile_data, 'w', encoding='utf-8') as file:
        file.write(f'Logical Qubit Count: {len(circuit.all_qubits())}\n')
        file.write(f'Total Gate Count: {total_gate_count}\n')
        file.write(f'Total Gate Depth: {total_gate_depth}\n')
        file.write(f'Total T Count: {total_T_count}\n')
        file.write(f'Total Clifford Count: {total_clifford_count}\n')
        file.write('Subcircuit Info:\n')
        for gate in subcircuit_counts:
            file.write(f'{gate}\n')
            file.write(f'Subcircuit Occurrences: {subcircuit_counts[gate]}\n')
            file.write(f'Gate count: {gate_counts[gate]}\n')
            file.write(f'Gate depth: {subcircuit_counts[gate]}\n')
            file.write(f'T Count: {t_counts[gate]}\n')
            file.write(f'Clifford Count: {clifford_counts[gate]}\n')
