import unittest
from math import pow

import cirq

import numpy as np

from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform

from qca.utils.utils import (
    count_gates,
    count_T_gates,
    get_T_depth,
    get_T_depth_wire,
    gen_resource_estimate,
    scale_resource
)


class UtilsTest(unittest.TestCase):
    def test_count_gates(self):
        qubits = [cirq.LineQubit(i) for i in range(4)]

        layer_rz = [cirq.Rz(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_rx = [cirq.Rx(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_measurement = [cirq.measure(qubit) for qubit in qubits]
        circuit = cirq.Circuit()
        circuit.append(layer_rz)
        circuit.append(layer_rx)
        circuit.append(layer_measurement)
        self.assertEqual(count_gates(circuit), 12)

    def test_T_gate_info(self):
        qubits = [cirq.LineQubit(i) for i in range(4)]

        layer_rz = [cirq.Rz(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_rx = [cirq.Rx(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_measurement = [cirq.measure(qubit) for qubit in qubits]
        circuit = cirq.Circuit()
        circuit.append(layer_rz)
        circuit.append(layer_rx)
        circuit.append(layer_measurement)
        circ_cpt =  clifford_plus_t_direct_transform(circuit)

        self.assertEqual(count_T_gates(circ_cpt), 8)
        self.assertEqual(get_T_depth(circ_cpt), 2)
        self.assertEqual(get_T_depth_wire(circ_cpt), 2)


    def test_clifford_info(self):
        qubits = [cirq.LineQubit(i) for i in range(4)]

        layer_rz = [cirq.Rz(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_rx = [cirq.Rx(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_measurement = [cirq.measure(qubit) for qubit in qubits]
        circuit = cirq.Circuit()
        circuit.append(layer_rz)
        circuit.append(layer_rx)
        circuit.append(layer_measurement)
        circ_cpt =  clifford_plus_t_direct_transform(circuit)

        gate_count = count_gates(circ_cpt)
        t_count = count_T_gates(circ_cpt)
        clifford_count = gate_count - t_count
        circ_depth = len(circ_cpt)
        self.assertEqual(circ_depth, 5)
        self.assertEqual(clifford_count, 12)

    def test_get_resource_estimate(self):
        qubits = [cirq.LineQubit(i) for i in range(4)]

        layer_rz = [cirq.Rz(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_rx = [cirq.Rx(rads = np.pi/4).on(qubit) for qubit in qubits]
        layer_measurement = [cirq.measure(qubit) for qubit in qubits]
        circuit = cirq.Circuit()
        circuit.append(layer_rz)
        circuit.append(layer_rx)
        circuit.append(layer_measurement)
        circ_cpt =  clifford_plus_t_direct_transform(circuit)
        circ_estimate = gen_resource_estimate(circ_cpt)
        correct_estimate = {'num_qubits': 4,
                            't_count': 8,
                            't_depth': 2,
                            'gate_count': 20,
                            'clifford_count': 12,
                            'circuit_depth': 5}
        self.assertEqual(circ_estimate, correct_estimate)

    def test_scale_t_count(self):
        t_count = 100
        nsteps = 2 
        bits_precision = 4
        scaled_t_count = scale_resource(t_count, nsteps, bits_precision)
        
        expect = t_count * nsteps * pow(2, bits_precision -1)
        self.assertEqual(scaled_t_count, expect)

    def scale_toffolis(self):
        tofolli_count = 30
        t_count = tofolli_count * 4
        bits_precision = 10
        scaled_t_count = scale_resource(resource=t_count, bits_precision=bits_precision)
        
        expect = t_count * pow(2, bits_precision-1) 
        self.assertEqual(scaled_t_count, expect)
    
    def verify_nop_scaler(self):
        t_count = 0
        nsteps = 2
        bits_precision = 10
        scaled_t_count = scale_resource(resource=t_count, bits_precision=bits_precision, nsteps=nsteps)

        expect = 0
        self.assertEqual(scaled_t_count, expect)

if __name__ == '__main__':
    unittest.main()
