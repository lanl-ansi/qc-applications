import numpy as np
import unittest
import cirq
from qca.utils.utils import count_gates

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

if __name__ == '__main__':
    unittest.main()
