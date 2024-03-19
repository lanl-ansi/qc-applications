import unittest
from pyLIQTR.utils.Hamiltonian import Hamiltonian as pyH
from networkx.generators.lattice import hexagonal_lattice_graph
from qca.utils.hamiltonian_utils import (
    nx_triangle_lattice,
    assign_hexagon_labels,
    generate_two_orbital_nx,
    generate_square_hamiltonian,
    nx_to_two_orbital_hamiltonian,
    assign_directional_triangular_labels,
    pyliqtr_hamiltonian_to_openfermion_qubit_operator
)


class HamiltonianUtilsTest(unittest.TestCase):
    def test_triangle_hamiltonian_generation(self):
        graph_triangle = nx_triangle_lattice(lattice_size=2)
        assign_directional_triangular_labels(graph_triangle, 2)
        edge_labels = dict([((n1, n2), d['label']) for n1, n2, d in graph_triangle.edges(data=True)])
        correct_labels = {((0,0), (1,0)): 'Z',
                          ((0,0), (0,1)): 'X',
                          ((0,0), (1, 1)): 'Y',
                          ((0, 1), (1, 1)): 'Z',
                          ((1, 0), (1, 1)): 'X'}
        for edge, label in edge_labels.items():
            self.assertTrue(edge in correct_labels.keys())
            self.assertEqual(label, correct_labels[edge])

    def test_hexagonal_labels(self):
        hexagon_graph = hexagonal_lattice_graph(1,1)
        assign_hexagon_labels(hexagon_graph, 'X', 'Y', 'Z')
        edge_labels = dict([((n1, n2), d['label']) for n1, n2, d in hexagon_graph.edges(data=True)])
        correct_labels = {((0, 0), (0,1)): 'Y',
                          ((0, 0), (1, 0)): 'Z',
                          ((0, 1), (0, 2)): 'X',
                          ((0, 2), (1, 2)): 'Z',
                          ((1, 0), (1, 1)): 'X',
                          ((1, 1), (1, 2)): 'Y'}
        for edge, label in edge_labels.items():
            self.assertTrue(edge in correct_labels.keys())
            self.assertEqual(label, correct_labels[edge])

    def test_grabbing_transverse_ising_terms(self):
        transverse_test = generate_square_hamiltonian(lattice_size=2, dim=2)[0]
        transverse_hamiltonian = ['XIII', 'IXII', 'IIXI', 'IIIX']
        self.assertEqual(len(transverse_test), len(transverse_hamiltonian))
        for idx, transverse_term in enumerate(transverse_test):
            correct_hamiltonian = transverse_hamiltonian[idx]
            self.assertEqual(transverse_term[0], correct_hamiltonian)

    def test_grabbing_longitudinal_ising_terms(self):
        longitudinal_test = generate_square_hamiltonian(lattice_size=2, dim=2)[1]
        longitudinal_hamiltonian = ['ZIZI', 'ZZII', 'IZIZ', 'IIZZ']
        self.assertEqual(len(longitudinal_hamiltonian), len(longitudinal_test))
        for idx, longitudinal_term in enumerate(longitudinal_test):
            correct_hamiltonian = longitudinal_hamiltonian[idx]
            self.assertEqual(longitudinal_term[0], correct_hamiltonian)

    def test_pyliqtr_hamiltonian_to_qubit_op(self):
        square_hamiltonian = generate_square_hamiltonian(lattice_size=2, dim=2)
        H_square = pyH(square_hamiltonian[0]+square_hamiltonian[1])
        openfermion_hamiltonian_square = pyliqtr_hamiltonian_to_openfermion_qubit_operator(H_square)
        correct_ops = ['X0', 'X1', 'X2', 'X3', 'Z0Z2', 'Z0Z1', 'Z1Z3', 'Z2Z3']
        for idx, op in enumerate(openfermion_hamiltonian_square):
            qubit_op = ''
            for term in op.terms:
                for ops in term:
                    qubit_op += f'{ops[1]}{ops[0]}'
            self.assertEqual(qubit_op, correct_ops[idx])

    def test_two_orbital_gen_test(self):
        two_orbital = generate_two_orbital_nx(2, 2)
        edge_labels = dict([((n1, n2), d['label'])
                           for n1, n2, d in two_orbital.edges(data=True)])
        correct_labels = {((0, 0, 0, 0), (0, 1, 0, 0)): '-t3',
                          ((0, 0, 0, 0), (1, 0, 0, 0)): '-t3',
                          ((0, 0, 0, 0), (1, 1, 0, 0)): '-t3',
                          ((0, 0, 0, 0), (1, 1, 1, 0)): '+t4',
                          ((0, 0, 0, 1), (0, 1, 0, 1)): '-t3',
                          ((0, 0, 0, 1), (1, 0, 0, 1)): '-t3',
                          ((0, 0, 0, 1), (1, 1, 0, 1)): '-t3',
                          ((0, 0, 0, 1), (1, 1, 1, 1)): '+t4',
                          ((0, 0, 1, 0), (1, 0, 1, 0)): '-t3',
                          ((0, 0, 1, 0), (0, 1, 1, 0)): '-t3',
                          ((0, 0, 1, 0), (1, 1, 1, 0)): '-t3',
                          ((0, 0, 1, 0), (1, 1, 0, 0)): '+t4',
                          ((0, 0, 1, 1), (1, 0, 1, 1)): '-t3',
                          ((0, 0, 1, 1), (0, 1, 1, 1)): '-t3',
                          ((0, 0, 1, 1), (1, 1, 1, 1)): '-t3',
                          ((0, 0, 1, 1), (1, 1, 0, 1)): '+t4',
                          ((0, 1, 0, 0), (1, 1, 0, 0)): '-t3',
                          ((0, 1, 0, 0), (1, 0, 0, 0)): '-t3',
                          ((0, 1, 0, 0), (1, 0, 1, 0)): '-t4',
                          ((0, 1, 0, 1), (1, 1, 0, 1)): '-t3',
                          ((0, 1, 0, 1), (1, 0, 0, 1)): '-t3',
                          ((0, 1, 0, 1), (1, 0, 1, 1)): '-t4',
                          ((0, 1, 1, 0), (1, 1, 1, 0)): '-t3',
                          ((0, 1, 1, 0), (1, 0, 1, 0)): '-t3',
                          ((0, 1, 1, 0), (1, 0, 0, 0)): '-t4',
                          ((0, 1, 1, 1), (1, 1, 1, 1)): '-t3',
                          ((0, 1, 1, 1), (1, 0, 1, 1)): '-t3',
                          ((0, 1, 1, 1), (1, 0, 0, 1)): '-t4',
                          ((1, 0, 0, 0), (1, 1, 0, 0)): '-t3',
                          ((1, 0, 0, 1), (1, 1, 0, 1)): '-t3',
                          ((1, 0, 1, 0), (1, 1, 1, 0)): '-t3',
                          ((1, 0, 1, 1), (1, 1, 1, 1)): '-t3'}
        for edge, label in edge_labels.items():
            self.assertTrue(edge in correct_labels.keys())
            self.assertEqual(label, correct_labels[edge])
    
    def test_nx_to_two_orbital_hamiltonian(self):
        two_orbital = generate_two_orbital_nx(2, 2)
        fermionic_two_orbital = nx_to_two_orbital_hamiltonian(
            two_orbital,
            -1,
            1.3,
            0.85,
            0.85,
            1
        )
        edge_labels = dict([((n1, n2), d['label']) for n1, n2, d in two_orbital.edges(data=True)]);
        correct_fermionic_terms = {((0, 1), (4, 0)): -0.85,
                                    ((4, 1), (0, 0)): -0.85,
                                    ((0, 1), (8, 0)): -0.85,
                                    ((8, 1), (0, 0)): -0.85,
                                    ((0, 1), (12, 0)): -0.85,
                                    ((12, 1), (0, 0)): -0.85,
                                    ((0, 1), (14, 0)): 0.85,
                                    ((14, 1), (0, 0)): 0.85,
                                    ((1, 1), (5, 0)): -0.85,
                                    ((5, 1), (1, 0)): -0.85,
                                    ((1, 1), (9, 0)): -0.85,
                                    ((9, 1), (1, 0)): -0.85,
                                    ((1, 1), (13, 0)): -0.85,
                                    ((13, 1), (1, 0)): -0.85,
                                    ((1, 1), (15, 0)): 0.85,
                                    ((15, 1), (1, 0)): 0.85, 
                                    ((2, 1), (10, 0)): -0.85,
                                    ((10, 1), (2, 0)): -0.85,
                                    ((2, 1), (6, 0)): -0.85,
                                    ((6, 1), (2, 0)): -0.85,
                                    ((2, 1), (14, 0)): -0.85,
                                    ((14, 1), (2, 0)): -0.85,
                                    ((2, 1), (12, 0)): 0.85,
                                    ((12, 1), (2, 0)): 0.85,
                                    ((3, 1), (11, 0)): -0.85,
                                    ((11, 1), (3, 0)): -0.85,
                                    ((3, 1), (7, 0)): -0.85,
                                    ((7, 1), (3, 0)): -0.85,
                                    ((3, 1), (15, 0)): -0.85,
                                    ((15, 1), (3, 0)): -0.85,
                                    ((3, 1), (13, 0)): 0.85,
                                    ((13, 1), (3, 0)): 0.85,
                                    ((4, 1), (12, 0)): -0.85,
                                    ((12, 1), (4, 0)): -0.85,
                                    ((4, 1), (8, 0)): -0.85,
                                    ((8, 1), (4, 0)): -0.85,
                                    ((4, 1), (10, 0)): -0.85,
                                    ((10, 1), (4, 0)): -0.85,
                                    ((5, 1), (13, 0)): -0.85,
                                    ((13, 1), (5, 0)): -0.85,
                                    ((5, 1), (9, 0)): -0.85,
                                    ((9, 1), (5, 0)): -0.85,
                                    ((5, 1), (11, 0)): -0.85,
                                    ((11, 1), (5, 0)): -0.85,
                                    ((6, 1), (14, 0)): -0.85,
                                    ((14, 1), (6, 0)): -0.85,
                                    ((6, 1), (10, 0)): -0.85,
                                    ((10, 1), (6, 0)): -0.85,
                                    ((6, 1), (8, 0)): -0.85,
                                    ((8, 1), (6, 0)): -0.85,
                                    ((7, 1), (15, 0)): -0.85,
                                    ((15, 1), (7, 0)): -0.85,
                                    ((7, 1), (11, 0)): -0.85,
                                    ((11, 1), (7, 0)): -0.85,
                                    ((7, 1), (9, 0)): -0.85,
                                    ((9, 1), (7, 0)): -0.85,
                                    ((8, 1), (12, 0)): -0.85,
                                    ((12, 1), (8, 0)): -0.85,
                                    ((9, 1), (13, 0)): -0.85,
                                    ((13, 1), (9, 0)): -0.85,
                                    ((10, 1), (14, 0)): -0.85,
                                    ((14, 1), (10, 0)): -0.85,
                                    ((11, 1), (15, 0)): -0.85,
                                    ((15, 1), (11, 0)): -0.85,
                                    ((0, 1), (0, 0)): -1.0,
                                    ((1, 1), (1, 0)): -1.0,
                                    ((2, 1), (2, 0)): -1.0,
                                    ((3, 1), (3, 0)): -1.0,
                                    ((4, 1), (4, 0)): -1.0,
                                    ((5, 1), (5, 0)): -1.0,
                                    ((6, 1), (6, 0)): -1.0,
                                    ((7, 1), (7, 0)): -1.0,
                                    ((8, 1), (8, 0)): -1.0,
                                    ((9, 1), (9, 0)): -1.0,
                                    ((10, 1), (10, 0)): -1.0,
                                    ((11, 1), (11, 0)): -1.0,
                                    ((12, 1), (12, 0)): -1.0,
                                    ((13, 1), (13, 0)): -1.0,
                                    ((14, 1), (14, 0)): -1.0,
                                    ((15, 1), (15, 0)): -1.0}
        for edge, label in correct_fermionic_terms.items():
            self.assertTrue(edge in fermionic_two_orbital.terms)
            self.assertEqual(label, fermionic_two_orbital.terms[edge])


if __name__ == '__main__':
    unittest.main()
