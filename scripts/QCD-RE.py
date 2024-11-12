from argparse import ArgumentParser

import numpy as np

import networkx as nx

from qca.utils.algo_utils import estimate_trotter
from qca.utils.hamiltonian_utils import flatten_nx_graph, pyliqtr_hamiltonian_to_openfermion_qubit_operator

from pyLIQTR.utils.Hamiltonian import Hamiltonian as pyH

def parse_args():
	parser = ArgumentParser(prog='QCD Resource Estimate Generator')
	parser.add_argument('-N', '--n_neutrinos', type=int, help='Number of neutrinos in the forward scattering model')
	parser.add_argument('-T', '--trotter_steps', type=int, default=None, help='Number of trotter steps')
	parser.add_argument('-d', '--directory', type=str, default='./', help='output file directory')
	parser.add_argument('-S', '--site_inter', type=float, default=0.0, help='site interaction terms')
	parser.add_argument('-P', '--energy_precision', type=float, required=True, help='acceptable shift in state energy')
	return parser.parse_args()

def generate_spherical_momentum() -> list[float]:
	rng = np.random.default_rng()
	x = rng.normal(0, 1)
	y = rng.normal(0, 1)
	z = rng.normal(0, 1)
	constant = 1/(np.sqrt(x**2 + y**2 + z**2))
	ith_momentum = [
		constant*x,
		constant*y,
		constant*z
	]
	return ith_momentum

def define_forward_scattering_term(n_neutrinos, curr_momentum, neighbor_momentum) -> float:
	normalization_factor = 1/(np.sqrt(2)*n_neutrinos)
	couplings = 1 - np.inner(curr_momentum, neighbor_momentum)
	normalized_couplings = normalization_factor*couplings
	return normalized_couplings

def gen_couplings(n_neutrinos: int, curr_id: int, neighbor_id: int, momentum: dict) -> float:
	curr_momentum = momentum[curr_id]
	neighbor_momentum = momentum[neighbor_id]
	return define_forward_scattering_term(
		n_neutrinos=n_neutrinos,
		curr_momentum=curr_momentum,
		neighbor_momentum=neighbor_momentum
	)

def nx_heisenberg_terms(g:nx.Graph) -> list:
	hamiltonian = []
	n = len(g.nodes)
	for (n1, n2, d) in g.edges(data=True):
		weight = d['weight']
		pauli_string = n * 'I'
		for pauli in ['X', 'Y', 'Z']:
			for i in range(len(g)):
				if i == n1 or i == n2:
					pauli_string = f'{pauli_string[:i]}{pauli}{pauli_string[i+1:]}'
				hamiltonian.append((pauli_string, weight))
		
	return hamiltonian

def generate_heisenberg_graph(n_neutrinos: int, site_interaction:float=0) -> nx.Graph:
	graph = nx.Graph()
	momentum = {}
	seen = {}
	node_id = 0
	for _ in range(n_neutrinos):
		graph.add_node(node_id, weight=site_interaction)
		momentum[node_id] = generate_spherical_momentum()
		seen[node_id] = []
		node_id += 1

	for node in graph.nodes:
		curr_id = node
		for neighbor in graph.nodes:
			neighbor_id = neighbor
			if neighbor != node and curr_id not in seen[neighbor_id] and neighbor_id not in seen[curr_id]:
				coupling_terms = gen_couplings(
					n_neutrinos = n_neutrinos,
					curr_id = curr_id,
					neighbor_id = neighbor_id,
					momentum=momentum
				)
				graph.add_edge(node, neighbor, weight=coupling_terms)
				seen[curr_id].append(neighbor_id)
				seen[neighbor_id].append(curr_id)
	return graph

def generate_forward_scattering(n_neutrinos: int, site_interactions:float=0):
	graph = generate_heisenberg_graph(
		n_neutrinos=n_neutrinos,
			site_interaction=site_interactions
		)
	flat_graph = flatten_nx_graph(graph)
	scattering_hamiltonian = nx_heisenberg_terms(flat_graph)
	return scattering_hamiltonian

	
def main():
	args = parse_args()
	n_neutrinos = args.n_neutrinos
	num_steps = args.trotter_steps
	site_interactions = args.site_inter
	hamiltonian = generate_forward_scattering(n_neutrinos, site_interactions)
	
	evolution_time = np.sqrt(n_neutrinos)
	h_neutrino_pyliqtr = pyH(hamiltonian)
	qb_op_hamiltonian = pyliqtr_hamiltonian_to_openfermion_qubit_operator(h_neutrino_pyliqtr)

	fname = f'{num_steps}_step_fs_{n_neutrinos}' if num_steps else f'estimated_fs_{n_neutrinos}'
	outdir = args.directory
	energy_precision = args.energy_precision
	estimate_trotter(
		qb_op_hamiltonian,
		evolution_time,
		energy_precision,
		outdir,
		hamiltonian_name=fname,
		nsteps=num_steps
	)

if __name__ == '__main__':
	main()	