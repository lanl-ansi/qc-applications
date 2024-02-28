import random
from pyLIQTR.utils import Hamiltonian
from networkx import relabel_nodes, Graph, grid_graph
from networkx.generators.lattice import grid_2d_graph
from openfermion import FermionOperator, QubitOperator

def flatten_nx_graph(graph: Graph) -> Graph:
    """
    This function takes as input a graph and flattens it

    :param graph: The graph to be flattened
    :type graph: networkx.Graph
    :return: The flattened graph
    :rtype: networkx.Graph

    """
    new_ids = {}
    count = 0
    for node in graph.nodes:
        if node not in new_ids:
            new_ids[node] = count
            count = count + 1
    new_graph = relabel_nodes(graph, new_ids)
    return new_graph


def generate_two_orbital_nx(Lx: int, Ly: int) -> Graph:
    # can combine logic between loops if this is slow
    g = Graph()
    for m in range(Lx):
        for n in range(Ly):
            for a in range(2):
                for s in range(2):
                    g.add_node((m, n, a, s), pos=(
                        m + a * (Lx + 1), n + s * (Ly + 1)))

    for m in range(Lx):
        for n in range(Ly):
            for s in range(2):
                # t_1 terms
                n1, n2 = (m, n, 0, s), (m, n + 1, 0, s)
                n3, n4 = (m, n, 1, s), (m + 1, n, 1, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t1")
                if n4 in g:
                    g.add_edge(n3, n4, label="-t1")

                # t_2 terms
                n1, n2 = (m, n, 0, s), (m + 1, n, 0, s)
                n3, n4 = (m, n, 1, s), (m, n + 1, 1, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t2")
                if n4 in g:
                    g.add_edge(n3, n4, label="-t2")

                # t_3 terms
                n1, n2 = (m, n, 0, s), (m + 1, n + 1, 0, s)
                n3, n4 = (m, n, 1, s), (m + 1, n + 1, 1, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t3")
                if n4 in g:
                    g.add_edge(n3, n4, label="-t3")

                n1, n2 = (m, n, 0, s), (m + 1, n - 1, 0, s)
                n3, n4 = (m, n, 1, s), (m + 1, n - 1, 1, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t3")
                if n4 in g:
                    g.add_edge(n3, n4, label="-t3")

                n1, n2 = (m, n, 0, s), (m + 1, n, 0, s)
                n3, n4 = (m, n, 1, s), (m + 1, n, 1, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t3")
                if n4 in g:
                    g.add_edge(n3, n4, label="-t3")

                n1, n2 = (m, n, 0, s), (m, n + 1, 0, s)
                n3, n4 = (m, n, 1, s), (m, n + 1, 1, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t3")
                if n4 in g:
                    g.add_edge(n3, n4, label="-t3")

                # +t_4 terms
                n1, n2 = (m, n, 0, s), (m + 1, n + 1, 1, s)
                n3, n4 = (m, n, 1, s), (m + 1, n + 1, 0, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="+t4")
                if n4 in g:
                    g.add_edge(n3, n4, label="+t4")

                # -t4 terms
                n1, n2 = (m, n, 0, s), (m + 1, n - 1, 1, s)
                n3, n4 = (m, n, 1, s), (m + 1, n - 1, 0, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t4")
                if n4 in g:
                    g.add_edge(n3, n4, label="-t4")
    return g


def nx_to_two_orbital_hamiltonian(
        graph: Graph,
        t1: float,
        t2: float,
        t3: float,
        t4: float,
        mu: float) -> FermionOperator:
    g_flat = flatten_nx_graph(graph)
    H = FermionOperator()

    # generating hopping terms on each edge
    for i, j, d in g_flat.edges(data=True):
        w = 0
        label = d['label']
        if label == "-t1":
            w = -t1
        elif label == "-t2":
            w = -t2
        elif label == "-t3":
            w = -t3
        elif label == "-t4":
            w = -t4
        elif label == "+t4":
            w = t4
        else:
            raise ValueError("Graph improperly labeled")

        H += FermionOperator(((i, 1), (j, 0)), w)
        H += FermionOperator(((j, 1), (i, 0)), w)

    # applying number operator to each qubit
    for i in g_flat.nodes:
        H += FermionOperator(((i, 1), (i, 0)), -mu)

    return H

# given a networkx graph g, construct a hamiltonian with random weights
# which can be processed by pyLIQTR
def nx_longitudinal_ising_terms(graph,p,magnitude=1) -> list[tuple[str, float]]:
    H_longitudinal = []
    n = len(graph.nodes)
    for n1, n2 in graph.edges:
        weight = magnitude if random.random() < p else -magnitude
        curr_hamil_string = n * 'I' 
        for idx in range(len(graph)):
            if idx == n1 or idx == n2:
                curr_hamil_string = f'{curr_hamil_string[:idx]}Z{curr_hamil_string[idx+1:]}'
        H_longitudinal.append((curr_hamil_string, weight))
    return H_longitudinal

def nx_transverse_ising_terms(graph: Graph,p,magnitude=0.1) -> list[tuple[str, float]]:
    H_transverse = []
    n = len(graph)
    for idx in range(n):
        w = magnitude if random.random() < p else -magnitude
        curr_hamil_string = n * 'I' 
        for k in range(n):
            if idx == k:
                curr_hamil_string = f'{curr_hamil_string[:idx]}X{curr_hamil_string[idx+1:]}'
        H_transverse.append((curr_hamil_string, w))
    return H_transverse


def nx_triangle_lattice(lattice_size: int) -> Graph:
    graph = grid_2d_graph(lattice_size, lattice_size)
    for i in range(lattice_size - 1):
        for j in range(lattice_size - 1):
            graph.add_edge((i,j),(i+1,j+1))
    return graph

def generate_triangle_hamiltonian(lattice_size: int, longitudinal_weight_prob:float=0.5, transverse_weight_prob:float=1):
    graph = nx_triangle_lattice(lattice_size)
    graph = flatten_nx_graph(graph)
    H_transverse = nx_transverse_ising_terms(graph, transverse_weight_prob)
    H_longitudinal = nx_longitudinal_ising_terms(graph, longitudinal_weight_prob)
    return H_transverse, H_longitudinal

def generate_square_hamiltonian(lattice_size: int, dim:int, longitudinal_weight_prob:float=0.5, transverse_weight_prob:float=1):
    dimensions = (lattice_size, lattice_size) if dim == 2 else (lattice_size, lattice_size, lattice_size)
    graph = grid_graph(dim=dimensions)
    graph = flatten_nx_graph(graph)
    H_transverse = nx_transverse_ising_terms(graph, transverse_weight_prob)
    H_longitudinal = nx_longitudinal_ising_terms(graph, longitudinal_weight_prob)
    return H_transverse, H_longitudinal

def pyliqtr_hamiltonian_to_openfermion_qubit_operator(H:Hamiltonian) -> QubitOperator:
    open_fermion_operator = QubitOperator()
    for term in H.terms:
        open_fermion_term = ''
        for i, pauli in enumerate(term[0]):
            if pauli != 'I':
                open_fermion_term = f'{open_fermion_term}{pauli}{i} '
        open_fermion_term_op = QubitOperator(open_fermion_term)
        if open_fermion_term:
            open_fermion_operator += term[1] * open_fermion_term_op
    return open_fermion_operator

def assign_hexagon_labels(graph:Graph):
    for n1, n2 in graph.edges:
        # start by making sure that the edges are ordered correctly
        r1,c1 = n1
        r2,c2 = n2
        if r2 - r1 < 0 or c2 - c1 < 0:
            r1, r2 = r2, r1
            c1, c2 = c2, c1
        
        # now that they are ordered correctly, we can assign labels
        label = ''
        if c1 == c2:
            label = 'Z'
        # You can differentiate X and Y labels based off nx's node label parity
        elif (((r1 % 2) + (c1 % 2)) % 2 == 0):
            label = 'Y'
        else:
            label = 'X'
        
        graph[n1][n2]['label'] = label

def assign_directional_triangular_labels(g:Graph, lattice_size:int) -> None: 
    for i in range(lattice_size - 1):
        for j in range(lattice_size - 1):
            g[(i,j)][(i+1,j)]['label'] = 'Z'
            g[(i,j)][(i,j+1)]['label'] = 'X'
            g[(i,j)][i+1,j+1]['label'] = 'Y'
        g[(i,lattice_size-1)][(i+1,lattice_size-1)]['label'] = 'Z'
    for j in range(lattice_size - 1):
        g[(lattice_size-1,j)][(lattice_size-1,j+1)]['label'] = 'X'
