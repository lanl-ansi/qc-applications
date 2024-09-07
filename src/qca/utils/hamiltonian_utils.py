import random
from pyLIQTR.utils import Hamiltonian
from networkx import (relabel_nodes, Graph, MultiGraph, grid_graph,
                      compose, path_graph, set_node_attributes, get_node_attributes)
from networkx.generators.lattice import grid_2d_graph
from openfermion import FermionOperator, QubitOperator

def flatten_nx_graph(graph: Graph) -> Graph:
    new_ids = {}
    count = 0
    for node in graph.nodes:
        if node not in new_ids:
            new_ids[node] = count
            count = count + 1
    new_graph = relabel_nodes(graph, new_ids)
    return new_graph


def generate_two_orbital_nx(Lx: int, Ly: int) -> MultiGraph:
    # can combine logic between loops if this is slow
    g = MultiGraph()
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
        graph: MultiGraph,
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

def generate_xz_yz_hamiltonian_nx(Lx: int, Ly: int) -> Graph:
    #xz is given orbital label 0, yz is given orbital label 1
    g = MultiGraph()
    for m in range(Lx):
        for n in range(Ly):
            for a in range(2):
                for s in range(2):
                    g.add_node((m, n, a, s), pos=(
                        m + a * (Lx + 1), n + s * (Ly + 1)),
                              delta=False)

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

def generate_xy_hamiltonian_nx(Lx: int, Ly: int) -> Graph:
    #xy is given orbital label 2
    g = MultiGraph()
    for m in range(Lx):
        for n in range(Ly):
            for a in [2]:
                for s in range(2):
                    #Adding nodes to graph to make edge checking easier
                    #Also adding node positions for drawing functionality
                    g.add_node((m, n, a, s), pos=(
                        m + a * (Lx + 1), n + s * (Ly + 1)),
                              delta=True)

    for m in range(Lx):
        for n in range(Ly):
            for s in range(2):
                #t_5 terms
                n1, n2 = (m, n, 2, s), (m, n + 1, 2, s)
                n3, n4 = (m, n, 2, s), (m + 1, n, 2, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="+t5")
                if n4 in g:
                    g.add_edge(n3, n4, label="+t5")

                #t_6 terms
                n1, n2 = (m, n, 2, s), (m + 1, n + 1, 2, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t6")

                n1, n2 = (m, n, 2, s), (m + 1, n - 1, 2, s)
                if n2 in g:
                    g.add_edge(n1, n2, label="-t6")
    return g

def generate_xz_yz_xy_hamiltonian_nx(Lx: int, Ly: int) -> Graph:
    #xz is orbital label 0
    #yz is orbital label 1
    #xy is orbital label 2
    g = MultiGraph()
    for m in range(Lx):
        for n in range(Ly):
            for a in range(3):
                for s in range(2):
                    #Adding nodes to graph to make edge checking easier
                    #Also adding node positions for drawing functionality
                    g.add_node((m, n, a, s), pos=(
                        m + a * (Lx + 1), n + s * (Ly + 1)),
                              delta=(a==2))

    for m in range(Lx):
        for n in range(Ly):
            for s in range(2):
                #t_7 terms
                n1, n2 = (m, n, 0, s), (m + 1, n, 2, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t7")
                    else:
                        g.add_edge(n1, n2, label="+t7")

                n1, n2 = (m, n, 2, s), (m + 1, n, 0, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t7")
                    else:
                        g.add_edge(n1, n2, label="+t7")

                n1, n2 = (m, n, 1, s), (m, n + 1, 2, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t7")
                    else:
                        g.add_edge(n1, n2, label="+t7")

                n1, n2 = (m, n, 2, s), (m, n + 1, 1, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t7")
                    else:
                        g.add_edge(n1, n2, label="+t7")

                #t_8 terms
                n1, n2 = (m, n, 0, s), (m + 1, n + 1, 2, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t8")
                    else:
                        g.add_edge(n1, n2, label="+t8")

                n1, n2 = (m, n, 2, s), (m + 1, n + 1, 0, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="+t8")
                    else:
                        g.add_edge(n1, n2, label="-t8")

                n1, n2 = (m, n, 0, s), (m + 1, n - 1, 2, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t8")
                    else:
                        g.add_edge(n1, n2, label="+t8")

                n1, n2 = (m, n, 2, s), (m + 1, n - 1, 0, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="+t8")
                    else:
                        g.add_edge(n1, n2, label="-t8")

                n1, n2 = (m, n, 1, s), (m + 1, n + 1, 2, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t8")
                    else:
                        g.add_edge(n1, n2, label="+t8")

                n1, n2 = (m, n, 2, s), (m + 1, n + 1, 1, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="+t8")
                    else:
                        g.add_edge(n1, n2, label="-t8")

                n1, n2 = (m, n, 1, s), (m + 1, n - 1, 2, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="+t8")
                    else:
                        g.add_edge(n1, n2, label="-t8")

                n1, n2 = (m, n, 2, s), (m + 1, n - 1, 1, s)
                if n2 in g:
                    if (m + n) % 2 == 0:
                        g.add_edge(n1, n2, label="-t8")
                    else:
                        g.add_edge(n1, n2, label="+t8")
    return g

def generate_three_orbital_nx(Lx: int, Ly: int) -> MultiGraph:
    # can combine logic between loops if this is slow
    g_xz_yz = generate_xz_yz_hamiltonian_nx(Lx, Ly)
    g_xy = generate_xy_hamiltonian_nx(Lx, Ly)
    g_xz_yz_xy = generate_xz_yz_xy_hamiltonian_nx(Lx, Ly)

    g_three_band = compose(g_xz_yz, g_xy)
    g_three_band = compose(g_three_band, g_xz_yz_xy)
    return g_three_band

def nx_to_three_orbital_hamiltonian(
        graph: MultiGraph,
        t1: float,
        t2: float,
        t3: float,
        t4: float,
        t5: float,
        t6: float,
        t7: float,
        t8: float,
        mu: float,
        delta: float) -> FermionOperator:

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
        elif label == "+t5":
            w = t5
        elif label == "-t6":
            w = -t6
        elif label == "+t7":
            w = t7
        elif label == "-t7":
            w = -t7
        elif label == "+t8":
            w = t8
        elif label == "-t8":
            w = -t8
        else:
            raise ValueError("Graph improperly labeled")

        H += FermionOperator(((i, 1), (j, 0)), w)
        H += FermionOperator(((j, 1), (i, 0)), w)

    # applying number operator to each qubit
    for i,d in g_flat.nodes(data=True):
        H += FermionOperator(((i, 1), (i, 0)), -mu)
        if d['delta']:
            H += FermionOperator(((i, 1), (i, 0)), delta)

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

def generate_triangle_hamiltonian(
        lattice_size: int,
        longitudinal_weight_prob:float=0.5,
        transverse_weight_prob:float=1
    ):
    graph = nx_triangle_lattice(lattice_size)
    graph = flatten_nx_graph(graph)
    H_transverse = nx_transverse_ising_terms(graph, transverse_weight_prob)
    H_longitudinal = nx_longitudinal_ising_terms(graph, longitudinal_weight_prob)
    return H_transverse, H_longitudinal

def generate_square_hamiltonian(
        lattice_size: int,
        dim:int,
        longitudinal_weight_prob:float=0.5,
        transverse_weight_prob:float=1
    ):
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

def assign_hexagon_labels(graph:Graph, x:str='X', y:str='Y', z:str='Z'):
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
            label = z
        # You can differentiate X and Y labels based off nx's node label parity
        elif ((r1 % 2) + (c1 % 2)) % 2 == 0:
            label = y
        else:
            label = x

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

def dicke_model_qubit_hamiltonian(n_s:int=10, 
                                  n_b:int=5,
                                  omega_c:float=1.3,
                                  omega_o:float=1, #offset that won't affect ground state
                                  lam:float=1.5,
                                  h_bar:float=1):
    #H = h_bar * omega_c * a'a + (h_bar * \omega_o)/2 * \sum_{n = 1}^{n_spins} (\sigma^z_n + I) + h_bar \lambda
    creation = bosonic_creation_operator(n_b)
    annihilation = bosonic_annihilation_operator(n_b)
    number = bosonic_number_operator(n_b)

    H = h_bar * omega_c * number
    for spin_site in range(n_b + 1, n_b + n_s+1):
        H += (h_bar * omega_o / 2) * (QubitOperator(f'Z{spin_site}') + 1)
        H += (h_bar * lam * (annihilation + creation)) * (sigma_minus(spin_site) + sigma_plus(spin_site))
    return H

def tavis_cummings_model_qubit_hamiltonian(n_s:int=10, 
                                  n_b:int=5,
                                  omega_c:float=1.3,
                                  omega_o:float=1, #offset that won't affect ground state
                                  lam:float=1.5,
                                  h_bar:float=1):
    #H = h_bar * omega_c * a'a + (h_bar * \omega_o)/2 * \sum_{n = 1}^{n_spins} (\sigma^z_n + I) + h_bar \lambda
    creation = bosonic_creation_operator(n_b)
    annihilation = bosonic_annihilation_operator(n_b)
    number = bosonic_number_operator(n_b)

    H = h_bar * omega_c * number
    for spin_site in range(n_b + 1, n_b + n_s+1):
        H += (h_bar * omega_o / 2) * (QubitOperator(f'Z{spin_site}') + 1)
        H += h_bar * lam * (creation * sigma_minus(spin_site) + annihilation * sigma_plus(spin_site))
    return H

def sigma_plus(n):
    return (QubitOperator(f'X{n}') + 1j * QubitOperator(f'Y{n}')) / 2

def sigma_minus(n):
    return (QubitOperator(f'X{n}') - 1j * QubitOperator(f'Y{n}')) / 2

def bosonic_creation_operator(n_b):
    H = QubitOperator() #we have states 0 -> n_b (n_b+1 states)
    for i in range(n_b):
        H += pow(i+1, 0.5) * sigma_plus(i+1) * sigma_minus(i)
    return H

def bosonic_annihilation_operator(n_b):
    H = QubitOperator()
    for i in range(n_b): #we have states 0 -> n_b (n_b+1 states)
        H += pow(i+1, 0.5) * sigma_minus(i+1) * sigma_plus(i)
    return H

def bosonic_number_operator(n_b):
    H = QubitOperator()
    for i in range(n_b+1): #accounting for this being a line rather than a ring
        H += (i) * 0.5 * (QubitOperator(f'Z{i}') + 1)
    return H

def generate_dicke_model_nx(n_s:int=10, n_b:int=5):
    g = path_graph(n_b + 1) #levels 0 -> n_b
    boson_labels = {b: 'b' for b in range(n_b+1)}
    boson_positions = {b: (0, b) for b in range(n_b+1)}
    set_node_attributes(g, boson_labels, 'label')
    set_node_attributes(g, boson_positions, 'pos')
    #adding spins
    for spin_node in range(n_b+1, n_s+n_b+1):
        g.add_node(spin_node, label='s', pos=(1, spin_node - n_b))
        for boson_node in range(n_b+1):
            g.add_edge(boson_node, spin_node)
    return g
