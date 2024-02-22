from openfermion import FermionOperator
from networkx import relabel_nodes, Graph


def flatten_nx_graph(graph: Graph) -> Graph:
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
