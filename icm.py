import numpy as np
import random
from numba import jit


def indicator(S, n):
    x = np.zeros(n)
    x[list(S)] = 1
    return x


def sample_live_icm(g, num_graphs):

    import networkx as nx
    live_edge_graphs = []
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u, v in g.edges():
            if random.random() < g[u][v]['p']:
                h.add_edge(u, v)
        live_edge_graphs.append(h)
    return live_edge_graphs


def f_all_influmax_multlinear(x, Gs, Ps, ws):

    n = len(Gs)
    sample_weights = 1. / n * np.ones(n)
    return objective_live_edge(x, Gs, Ps, ws, sample_weights)


def make_multilinear_objective_samples(live_graphs, target_nodes, selectable_nodes, p_attend):

    Gs, Ps, ws = live_edge_to_adjlist(live_graphs, target_nodes, p_attend)

    def f_all(x):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        return f_all_influmax_multlinear(x_expand, Gs, Ps, ws)

    return f_all


def make_multilinear_gradient_samples(live_graphs, target_nodes, selectable_nodes, p_attend):

    import random
    Gs, Ps, ws = live_edge_to_adjlist(live_graphs, target_nodes, p_attend)

    def gradient(x, batch_size):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        samples = random.sample(range(len(Gs)), batch_size)
        grad = gradient_live_edge(x_expand, [Gs[i] for i in samples], [Ps[i] for i in samples],
                                  [ws[i] for i in samples], 1. / batch_size * np.ones(len(Gs)))
        return grad[selectable_nodes]

    return gradient


def live_edge_to_adjlist(live_edge_graphs, target_nodes, p_attend):

    import networkx as nx
    Gs = []
    Ps = []
    ws = []
    target_nodes = set(target_nodes)
    for g in live_edge_graphs:
        cc = list(nx.connected_components(g))
        n = len(cc)
        max_degree = max([len(c) for c in cc])
        G_array = np.zeros((n, max_degree), dtype=np.int)
        P = np.zeros((n, max_degree))
        G_array[:] = -1
        for i in range(n):
            for j, v in enumerate(cc[i]):
                G_array[i, j] = v
                P[i, j] = p_attend[v]
        Gs.append(G_array)
        Ps.append(P)
        w = np.zeros((n))
        for i in range(n):
            w[i] = len(target_nodes.intersection(cc[i]))
        ws.append(w)
    return Gs, Ps, ws


@jit
def gradient_live_edge(x, Gs, Ps, ws, weights):

    grad = np.zeros((len(x)))
    for i in range(len(Gs)):
        grad += weights[i] * gradient_coverage(x, Gs[i], Ps[i], ws[i])
    grad /= len(x)
    return grad


@jit
def objective_live_edge(x, Gs, Ps, ws, weights):

    total = 0
    for i in range(len(Gs)):
        total += weights[i] * objective_coverage(x, Gs[i], Ps[i], ws[i])
    return total


@jit
def gradient_coverage(x, G, P, w):

    grad = np.zeros((x.shape[0]))
    # process gradient entries one node at a time
    for v in range(G.shape[0]):
        p_all_fail = 1
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            p_all_fail *= 1 - x[G[v, j]] * P[v, j]
        for j in range(G.shape[1]):
            u = G[v, j]
            if u == -1:
                break
            # 0/0 should be 0 here
            if p_all_fail == 0:
                p_others_fail = 0
            else:
                p_others_fail = p_all_fail / (1 - x[u] * P[v, j])
            grad[u] += w[v] * P[v, j] * p_others_fail
    return grad


@jit
def marginal_coverage(x, G, P, w):

    probs = np.ones((G.shape[0]))
    for v in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            u = G[v, j]
            probs[v] *= 1 - x[u] * P[v, j]
    probs = 1 - probs
    return probs


@jit
def objective_coverage(x, G, P, w):

    return np.dot(w, marginal_coverage(x, G, P, w))