# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:24:02 2025

@author: Vollmar
"""
import numpy as np
from scipy.linalg import eig
import networkx as nx
import matplotlib.pyplot as plt

def compute_statPiCalc(wA):
    """
    Compute the stationary distribution (statPi) from a transition matrix wA.

    Parameters:
        wA (np.ndarray): Transition matrix

    Returns:
        statPi (np.ndarray): Stationary state population probabilities
    """
    # Compute left and right eigenvectors
    EigenValues, leftEigenVectors = eig(wA, left=True, right=False)

    # Find the index of the eigenvalue closest to 1 (within numerical tolerance)
    idxs = np.where(np.isclose(EigenValues, 1, atol=1e-8))[0]
    if len(idxs) == 0:
        print("No eigenvalue == 1 found")
        return None, None
    idx = idxs[0]

    # Extract the left eigenvector corresponding to eigenvalue 1
    statPi = np.real(leftEigenVectors[:, idx])
    statPi = statPi / np.sum(statPi)

    return statPi
# %% Entropy production
def compute_entropy_production(Transitionmatrix, P):
    """
    Compute average entropy production rate ⟨Ṡ⟩ for a Markov system.
    
    Parameters:
        Transitionmatrix (ndarray): NxN array of transition probabilities (k_ij)
        P (ndarray): 1D array of state populations (P_i), shape (N,)
    
    Returns:
        float: Average entropy production rate ⟨Ṡ⟩
    """
    N = Transitionmatrix.shape[0]
    S_dot = 0.0

    for i in range(N):
        for j in range(N):
            if i != j:
                k_ij = Transitionmatrix[i, j]
                k_ji = Transitionmatrix[j, i]
                Pi = P[i]
                Pj = P[j]

                num = k_ij * Pi
                den = k_ji * Pj

                # Skip if either term is zero (avoids log(0))
                if num > 0 and den > 0:
                    S_dot += (num - den) * np.log(num / den)

    return 0.5 * S_dot

# %% Thermodynamic force by Hill = affinity = DeltaG
def compute_cycle_affinity(Transitionmatrix, cycle):
    """
    Compute the thermodynamic force (cycle affinity) ΔG for a given cycle.

    Parameters:
        Transitionmatrix (ndarray): NxN array of transition rates (k_ij)
        cycle (list of int): ordered list of states in the cycle (e.g., [0, 1, 2])

    Returns:
        float: cycle affinity ΔG (in units of k_B*T if dimensionless)
    """
    forward_product = 1.0
    reverse_product = 1.0
    n = len(cycle)

    for i in range(n):
        from_state = cycle[i]
        to_state = cycle[(i + 1) % n]  # wrap around to start
        forward_product *= Transitionmatrix[from_state, to_state]
        reverse_product *= Transitionmatrix[to_state, from_state]

    if forward_product > 0 and reverse_product > 0:
        delta_G = np.log(forward_product / reverse_product)
    else:
        delta_G = np.nan  # undefined if any rate is zero

    return delta_G
# %% draw HMM
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_HMM_graph(Transitionmatrix, steadyStatePopulations, threshold=1e-3):
    """
    Draws a directed graph of an HMM using transition matrix and steady state populations.
    
    Parameters:
        Transitionmatrix (ndarray): NxN transition matrix (k_ij)
        steadyStatePopulations (ndarray): 1D array of length N (P_i)
        threshold (float): minimum transition probability to show an edge
    """
    N = len(steadyStatePopulations)
    G = nx.DiGraph()

    # Add nodes with steady state population as node size or label
    for i in range(N):
        label = f"S{i}\nπ={steadyStatePopulations[i]:.2f}"
        G.add_node(i, label=label)

    # Add edges for nonzero transitions above a threshold
    for i in range(N):
        for j in range(N):
            weight = Transitionmatrix[i, j]
            if i != j and weight > threshold:
                G.add_edge(i, j, weight=weight)

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # or use circular_layout for a ring-like cycle
    edge_labels = {(i, j): f"$p_{{{i}{j}}}$={d['weight']:.2f}" for i, j, d in G.edges(data=True)}

    node_labels = nx.get_node_attributes(G, 'label')  #  retrieves the "label" attribute from all nodes in the graph G
    node_sizes = 3000 * steadyStatePopulations  # scale for visibility

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20,
                           connectionstyle='arc3,rad=0.1',
                           min_source_margin=20, min_target_margin=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                 label_pos=0.3,verticalalignment='center')# verticalalignment='top'

    plt.title("HMM Transition Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

T = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.3, 0.4]
])
pi = np.array([0.4, 0.35, 0.25])

draw_HMM_graph(T, pi)
# %% version 2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_HMM_graph_with_curved_edges(Transitionmatrix, steadyStatePopulations, threshold=1e-3):
    N = len(steadyStatePopulations)
    G = nx.DiGraph()

    # Add nodes
    for i in range(N):
        label = f"S{i}\nπ={steadyStatePopulations[i]:.2f}"
        G.add_node(i, label=label)

    # Add edges
    for i in range(N):
        for j in range(N):
            if i != j and Transitionmatrix[i, j] > threshold:
                G.add_edge(i, j, weight=Transitionmatrix[i, j])

    pos = nx.spring_layout(G, seed=3)
    #pos = nx.circular_layout(G) 
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = {(i, j): f"$p_{{{i}{j}}}$={d['weight']:.2f}" for i, j, d in G.edges(data=True)}
    node_sizes = 3000 * steadyStatePopulations

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    drawn_edges = set()

    # Draw edges and individual edge labels
    for i, j in G.edges():
        if (j, i) in G.edges() and (j, i) not in drawn_edges:
            # Draw both directions with same curvature (rad=0.2)
            # Forward edge i -> j
            nx.draw_networkx_edges(
                G, pos, edgelist=[(i, j)],
                connectionstyle=f'arc3,rad=0.1',
                arrowstyle='->', arrowsize=20,
                min_source_margin=20, min_target_margin=20
            )
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(i, j): f"$p_{{{i}{j}}}$={Transitionmatrix[i, j]:.2f}"},
                label_pos=0.4, font_size=8,
                rotate=True,
                verticalalignment='top',
                bbox=None
            )

            # Backward edge j -> i
            nx.draw_networkx_edges(
                G, pos, edgelist=[(j, i)],
                connectionstyle=f'arc3,rad=0.1',
                arrowstyle='->', arrowsize=20,
                edge_color='blue',
                min_source_margin=20, min_target_margin=20
            )
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(j, i): f"$p_{{{j}{i}}}$={Transitionmatrix[j, i]:.2f}"},
                label_pos=0.4, font_size=8,
                rotate=True,
                verticalalignment='bottom',
                bbox=None,
                font_color='g'
            )

            drawn_edges.add((i, j))
            drawn_edges.add((j, i))
        elif (i, j) not in drawn_edges:
            # Single direction
            nx.draw_networkx_edges(
                G, pos, edgelist=[(i, j)],
                connectionstyle='arc3,rad=0.0',
                arrowstyle='->', arrowsize=20
            )
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(i, j): f"{Transitionmatrix[i, j]:.2f}"},
                label_pos=0.5, font_size=8
            )
            drawn_edges.add((i, j))



    plt.title("HMM with Curved Bidirectional Edges")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
T = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.3, 0.4]
])
pi = np.array([0.4, 0.35, 0.25])

draw_HMM_graph_with_curved_edges(T, pi)