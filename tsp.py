from sys import maxsize 
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

V = 10

# implementation of traveling Salesman Problem 
def travellingSalesmanProblem(graph, s): 

    # store all vertex apart from source vertex 
    vertex = [] 
    visited = set()
    for i in range(V): 
        if i != s: 
            vertex.append(i) 
  
    # store minimum weight Hamiltonian Cycle 
    min_path = maxsize 
  
    while True: 
  
        # store current Path weight(cost) 
        current_pathweight = 0
  
        # compute current path weight 
        k = s 
        for i in range(len(vertex)): 
            current_pathweight += graph[k][vertex[i]] 
            k = vertex[i] 
        
        if k not in visited:
            path.append(k)
        visited.add(k)

        current_pathweight += graph[k][s] 
        # update minimum 
       
        min_path = min(min_path, current_pathweight) 
        if not next_permutation(vertex): 
            break
  
    return min_path 
# next_permutation implementation 
def next_permutation(L): 
  
    n = len(L) 
  
    i = n - 2
    while i >= 0 and L[i] >= L[i + 1]: 
        i -= 1
  
    if i == -1: 
        return False
  
    j = i + 1
    while j < n and L[j] > L[i]: 
        j += 1
    j -= 1
  
    L[i], L[j] = L[j], L[i] 
  
    left = i + 1
    right = n - 1
  
    while left < right: 
        L[left], L[right] = L[right], L[left] 
        left += 1
        right -= 1
  
    return True

def plot():
    A = [[ 0., 18., 13.,  5.,  8.,  4.,  7., 11.,  2.,  6.],
       [18.,  0.,  7., 23., 16., 22., 17., 19., 20., 12.],
       [13.,  7.,  0., 18., 11., 17., 12., 14., 15.,  7.],
       [ 5., 23., 18.,  0.,  7.,  1., 12.,  8.,  7., 11.],
       [ 8., 16., 11.,  7.,  0.,  6.,  9.,  3., 10.,  4.],
       [ 4., 22., 17.,  1.,  6.,  0., 11.,  7.,  6., 10.],
       [ 7., 17., 12., 12.,  9., 11.,  0., 12.,  5.,  5.],
       [11., 19., 14.,  8.,  3.,  7., 12.,  0., 13.,  7.],
       [ 2., 20., 15.,  7., 10.,  6.,  5., 13.,  0.,  8.],
       [ 6., 12.,  7., 11.,  4., 10.,  5.,  7.,  8.,  0.]]

    G = nx.from_numpy_matrix(np.array(A)) 
    edge_labels = dict( ((u, v), d["weight"]) for u, v, d in G.edges(data=True) )
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Driver Code 
if __name__ == "__main__": 
    path = []
    # matrix representation of graph 

    plot()

    graph = [[ 0., 18., 13.,  5.,  8.,  4.,  7., 11.,  2.,  6.],
       [18.,  0.,  7., 23., 16., 22., 17., 19., 20., 12.],
       [13.,  7.,  0., 18., 11., 17., 12., 14., 15.,  7.],
       [ 5., 23., 18.,  0.,  7.,  1., 12.,  8.,  7., 11.],
       [ 8., 16., 11.,  7.,  0.,  6.,  9.,  3., 10.,  4.],
       [ 4., 22., 17.,  1.,  6.,  0., 11.,  7.,  6., 10.],
       [ 7., 17., 12., 12.,  9., 11.,  0., 12.,  5.,  5.],
       [11., 19., 14.,  8.,  3.,  7., 12.,  0., 13.,  7.],
       [ 2., 20., 15.,  7., 10.,  6.,  5., 13.,  0.,  8.],
       [ 6., 12.,  7., 11.,  4., 10.,  5.,  7.,  8.,  0.]]
    s = 0
    print(travellingSalesmanProblem(graph, s)) 
    path.reverse()
    print(path)
    