import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        # Checks: adjacency matrix is in the expected format
        
        # 1. Check it is non-zero 
        if self.adj_mat.shape[0] == 0 | self.adj_mat == None: 
            raise ValueError("Problem with supplied adjacency matrix: adj_mat is empty") 

        # 2. Check adjacency matrix is square
        if self.adj_mat.shape[0] != self.adj_mat.shape[1]:
            raise ValueError("Problem with supplied adjacency matrix: adj_mat is not square") 
        
        # 3. Check adjacency matrix is symmetric 
        if np.array_equal(self.adj_mat, self.adj_mat.T): 
            raise ValueError("Problem with supplied adjacency matrix: adj_mat is not symmetric")
        
        
        # Initialization 
        #self.mst = None

        # Initialize S  with an arbitrary node (node 0) 
        S = [0]
        
        # number of nodes 
        n_nodes = self.adj_mat.shape[0]
        
        # In trivial case of a single node, mst is a matrix of one '0'
        if n_nodes == 1:
            self.mst = np.zeros((1,1))
            return self.mst
        
        # Initialize mst as a square matrix of zeros
        mst = np.zeros((n_nodes, n_nodes))
        
        # Keep track of visited nodes 
        visited = []
        visited.add(S)
        
        # Edges queue
        edges = []
        heapq.heapify(edges)

        # Add all edges from current node (S) to the priority queue
        for v in range(n_nodes):
            
            # if a node is connected to this node: 
            if self.adj_mat[S, v] > 0 and v not in visited:
                # adding to queue: edge weight of S -> v, S node id, v node id
                heapq.heappush(edges, (self.adj_mat[S, v], S, v))
       
        while len(visited) < n_nodes:
          
            # pick the minimum edge weight:
            edge_weight, current_node, to_add_node = heapq.heappop(edges)
            
            # unless picking this minimum edge weight would
            # create a cycle
            if to_add_node not in visited:
                # Add node and edge to MST
                # Mark visited 
                visited.add(to_add_node)
                S = [to_add_node]
                mst[current_node, to_add_node] = edge_weight
                mst[to_add_node, current_node] = edge_weight 
                
                # Add new edges from new current node to the edge weight priority queue
                # repeating the above steps
                for v in range(n_nodes):
                
                    if self.adj_mat[to_add_node, v] > 0 and v not in visited:
                       heapq.heappush(edges, (self.adj_mat[to_add_node, v], to_add_node, v))
              

        self.mst = mst
