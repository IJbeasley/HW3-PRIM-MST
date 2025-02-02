import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    
    # Check there is the correct number of edges
    # expected for a mst
    # i.e. v - 1
    
    v = adj_mat.shape[0]
    observed_edges = np.count_nonzero(mst) / 2 
    expected_edges = v - 1
    
    assert expected_edges == observed_edges, "Proposed MST has incorrected number of edges"
    
    
    # Check number of nodes in mst = number of nodes in adjacency matrix
    assert adj_mat.shape[0] ==  mst.shape[0],  "Proposed MST has incorrected number of nodes"
    
    # ? Maybe add later: Check mst is connected or acyclic 


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """

    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_notsquare_mat_mst():
    """

    Unit test to check that the mst function correctly fails when a non-square matrix is loaded

    """
    # non-symmetric matrix is made from deleting
    # one row from small.csv

    test_mat = "./data/small_nonsquare.csv"
    g = Graph(test_mat)

    try:
      g.construct_mst()
      assert False, "construct_mst should have raised a value error when given a non-square matrix"

    except ValueError as e:
      assert str(e) == "Problem with supplied adjacency matrix: adj_mat is not square",  "construct_mst should have raised a different value error when given a non-square matrix"



def test_nonsymmetric_mat_mst():
    """

    Unit test to check that the mst function correctly fails when non-symmetric matrix is loaded

    """
    # non-symmetric matrix is made from editing
    # the small.csv and modifying the value of two entries
    # so is square, but not symmetric

    test_mat = "./data/small_nonsymmetric.csv"
    g = Graph(test_mat)

    try:
      g.construct_mst()
      assert False, "construct_mst should have raised a value error when given a non-symmetric square matrix"

    except ValueError as e:
      assert str(e) == "Problem with supplied adjacency matrix: adj_mat is not symmetric",  "construct_mst should have raised a different value error when given a non-symmetric square matrix"



def test_unconnected_mat_mst():
    """

    Unit test to check that the mst function correctly fails when a non-connected network is loaded

    """
    # non-symmetric matrix is made from deleting
    # one row from small.csv

    test_mat = "./data/small_nonconnected.csv"
    g = Graph(test_mat)

    try:
      g.construct_mst()
      assert False, "construct_mst should have raised a value error when given a non-connected matrix"

    except ValueError as e:
      assert str(e) == "Graph is not connected; MST cannot be formed.",  "construct_mst should have raised a different value error when given a non-connected matrix"

