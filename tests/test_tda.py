import pytest
import numpy as np
from tda import VietorisRips, PersistentHomology

def test_vietoris_rips_simple():
    # 3 points forming an equilateral triangle of side length 1.0
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2]
    ])
    
    vr = VietorisRips(max_dim=2, max_epsilon=1.5)
    filtration = vr.fit_transform(X)
    
    # Vertices (3), edges (3), triangle (1) = 7 simplices
    assert len(filtration) == 7
    
    # Check that all vertices have birth time 0
    vertices = [s for s, t in filtration if len(s) == 1]
    assert len(vertices) == 3
    for s, t in filtration:
        if len(s) == 1:
            assert t == 0.0
            
    # Check that the triangle has birth time ~1.0
    triangle = [s for s, t in filtration if len(s) == 3]
    assert len(triangle) == 1
    t_simplex, t_birth = triangle[0]
    assert t_simplex == (0, 1, 2)
    assert pytest.approx(t_birth) == 1.0


def test_homology_edge():
    # 2 points separated by distance 2.0
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0]
    ])
    
    vr = VietorisRips(max_dim=1, max_epsilon=3.0)
    filtration = vr.fit_transform(X)
    
    ph = PersistentHomology()
    intervals = ph.fit_transform(filtration)
    
    # H0: Should have:
    # - one component born at 0.0 that never dies (lives to inf)
    # - one component born at 0.0 that dies when the edge is added at 2.0
    h0 = intervals.get(0, [])
    assert len(h0) == 2
    assert h0[0] == (0.0, 2.0)
    assert h0[1] == (0.0, float('inf'))
    
    # H1: Should be empty
    h1 = intervals.get(1, [])
    assert len(h1) == 0


def test_homology_triangle():
    # 3 points forming a triangle:
    # d(0,1) = 1.0, d(1,2) = 1.5, d(0,2) = 2.0
    # Point coordinates to achieve these distances:
    # 0 is at (0,0)
    # 1 is at (1.0, 0)
    # 2 is at (x, y) such that x^2 + y^2 = 4 and (x-1)^2 + y^2 = 2.25
    # x^2 + y^2 - (x^2 - 2x + 1 + y^2) = 4 - 2.25
    # 2x - 1 = 1.75 => 2x = 2.75 => x = 1.375
    # y = sqrt(4 - 1.375^2) = sqrt(4 - 1.890625) = sqrt(2.109375) ~ 1.452
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.375, np.sqrt(2.109375)]
    ])
    
    # Max epsilon = 2.5, so the triangle (0,1,2) with birth time 2.0 is included
    vr = VietorisRips(max_dim=2, max_epsilon=2.5)
    filtration = vr.fit_transform(X)
    
    ph = PersistentHomology()
    intervals = ph.fit_transform(filtration)
    
    # H0 components:
    # - Start with 3 components born at 0
    # - Edge (0,1) added at 1.0: one dies
    # - Edge (1,2) added at 1.5: one dies
    # - One component lives to infinity
    h0 = intervals.get(0, [])
    assert len(h0) == 3
    assert h0[0] == (0.0, 1.0)
    assert h0[1] == (0.0, 1.5)
    assert h0[2] == (0.0, float('inf'))
    
    # H1 components:
    # - The loop is completed when edge (0,2) is added at 2.0 -> birth of H1
    # - The loop is filled when triangle (0,1,2) is added at 2.0 -> death of H1
    # Note: since edge (0,2) has birth 2.0 and triangle has birth 2.0, the loop dies immediately at 2.0.
    h1 = intervals.get(1, [])
    assert len(h1) == 1
    assert h1[0] == (2.0, 2.0)
