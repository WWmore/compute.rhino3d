# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:26:01 2020

@author: Davide
"""

import numpy as np


def orient_rings(mesh):
    h_boundary = mesh.boundary_halfedges()
    v_visited = np.full(mesh.V, False)
    _, _, l = mesh.vertex_ring_vertices_iterators(return_lengths=True, sort=True)
    v_visited[np.where(l != 4)] = True
    R = []
    for h in range(2*mesh.E):
        h_ring = mesh.halfedge_ring(h)
        if len(h_ring) == 4:
            v = mesh.origin(h)
            v_ring = mesh.origin(mesh.twin(h_ring))
            R.append([v, v_ring[0], v_ring[1], v_ring[2], v_ring[3]])
            v_visited[v] = True
            H1 = [h_ring[2]]
            H2 = [h_ring[1]]
            H3 = [h_ring[0]]
            H4 = [h_ring[3]]
            break
    while not np.all(v_visited):
    #for i in range(80):
        H1_new = []
        H2_new = []
        H3_new = []
        H4_new = []
        for h in H1:
            ht = mesh.twin(h)
            h_ring = mesh.halfedge_ring(ht)
            if len(h_ring) == 4:
                v = mesh.origin(ht)
                v_ring = mesh.origin(mesh.twin(h_ring))
                if not v_visited[v]:
                    pass
                    R.append([v, v_ring[0], v_ring[1], v_ring[2], v_ring[3]])
                v_visited[v] = True
                H1_new.append(h_ring[2])
                H2_new.append(h_ring[1])
                H4_new.append(h_ring[3])
            else:
                v_visited[mesh.origin(ht)] = True
        for h in H2:
            ht = mesh.twin(h)
            h_ring = mesh.halfedge_ring(ht)
            if len(h_ring) == 4:
                v = mesh.origin(ht)
                v_ring = mesh.origin(mesh.twin(h_ring))
                if not v_visited[v]:
                    pass
                    R.append([v, v_ring[1], v_ring[2], v_ring[3], v_ring[0]])
                v_visited[v] = True
                H2_new.append(h_ring[2])
                H1_new.append(h_ring[3])
                H3_new.append(h_ring[1])
            else:
                v_visited[mesh.origin(ht)] = True
        for h in H3:
            ht = mesh.twin(h)
            h_ring = mesh.halfedge_ring(ht)
            if len(h_ring) == 4:
                v = mesh.origin(ht)
                v_ring = mesh.origin(mesh.twin(h_ring))
                if not v_visited[v]:
                    pass
                    R.append([v, v_ring[2], v_ring[3], v_ring[0], v_ring[1]])
                v_visited[v] = True
                H3_new.append(h_ring[2])
                H4_new.append(h_ring[1])
                H2_new.append(h_ring[3])
            else:
                v_visited[mesh.origin(ht)] = True
        for h in H4:
            ht = mesh.twin(h)
            h_ring = mesh.halfedge_ring(ht)
            if len(h_ring) == 4:
                v = mesh.origin(ht)
                v_ring = mesh.origin(mesh.twin(h_ring))
                if not v_visited[v]:
                    pass
                    R.append([v, v_ring[3], v_ring[0], v_ring[1], v_ring[2]])
                v_visited[v] = True
                H3_new.append(h_ring[3])
                H4_new.append(h_ring[2])
                H1_new.append(h_ring[1])
            else:
                v_visited[mesh.origin(ht)] = True
        H1 = np.unique(np.array(H1_new))
        H1 = H1[np.invert(np.in1d(H1, h_boundary))]
        H1 = H1[np.invert(np.in1d(H1, mesh.twin(h_boundary)))]
        H2 = np.unique(np.array(H2_new))
        H2 = H2[np.invert(np.in1d(H2, h_boundary))]
        H2 = H2[np.invert(np.in1d(H2, mesh.twin(h_boundary)))]
        H3 = np.unique(np.array(H3_new))
        H3 = H3[np.invert(np.in1d(H3, h_boundary))]
        H3 = H3[np.invert(np.in1d(H3, mesh.twin(h_boundary)))]
        H4 = np.unique(np.array(H4_new))
        H4 = H4[np.invert(np.in1d(H4, h_boundary))]
        H4 = H4[np.invert(np.in1d(H4, mesh.twin(h_boundary)))]
    return(R)
