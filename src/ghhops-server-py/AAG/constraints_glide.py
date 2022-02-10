# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:29:34 2021

@author: WANGH0M
"""

import numpy as np

from scipy import sparse

import utilities

from constraints_basic import columnnew

    # -------------------------------------------------------------------------
    #                             Glide / Alignment
    # -------------------------------------------------------------------------
def con_alignments(w,refPoly, glideInd, overlap=False,**kwargs):
    #w = kwargs.get('i_boundary_glide')
    N = kwargs.get('N')
    null = np.zeros([0])
    H = sparse.coo_matrix((null,(null,null)), shape=(0,N))
    r = np.array([])
    for i in range(len(glideInd)):
        v, crv = glideInd[i], refPoly[i]
        Hi,ri = con_alignment(w,crv,v,**kwargs)
        H = sparse.vstack((H,Hi))
        r = np.r_[r,ri]
    return H,r

def con_alignment(w, refPoly, glideInd, overlap=False,**kwargs):
    """glide on a given polylineCurve (may be not closed)
       refPoly: reference polylineCurve
       glideInd: indices of constraint vertices from the mesh
       constraints: from tangents e1 to get orthogonal e2,e3,
                    then(P-Q)*e2=0; (P-Q)*e3=0
                    such that P-Q align nealy on direction e1
            where P (varaibles), and Q,e2,e3 are computed_concrete_values
       linear equations: P*e2=Q*e2; P*e3=Q*e3
                ~       [P;P] * [e2;e3] = [Q;Q] * [e2;e3]
       another way: P-Q // e1 <==> (P-Q) x e1 = 0
    """
    #w = kwargs.get('boundary_glide')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')

    ind = glideInd
    Vr = refPoly.vertices
    Tr = refPoly.vertex_tangents()
    c_p = columnnew(ind,0,mesh.V)
    P = X[c_p].reshape(-1,3,order='F')
    closeP = refPoly.closest_vertices(P)
    Q = Vr[closeP]
    e1 = Tr[closeP]
    e2 = utilities.orthogonal_vectors(e1)
    e3 = np.cross(e1,e2)
    r = np.r_[np.einsum('ij,ij->i',Q,e2),np.einsum('ij,ij->i',Q,e3)]

    num = len(ind)
    row = np.tile(np.arange(2*num),3)
    col = columnnew(np.tile(ind,2),0,mesh.V)
    ee = np.vstack((e2,e3))
    data = ee.flatten('F')
    H = sparse.coo_matrix((data,(row,col)), shape=(2*num,N))
    if overlap:
        "P=Q"
        row = np.arange(3*num)
        col = c_p
        data = np.ones(3*num)
        r0 = Q.flatten('F')
        H0 = sparse.coo_matrix((data,(row,col)), shape=(3*num,N))
        H = sparse.vstack((H,H0*1))
        r = np.r_[r,r0*1]
    return H*w,r*w

def con_fix_vertices( index, Vf,**kwargs):
    "X[column(index)]==Vf"
    w = kwargs.get('fix_point')
    mesh = kwargs.get('mesh')
    N = kwargs.get('N')
    row = np.arange(3*len(index))
    col = columnnew(index,0,mesh.V)
    data = np.ones(3*len(index))
    r = Vf
    H = sparse.coo_matrix((data,(row,col)), shape=(3*len(index),N))
    return H*w,r*w