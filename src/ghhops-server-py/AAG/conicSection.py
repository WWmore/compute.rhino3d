# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division


import numpy as np
#import copy
import scipy

from meshpy import Mesh

#------------------------------------------------------------------------------
#                                 GENERATION
#------------------------------------------------------------------------------

def ellipse(C=[0,0,0], a=2, b=1, Fa=80):
    """(x-c[0])**2/a**2 + (y-c[1])**2/b**2 = 1
        x = c[0] + a*cos(phi)
        y = c[1] + b*sin(phi)
        a: horizon half-axis
        b: vertical half-axis
    """
    phi = 2 * np.pi * np.arange(Fa+1)/float(Fa)
    x = C[0] + a * np.cos(phi)
    y = C[1] + b * np.sin(phi)
    v = np.vstack((x, y, np.zeros(Fa+1))).T
    return v

def ellipse3D(C=[0,0,0],r1=2,r2=1,e1=[1,0,0],e2=[0,1,0],Fa=80):
    """
    [x,y,z]=C + r1*e1*cos + r2*e2*sin
    """
    phi = 2 * np.pi * np.arange(Fa+1)/float(Fa)
    x = C[0] + r1*e1[0]*np.cos(phi)+r2*e2[0]*np.sin(phi)
    y = C[1] + r1*e1[1]*np.cos(phi)+r2*e2[1]*np.sin(phi)
    z = C[2] + r1*e1[2]*np.cos(phi)+r2*e2[2]*np.sin(phi)
    v = np.vstack((x, y, z)).T
    return v

def circle3D(C=[0,0,0], ri=1, N=[0,0,1], Fa=80):
    """
    center(list),
    radius(number),
    normal vectors(list)
    to get spatial circle mesh
    """
#    if C[2]==1:
#        u = [1,0,0]
#    else:
#        u = [-C[1],C[0],0]
#    u = np.array(u)/np.sqrt(u[0]**2+u[1]**2+u[2]**2)
#    v = np.cross(u, n)
#    v = v / np.linalg.norm(v,axis=1)
#    pass
    from huilab.huimesh.orthogonalVectors import Basis
    P1,P2,P3 = Basis(N,anchors=C,r=ri).orthogonalPlane3Vectors()
    from geometrylab.geometry.circle import circle_three_points
    C = circle_three_points(P1,P2,P3)
    C.make_vertices()
    verList = C.vertices
    Fa = C.sampling
    fList = (np.arange(len(verList)).reshape(-1,Fa)).tolist()
    cmesh = Mesh()
    cmesh.make_mesh(verList, fList)
    return C,cmesh

def vs_sphere_equation(V,vneib):
    "V=vertices_coordinate, index: vneib=[v,neib]"
    "P = [x^2+y^2+z^2,x,y,z,1]"
    Pc0 = np.einsum('ij,ij->i',V[vneib],V[vneib])
    P = np.c_[Pc0, V[vneib], np.ones(len(vneib))]
    H = np.einsum('ij,jk',P.T,P)
    Q = np.array([[0,0,0,0,-2],[ 0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[-2,0,0,0,0]])
    vals, vecs = scipy.linalg.eig(H,Q)
    vals = np.abs(vals)
    smallest = list(vals).index(min(list(vals[vals>=0])))
    ####smallest = np.argmin(vals)
    vector = vecs[:,smallest]
    A, B, C, D, E = vector[0],vector[1],vector[2],vector[3],vector[4]
    delt = np.sqrt(np.abs(B*B+C*C+D*D-4*A*E))
    A,B,C,D,E = A/delt,B/delt,C/delt,D/delt,E/delt
    eps = np.finfo(float).eps
    A = A+eps
    cM = -1/(2*A) * np.array([B, C, D])
    r = np.abs((B*B+C*C+D*D-4*A*E)/(4*A*A))
    if A < 0:
        coo = [-A,-B,-C,-D,-E]
    else:
        coo = [A,B,C,D,E]
    # test: coo * V[vneib] * coo--->0
   # print 'test', vals[smallest], np.dot(np.dot(np.array(coo), H),np.array(coo).reshape(-1,1))
    return coo, cM, np.sqrt(r)

def sphere_equation(V,order,somelist):
    "somelist = ringlist / vringlist"
    V4 = V[order]
    coolist,clist,rlist = [],[],[]
    for v in order:
        ring = somelist[v]
        coo,c,r = vs_sphere_equation(V,ring)
        coolist.append(coo)
        clist.append(c)
        rlist.append(r)
    coo = np.array(coolist)
    sphere_coeff = (coo.T).flatten()
    M = np.array(clist).reshape(-1,3)
    nx = 2*coo[:,0]*V4[:,0]+coo[:,1]
    ny = 2*coo[:,0]*V4[:,1]+coo[:,2]
    nz = 2*coo[:,0]*V4[:,2]+coo[:,3]
    sphere_n = np.c_[nx,ny,nz].flatten()
    return M,np.array(rlist),sphere_coeff,sphere_n


def sphere_equation0(V,order4,ringlist,neib_list=False):
    "V:all vertex_coor, V4: only valence 4"
    V4 = V[order4]
    coolist,clist,rlist,vnlist = [],[],[],[]
    for i in order4:
        vneib = [i]
        vneib.extend(ringlist[i])
        if neib_list:
            vnlist.append(ringlist[i])
        coo,c,r = vs_sphere_equation(V,vneib)
        coolist.append(coo)
        clist.append(c)
        rlist.append(r)
    coo = np.array(coolist)
    sphere_coeff = (coo.T).flatten()
    M = np.array(clist).reshape(-1,3)
    nx = 2*coo[:,0]*V4[:,0]+coo[:,1]
    ny = 2*coo[:,0]*V4[:,1]+coo[:,2]
    nz = 2*coo[:,0]*V4[:,2]+coo[:,3]
    sphere_n = np.c_[nx,ny,nz].flatten()

    if neib_list:
        return M,np.array(rlist),sphere_coeff,sphere_n, vnlist
    else:
        return M,np.array(rlist),sphere_coeff,sphere_n



def interpolate_sphere(V0,V1,V2,V3,V4):
    "interpolate 5 vertices"
    num = len(V0)
    allV = np.vstack((V0,V1,V2,V3,V4))
    arr = num*np.arange(5)
    def _vs(partV):
        "P = [x^2+y^2+z^2,x,y,z,1]"
        Pc0 = np.einsum('ij,ij->i',partV,partV)
        P = np.c_[Pc0, partV, np.ones(5)]
        H = np.einsum('ij,jk',P.T,P)
        Q = np.array([[0,0,0,0,-2],[ 0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[-2,0,0,0,0]])
        vals, vecs = scipy.linalg.eig(H,Q)
        vals = np.abs(vals)
        smallest = list(vals).index(min(list(vals[vals>=0])))
        vector = vecs[:,smallest]
        A, B, C, D, E = vector[0],vector[1],vector[2],vector[3],vector[4]
        delt = np.sqrt(np.abs(B*B+C*C+D*D-4*A*E))
        A,B,C,D,E = A/delt,B/delt,C/delt,D/delt,E/delt
        cM = -1/(2*A) * np.array([B, C, D])
        r = np.abs((B*B+C*C+D*D-4*A*E)/(4*A*A))
        if A < 0:
            coo = [-A,-B,-C,-D,-E]
        else:
            coo = [A,B,C,D,E]
        return coo, cM, np.sqrt(r)

    coolist,clist,rlist = [],[],[]
    for i in range(num):
        partV = allV[i+arr]
        coo,c,r = _vs(partV)
        coolist.append(coo)
        clist.append(c)
        rlist.append(r)
    coo = np.array(coolist)
    sphere_coeff = (coo.T).flatten()
    M = np.array(clist).reshape(-1,3)
    nx = 2*coo[:,0]*V0[:,0]+coo[:,1]
    ny = 2*coo[:,0]*V0[:,1]+coo[:,2]
    nz = 2*coo[:,0]*V0[:,2]+coo[:,3]
    sphere_n = -np.c_[nx,ny,nz]#.flatten()
    return M,np.array(rlist),sphere_coeff,sphere_n