# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:27:32 2021

@author: wangh0m
"""

import numpy as np

def osculating_circle(V0,V1,V3):
    """ Frenet frame
    inscribed circle of three points
    t = l1**2*(V3-V0) - l3**2*(V1-V0)
    """
    if V0.reshape(-1,3).shape[0] ==1:
        l1 = np.linalg.norm(V1-V0)
        l3 = np.linalg.norm(V3-V0)
        t = l1**2*(V3-V0) - l3**2*(V1-V0)
        t = t / np.linalg.norm(t)
        b = np.cross(V3-V0,V1-V0)
        b = b / np.linalg.norm(b)
        n = np.cross(b,t)
        cos = np.dot(t, (V3-V0)/np.linalg.norm(V3-V0))
        if cos**2>1:
            sin=0
        else:
            sin = np.sqrt(1-cos**2)
        eps = np.finfo(float).eps
        r = np.linalg.norm(V3-V0) / (2*sin+eps)
        O = V0 + r*n
        return [t,n,b], r, O
    else:
        "multiple input data, to get multiple frame data"
        l1 = np.linalg.norm(V1-V0,axis=1)
        l3 = np.linalg.norm(V3-V0,axis=1)
        t = (V3-V0)*(l1**2)[:,None] - (V1-V0)*(l3**2)[:,None]
        t = t / np.linalg.norm(t,axis=1)[:,None]
        b = np.cross(V3-V0,V1-V0)
        b = b / np.linalg.norm(b,axis=1)[:,None]
        n = np.cross(b,t)
        cos = np.einsum('ij,ij->i',t, (V3-V0)/np.linalg.norm(V3-V0,axis=1)[:,None])
        
        sin = np.zeros(len(cos))
        i = np.where(cos**2<1)[0]
        sin[i] = np.sqrt(1-cos[i]**2)
        
        eps = np.finfo(float).eps
        r = np.linalg.norm(V3-V0,axis=1) / (2*sin+eps)
        O = V0 + n*r[:,None]
        return [t,n,b], r, O   

def frenet_frame(V0,V1,V3,):
    "FRENET FRAME: (E1,E2,E3)"
    if V0.reshape(-1,3).shape[0] ==1:
        l1 = np.linalg.norm(V1-V0)
        l3 = np.linalg.norm(V3-V0)
        t = l1**2*(V3-V0) - l3**2*(V1-V0)
        t = t / np.linalg.norm(t)
        b = np.cross(V3-V0,V1-V0)
        b = b / np.linalg.norm(b)
        n = np.cross(b,t)
        return [t,n,b]
    else:
        "multiple input data, to get multiple frame data"
        l1 = np.linalg.norm(V1-V0,axis=1)
        l3 = np.linalg.norm(V3-V0,axis=1)
        t = (V3-V0)*(l1**2)[:,None] - (V1-V0)*(l3**2)[:,None]
        t = t / np.linalg.norm(t,axis=1)[:,None]
        b = np.cross(V3-V0,V1-V0)
        b = b / np.linalg.norm(b,axis=1)[:,None]
        n = np.cross(b,t)
        return [t,n,b]
        
def curvature(N,V0,V1,V3):
    """ integrate of normal_curvature + geodesic_curvature
    """
    frame, r, _ = osculating_circle(V0, V1, V3)
    n = frame[1]
    cos = np.dot(N, n)
    if cos**2>1:
        sin=0
    else:
        sin = np.sqrt(1-cos**2)
    k = 1/r
    kg = k * sin
    kn = k * cos
    return kn, kg

def darboux_vector(tau,V0,V1,V3):
    "d = tau*e1 + k*e3"
    frame, r, _ = osculating_circle(V0, V1, V3)
    k = 1/r
    if tau is None: #TODO
        "need to represent tau from 4 consecutive points"
        pass
    d = tau*frame[0] + k*frame[2]
    return d
    

def endpoint_curvature(V0,V1,V2, n):
    "get k of polyline V0-V1-V2 at V0, using Bezier curve polygon way"
    if V0.reshape(-1,3).shape[0] ==1:
        k = (n-1)/n * np.linalg.norm(np.cross(V1-V0,V2-V1))
        k /= np.linalg.norm(V1-V0)**3
    else:
        k = (n-1)/n * np.linalg.norm(np.cross(V1-V0,V2-V1),axis=1)
        k /= np.linalg.norm(V1-V0,axis=1)**3
    return k

def normal_curvature(N,V0,V1,V3):
    """ from osculating circle and surface normalN to get kn
        kn = k * cos
    """
    frame, r, _ = osculating_circle(V0, V1, V3)
    n = frame[1]
    cos = np.dot(N, n)
    k = 1/r
    kn = k * cos
    return kn

def geodesic_curvature(N,V0,V1,V3,way2=True):
    """ from osculating circle and surface normalN to get kg
        kg = k * sin
    """
    frame, r, _ = osculating_circle(V0, V1, V3)
    n = frame[1]
    cos = np.dot(N, n)
    if cos**2>1:
        sin=0
    else:
        sin = np.sqrt(1-cos**2)
    k = 1/r

    if way2:
        "if N passing through plane {V0,V1,V3}"
        n = np.cross(V1-V0,V3-V0)
        n = n / np.linalg.norm(n)
        cos = np.dot(N, n)
        kg = k * cos
    else:
        kg = k * sin
        
    return kg

def edge_geodesic_curvature(N0,V0,V1,V3,smooth=False):
    """N0 corresponding to normals at V0:=V[id0]
       len(N0)==len(N1)==len(N3)==len(V0) >=1
    """
    data = []
    for i in range(len(V0)):
        kg = geodesic_curvature(N0[i], V0[i], V1[i], V3[i])
        data.append(kg)
    data = np.array(data)
    #print('kg:[min,mean,max]=',np.min(data),np.mean(data),np.max(data)) 
    if smooth:
        "n-1 value smooth to n value"
        data = np.r_[data[0],(data[1:]+data[:-1])/2,data[-1]]
    daa = np.setdiff1d(data, np.r_[np.min(data),np.max(data)])
    print('kg:[min,mean,max]=','%.2g' % np.min(daa),'%.2g' % np.mean(daa),'%.2g' % np.max(daa))   
    return data

def edge_geodesic_curvature0(M,id0,id1,id3,N):
    """M: mesh; N corresponding to normals at v[id0]
       len(id0)==len(id1)==len(id3)==len(N) >=1
    """
    H = M.halfedges
    V = M.vertices
    V0,V1,V3 = V[id0],V[id1],V[id3]
    v1, v2 = M.edge_vertices()
    data = np.zeros(len(v1))
    
    for i in range(len(id0)):
        kg = geodesic_curvature(N[i], V0[i], V1[i], V3[i])
        ie0 = np.where(H[:,0]==id0[i])[0]
        ie1 = np.where(H[H[:,4],0]==id1[i])[0]
        ie3 = np.where(H[H[:,4],0]==id3[i])[0]
        i1,i3 = np.intersect1d(ie0,ie1)[0],np.intersect1d(ie0,ie3)[0]
        if data[H[i1,5]]==0:
            data[H[i1,5]] += kg
        else:
            data[H[i1,5]] = (data[H[i1,5]]+kg)/2
        if data[H[i3,5]]==0:
            data[H[i3,5]] += kg
        else:
            data[H[i3,5]] = (data[H[i3,5]]+kg)/2
    
    return data
        
def edge_liouville_equation(N0,V0,V1,V2,V3,V4,Va,Vb,Vc,Vd,smooth=True):
    """ parameter isolines are [1,v0,3; 2,v0,4], diagonal [a,0,c]
               a    1    d
               2    v0   4
               b    3    c     
       angle = <1-3  , a-c>        
    """
    kg_c,kg_s = [], []
    co = []
    for i in range(len(V0)):    
        kg13 = geodesic_curvature(N0[i],V0[i],V1[i],V3[i],way2=True)
        kg24 = geodesic_curvature(N0[i],V0[i],V2[i],V4[i],way2=True)  

        fr13,_,_ = osculating_circle(V0[i],V1[i],V3[i])
        t13= fr13[0]
        fr,_,_ = osculating_circle(V0[i],Va[i],Vc[i])
        t = fr[0]  
        cos = np.dot(t13,t)
        co.append(cos)
        if cos**2>1:
            sin=0
            print('cos**2>1',i)
        else:
            sin = np.sqrt(1-cos**2)
        kg1cos, kg2sin = kg13*cos, kg24*sin
        kg_c.append(kg1cos)
        kg_s.append(kg2sin)
    data = np.array(kg_c) + np.array(kg_s)
    #data = np.abs(np.array(kg_c)) - np.abs(np.array(kg_s))
    if smooth:
        "n-1 value smooth to n value"
        data = np.r_[data[0],(data[1:]+data[:-1])/2,data[-1]]
    print('kg1cos+kg2sin:[min,mean,max]=\n','%.2g' % np.min(data),'%.2g' % np.mean(data),'%.2g' % np.max(data)) 
   # print( np.array(kg_c), np.array(kg_s))
    #print(np.mean(np.array(cos)),np.max(np.array(cos)))
    return data    

