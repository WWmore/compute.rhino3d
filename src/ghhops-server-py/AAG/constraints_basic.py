# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:35:11 2021

@author: wangh0m
"""

import numpy as np

from scipy import sparse

"""
from constraints_basic import columnnew, con_laplacian_fairness,con_fair_midpoint,\
con_edge,con_unit_vector,con_unit,con_constl,con_assignvalue,\
con_positive,con_multiply,con_constangle,con_constangle2,\
con_isometry,con_isometry_edges,con_bigger_than,con_planarity,con_planar_vs,\
con_ortho,con_diagonal,con_equal_length,con_equal_vector,\
con_orthogonal_checkboard,con_ortho_vectors,con_osculating_tangent,\
con_equal_opposite_angle,\
con_dot,con_cross,con_cross_product2,con_dependent_vector,con_dependent_vector3,\
con_bisecting_vector,con_normal_constraints

con_planarity_constraints,con_edge_length_constraints,\
con_area_constraints,con_vector_area_constraints,con_circularity_constraints,\
con_unit_tangentplane_normal
"""
# -------------------------------------------------------------------------
#                           general / basic
# -------------------------------------------------------------------------

def columnnew(arr, num1, num2):
    """
    Parameters
    ----------
    array : array([1,4,7]).
    num1 : starting num.=100
    num2 : interval num.= 10

    Returns
    -------
    a : array(100+[1,4,7, 10,14,17, 20,24,27]).
    """
    a = num1 + np.r_[arr, num2+arr, 2*num2+arr]
    return a

def column(array):
    """
    Parameters
    ----------
    array : array([1,4,7]).

    Returns
    -------
    a : array([3,4,5, 12,13,14, 21,22,23]).
    """
    a, b, c = (3*array).repeat(3), (3*array+1).repeat(3), (3*array+2).repeat(3)
    a[1::3]=b[1::3]
    a[2::3]=c[2::3]
    return a

def con_square_number(X, Nr,Na,N):
    "only number equation: a=r^2"
    col = np.array([Nr,Na],dtype=int)
    row = np.zeros(2)
    data = np.array([2*X[Nr],-1])
    r = np.array([X[Nr]**2])
    H = sparse.coo_matrix((data,(row,col)), shape=(1, N))
    return H,r

def con_fair_midpoint0(c_v,c_vl,c_vr,N):
    "vl+vr-2v = 0"
    num = int(len(c_v)/3)     
    arr = np.arange(3*num)
    one = np.ones(3*num)
    row = np.tile(arr,3)
    col = np.r_[c_v,c_vl,c_vr]
    data = np.r_[2*one,-one,-one]
    K = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))       
    return K

def con_fair_midpoint(v,vl,vr,move,Vnum,N):
    "vl+vr-2v = 0"
    num = len(v)      
    arr = np.arange(3*num)
    one = np.ones(3*num)
    row = np.tile(arr,3)
    cc = columnnew(v, move,Vnum)
    c1 = columnnew(vl,move,Vnum)
    c2 = columnnew(vr,move,Vnum)
    col = np.r_[cc,c1,c2]
    data = np.r_[2*one,-one,-one]
    K = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))       
    return K

def con_fair_midpoint2(c_v,c_vr,Vl,N,efair):
    "Given Vl: 2*v-vr = Vl"
    num = int(len(c_v)/3)     
    arr = np.arange(3*num)
    one = np.ones(3*num)
    row = np.tile(arr,2)
    col = np.r_[c_v,c_vr]
    data = np.r_[2*one,-one]
    K = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))  
    s = Vl.flatten('F')     
    return K * efair, s * efair

def con_laplacian_fairness(v,neib,move,Vnum,N):
    "v1+v2+..+vn = n*v"
    valence = neib.shape[1] # column num 
    num = len(v)   
    one = np.ones(3*num)
    row = np.tile(np.arange(3*num),valence+1)
    col = columnnew(v, move,Vnum)
    data = valence*one
    for i in range(valence):
        ci = columnnew(neib[:,i], move,Vnum)
        col = np.r_[col, ci]
        data = np.r_[data,-one]
    K = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))       
    return K

def con_edge(X,c_v1,c_v3,c_ld1,c_ud1,num,N):
    "(v1-v3) = ld1*ud1"
    ld1 = X[c_ld1]
    ud1 = X[c_ud1]
    a3 = np.ones(3*num)
    row = np.tile(np.arange(3*num),4)
    col = np.r_[c_v1,c_v3,np.tile(c_ld1,3),c_ud1]
    data = np.r_[a3,-a3,-ud1,-np.tile(ld1,3)]
    r = -np.tile(ld1,3)*ud1
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))
    return H,r

def con_unit_tangentplane_normal(X,c_v1,c_v2,c_v3,c_v4,c_n,c_l,N):
    "uN * l = (v3-v1) x (v4-v2); uN^2=1"
    num = int(len(c_v1)/3)
    arr = np.arange(num)
    c_v1x,c_v1y,c_v1z = c_v1[:num],c_v1[num:2*num],c_v1[2*num:]
    c_v2x,c_v2y,c_v2z = c_v2[:num],c_v2[num:2*num],c_v2[2*num:]
    c_v3x,c_v3y,c_v3z = c_v3[:num],c_v3[num:2*num],c_v3[2*num:]
    c_v4x,c_v4y,c_v4z = c_v4[:num],c_v4[num:2*num],c_v4[2*num:]
    c_nx,c_ny,c_nz = c_n[:num],c_n[num:2*num],c_n[2*num:]
    
    row = np.r_[np.tile(arr,10),np.tile(arr,10)+num,np.tile(arr,10)+2*num]
        
    def _xyz(c_v1y,c_v1z,c_v2y,c_v2z,c_v3y,c_v3z,c_v4y,c_v4z, c_nx):
        "(v3y-v1y)*(v4z-v2z) - (v3z-v1z)*(v4y-v2y) - nx*l =0"
        xxx = np.r_[c_v1y,c_v1z,c_v2y,c_v2z,c_v3y,c_v3z,c_v4y,c_v4z]
        x1,x2 = X[c_v4z]-X[c_v2z],X[c_v3y]-X[c_v1y]
        x3,x4 = X[c_v4y]-X[c_v2y],X[c_v3z]-X[c_v1z]
        colx = np.r_[xxx,c_nx,c_l]
        dx = np.r_[-x1,+x3,+x4,-x2,+x1,-x3,-x4,+x2,-X[c_l],-X[c_nx]]
        return colx, dx
    colx,dx = _xyz(c_v1y,c_v1z,c_v2y,c_v2z,c_v3y,c_v3z,c_v4y,c_v4z,c_nx)
    coly,dy = _xyz(c_v1z,c_v1x,c_v2z,c_v2x,c_v3z,c_v3x,c_v4z,c_v4x,c_ny)
    colz,dz = _xyz(c_v1x,c_v1y,c_v2x,c_v2y,c_v3x,c_v3y,c_v4x,c_v4y,c_nz)        

    col,data = np.r_[colx,coly,colz], np.r_[dx,dy,dz]
    v13 = (X[c_v3]-X[c_v1]).reshape(-1,3,order='F')
    v24 = (X[c_v4]-X[c_v2]).reshape(-1,3,order='F')
    r = np.cross(v13,v24).flatten('F') - X[c_n]*np.tile(X[c_l],3)
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N)) 

    H0,r0 = con_unit(X,c_n,num,N)
    H = sparse.vstack((H,H0))
    r = np.r_[r,r0]
    return H,r 

def con_unit_vector(X,c_t,c_ut,c_l,num,N):
    "t = ut * l"
    row = np.tile(np.arange(3*num),3)
    col = np.r_[c_t,c_ut,np.tile(c_l,3)]
    data = np.r_[-np.ones(3*num),X[np.tile(c_l,3)],X[c_ut]]
    r = X[c_ut] * X[np.tile(c_l,3)]
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))
    H0,r0 = con_unit(X,c_ut,num,N)
    H = sparse.vstack((H,H0))
    r = np.r_[r,r0]
    return H,r 

def con_unit(X,c_ud1,num,N):
    "ud1**2=1"
    arr = np.arange(num)
    row2 = np.tile(arr,3)
    col = c_ud1
    data = 2*X[col]
    r =  np.linalg.norm(X[col].reshape(-1,3,order='F'),axis=1)**2 + np.ones(num)
    H = sparse.coo_matrix((data,(row2,col)), shape=(num, N))
    return H,r

def con_constl(c_ld1,init_l1,num,N):
    "ld1 == const."
    row3 = np.arange(num,dtype=int)
    col = c_ld1
    data = np.ones(num,dtype=int)
    r = init_l1
    H = sparse.coo_matrix((data,(row3,col)), shape=(num, N))
    return H,r

def con_assignvalue(ca,init_a,num,N):
    "A = const.a"
    row = np.arange(num)
    col = ca
    data = np.ones(num)
    r = np.tile(init_a,num)
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_positive(X,c_K,c_a,num,N):
    "K=a^2"
    col = np.r_[c_a,c_K]
    row = np.tile(np.arange(num),2)
    data = np.r_[2*X[c_a],-np.ones(num)]
    r = X[c_a]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_multiply(X,c_i,c_ii,c_k,num,N):
    "II=I*K"
    col = np.r_[c_i,c_ii,c_k]
    row = np.tile(np.arange(num),3)
    data = np.r_[X[c_k],-np.ones(num),X[c_i]]
    r = X[c_i]*X[c_k]
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r        
    
def con_constangle(X,c_ud1,c_ud2,angle0,num,N):
    "ud1*ud2 == const."
    row4 = np.tile(np.arange(num),6)
    col = np.r_[c_ud1,c_ud2]
    data = np.r_[X[c_ud2],X[c_ud1]]
    r = np.einsum('ij,ij->i',X[c_ud1].reshape(-1,3, order='F'),X[c_ud2].reshape(-1,3,order='F'))+angle0
    H = sparse.coo_matrix((data,(row4,col)), shape=(num, N))
    return H,r

def con_constangle2(X,c_u1,c_u2,c_a,num,N):
    "u1*u2 = a; a is 1 variable!"
    row = np.tile(np.arange(num),7)
    col = np.r_[c_u1,c_u2,c_a*np.ones(num)]
    data = np.r_[X[c_u2],X[c_u1],-np.ones(num)]
    r = np.einsum('ij,ij->i',X[c_u1].reshape(-1,3, order='F'),X[c_u2].reshape(-1,3, order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_isometry(l0,**kwargs):
    """
    keep all edge-lengths; no new variables
    (Vi-Vj)^2 = l0^2
    """
    w = kwargs.get('isometry')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    V = mesh.V
    vi, vj = mesh.vertex_ring_vertices_iterators(order=True)
    num = len(vi)
    c_vi = columnnew(vi,0,V)
    c_vj = columnnew(vj,0,V)
    H,r = con_isometry_edges(X,c_vi,c_vj,l0,num,N)
    return H*w, r*w
        
def con_isometry_edges(X,c_vi,c_vj,l0,num,N):
    "(Vi-Vj)^2 = l0^2"
    data1 = X[c_vi]
    data2 = X[c_vj]
    col = np.r_[c_vi, c_vj]
    data = 2*np.r_[data1-data2, data2-data1]
    row = np.tile(np.arange(num),6)
    r = np.linalg.norm((data1-data2).reshape(-1,3,order='F'),axis=1)**2 + l0**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_bigger_than(X,minl,c_vi,c_vj,c_ai,num,N):
    "(vi-vj)^2-ai^2=minl"
    col = np.r_[c_vi,c_vj,c_ai]
    row = np.tile(np.arange(num),7)
    data = 2*np.r_[X[c_vi]-X[c_vj], -X[c_vi]+X[c_vj], -X[c_ai]]
    r = np.linalg.norm((X[c_vi]-X[c_vj]).reshape(-1,3,order='F'),axis=1)**2
    r = r - X[c_ai]**2 + np.ones(num)*minl
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_planarity(X,c_v1,c_v2,c_n,num,N):
    """
    n*(v1-v2)=0
    """
    col = np.r_[c_n,c_v1,c_v2]
    row = np.tile(np.arange(num),9)
    data = np.r_[X[c_v1]-X[c_v2],X[c_n],-X[c_n]]
    r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),(X[c_v1]-X[c_v2]).reshape(-1,3, order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_ortho(X,c_v1,c_v2,c_n,num,N):
    """
    n*(v1+v2)=0
    """
    col = np.r_[c_n,c_v1,c_v2]
    row = np.tile(np.arange(num),9)
    data = np.r_[X[c_v1]+X[c_v2],X[c_n],X[c_n]]
    r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),(X[c_v1]+X[c_v2]).reshape(-1,3, order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_planar_vs(X,c_v,c_v1,c_v2,c_v3,c_v4,c_n,num,N):
    num = int(len(c_v)/3)
    H1,r1 = con_planarity(X,c_v,c_v1,c_n,num,N)
    H2,r2 = con_planarity(X,c_v,c_v2,c_n,num,N)
    H3,r3 = con_planarity(X,c_v,c_v3,c_n,num,N)
    H4,r4 = con_planarity(X,c_v,c_v4,c_n,num,N)
    Hn,rn = con_unit(X,c_n,num,N)
    H = sparse.vstack((H1,H2,H3,H4,Hn))
    r = np.r_[r1,r2,r3,r4,rn]
    return H,r
    
def con_diagonal(X,c_v1,c_v3,c_d1,num,N):
    "(v1-v3)^2=d1^2"
    row = np.tile(np.arange(num),7)
    col = np.r_[c_v1,c_v3,c_d1*np.ones(num)]
    dd = X[c_d1]*np.ones(num,dtype=int)
    data = 2*np.r_[X[c_v1]-X[c_v3],X[c_v3]-X[c_v1],-dd]
    r = np.linalg.norm((X[c_v1]-X[c_v3]).reshape(-1,3,order='F'),axis=1)**2-dd**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_diagonal2(X,c_v1,c_v3,c_ll,num,N):
    "(v1-v3)^2=ll"
    row = np.tile(np.arange(num),7)
    col = np.r_[c_v1,c_v3,c_ll]
    data = 2*np.r_[X[c_v1]-X[c_v3],X[c_v3]-X[c_v1],-0.5*np.ones(num)]
    r = np.linalg.norm((X[c_v1]-X[c_v3]).reshape(-1,3,order='F'),axis=1)**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_equal_length(X,c1,c2,c3,c4,num,N):
    "(v1-v3)^2=(v2-v4)^2"
    row = np.tile(np.arange(num),12)
    col = np.r_[c1,c2,c3,c4]
    data = 2*np.r_[X[c1]-X[c3],X[c4]-X[c2],X[c3]-X[c1],X[c2]-X[c4]]
    r = np.linalg.norm((X[c1]-X[c3]).reshape(-1,3, order='F'),axis=1)**2
    r = r-np.linalg.norm((X[c2]-X[c4]).reshape(-1,3, order='F'),axis=1)**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_equal_vector(c_vl,c_vr,num,N):
    "vl = vr"
    col = np.r_[c_vl,c_vr]
    row = np.tile(np.arange(3*num),2)
    one = np.ones(3*num)
    data = np.r_[one,-one]
    r = np.zeros(3*num)
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))
    return H,r

def con_orthogonal_checkboard(X,c_v1,c_v2,c_v3,c_v4,num,N):
    """for principal / isothermic / developable mesh / aux_diamond / aux_cmc
    (v1-v3)*(v2-v4)=0
    """
    col = np.r_[c_v1,c_v2,c_v3,c_v4]
    row = np.tile(np.arange(num),12)
    d1 = X[c_v2]-X[c_v4]
    d2 = X[c_v1]-X[c_v3]
    d3 = X[c_v4]-X[c_v2]
    d4 = X[c_v3]-X[c_v1]
    data = np.r_[d1,d2,d3,d4]
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    r = np.einsum('ij,ij->i',d1.reshape(-1,3, order='F'),d2.reshape(-1,3, order='F'))
    return H,r

def con_ortho_vectors(X,c_v1,c_v2,c_v3,c_v4,num,N):
    """for AGnet: Ng=(e2+e4) _|_ (v1-v),(v3-v)
    (v1+v3)*(v2-v4)=0
    """
    col = np.r_[c_v1,c_v2,c_v3,c_v4]
    row = np.tile(np.arange(num),12)
    d1 = X[c_v2]-X[c_v4]
    d2 = X[c_v1]+X[c_v3]
    d3 = X[c_v4]-X[c_v2]
    d4 = -X[c_v1]-X[c_v3]
    data = np.r_[d1,d2,d3,d4]
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    r = np.einsum('ij,ij->i',d1.reshape(-1,3, order='F'),d2.reshape(-1,3, order='F'))
    return H,r

def con_osculating_tangent(X,c_v,c_v1,c_v3,c_ll1,c_ll3,c_lt,c_t,num,N):
    """ [ll1,ll3,lt,t]
        lt*t = l1**2*(V3-V0) - l3**2*(V1-V0)
        t^2=1
    <===>
        ll1 = l1**2 = (V1-V0)^2
        ll3 = l3**2 = (V3-V0)^2
        ll1 * (v3-v0) - ll3 * (v1-v0) - t*l = 0
        t^2=1
    """
    col = np.r_[c_v,c_v1,c_v3,np.tile(c_ll1,3),np.tile(c_ll3,3),c_t,np.tile(c_lt,3)]
    row = np.tile(np.arange(3*num),7)
    d_l1, d_l3 = X[np.tile(c_ll1,3)], X[np.tile(c_ll3,3)]
    d_v,d_v1,d_v3 = X[c_v], X[c_v1], X[c_v3]
    data = np.r_[-d_l1+d_l3, -d_l3, d_l1, d_v3-d_v, d_v-d_v1, -X[np.tile(c_lt,3)],-X[c_t]]
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))
    r = (d_v3-d_v)*X[np.tile(c_ll1,3)]-(d_v1-d_v)*X[np.tile(c_ll3,3)]
    r -= X[np.tile(c_lt,3)]*X[c_t]
    Hl1,rl1 = con_diagonal2(X,c_v,c_v1,c_ll1,num,N)
    Hl3,rl3 = con_diagonal2(X,c_v,c_v3,c_ll3,num,N)
    Hu,ru = con_unit(X, c_t, num, N)
    H = sparse.vstack((H,Hl1,Hl3,Hu))
    r = np.r_[r,rl1,rl3,ru]
    return H,r

def con_equal_opposite_angle(X,c_e1,c_e2,c_e3,c_e4,num,N):
    "e1*e2-e3*e4=0"
    row = np.tile(np.arange(num),12)
    col = np.r_[c_e1,c_e2,c_e3,c_e4]
    data = np.r_[X[c_e2],X[c_e1], -X[c_e4], -X[c_e3]]
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    r = np.einsum('ij,ij->i',X[c_e1].reshape(-1,3, order='F'),X[c_e2].reshape(-1,3, order='F'))
    r -= np.einsum('ij,ij->i',X[c_e3].reshape(-1,3, order='F'),X[c_e4].reshape(-1,3, order='F'))
    return H,r
    
def con_dot(X,c_v1,c_v2,N):
    "v1*v2 = 0"
    num = int(len(c_v1)/3)
    row = np.tile(np.arange(num),6)
    col = np.r_[c_v1,c_v2]
    data = np.r_[X[c_v2], X[c_v1]]
    r = np.einsum('ij,ij->i',X[c_v1].reshape(-1,3,order='F'),X[c_v2].reshape(-1,3,order='F'))
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N)) 
    return H,r

def con_cross(X,c_a,c_b,c_n,N):
    "n = a x b = (a2b3-a3b2,a3b1-a1b3,a1b2-a2b3)"
    num = int(len(c_a)/3)
    arr = np.arange(num)
    one = np.ones(num)
    c_ax,c_ay,c_az = c_a[:num],c_a[num:2*num],c_a[2*num:]
    c_bx,c_by,c_bz = c_b[:num],c_b[num:2*num],c_b[2*num:]
    c_nx,c_ny,c_nz = c_n[:num],c_n[num:2*num],c_n[2*num:]
    
    row = np.tile(arr,5)
    def _cross(n1,a2,b3,a3,b2):
        "n1 = a2b3-a3b2"
        col = np.r_[a2,b3,a3,b2,n1]
        data = np.r_[X[b3],X[a2],-X[b2],-X[a3],-one]
        r = X[a2]*X[b3]-X[a3]*X[b2]
        H = sparse.coo_matrix((data,(row,col)), shape=(num,N)) 
        return H,r
    H1,r1 = _cross(c_nx,c_ay,c_bz,c_az,c_by)
    H2,r2 = _cross(c_ny,c_az,c_bx,c_ax,c_bz)
    H3,r3 = _cross(c_nz,c_ax,c_by,c_ay,c_bx)
    H = sparse.vstack((H1,H2,H3))
    r = np.r_[r1,r2,r3]
    return H,r
 
def con_cross_product(X,c_v1,c_v2,c_v3,c_v4,c_n,N):
    "(v3-v1) x (v4-v2) - N = 0"
    num = int(len(c_v1)/3)
    arr = np.arange(num)
    one = np.ones(num)
    c_v1x,c_v1y,c_v1z = c_v1[:num],c_v1[num:2*num],c_v1[2*num:]
    c_v2x,c_v2y,c_v2z = c_v2[:num],c_v2[num:2*num],c_v2[2*num:]
    c_v3x,c_v3y,c_v3z = c_v3[:num],c_v3[num:2*num],c_v3[2*num:]
    c_v4x,c_v4y,c_v4z = c_v4[:num],c_v4[num:2*num],c_v4[2*num:]
    c_nx,c_ny,c_nz = c_n[:num],c_n[num:2*num],c_n[2*num:]
    
    row = np.r_[np.tile(arr,9),np.tile(arr,9)+num,np.tile(arr,9)+2*num]
        
    def _xyz(c_v1y,c_v1z,c_v2y,c_v2z,c_v3y,c_v3z,c_v4y,c_v4z, c_nx):
        "(v3y-v1y)*(v4z-v2z) - (v3z-v1z)*(v4y-v2y) - nx =0"
        xxx = np.r_[c_v1y,c_v1z,c_v2y,c_v2z,c_v3y,c_v3z,c_v4y,c_v4z]
        x1,x2 = X[c_v4z]-X[c_v2z],X[c_v3y]-X[c_v1y]
        x3,x4 = X[c_v4y]-X[c_v2y],X[c_v3z]-X[c_v1z]
        colx = np.r_[xxx,c_nx]
        dx = np.r_[-x1,+x3,+x4,-x2,+x1,-x3,-x4,+x2,-one]
        return colx, dx
    colx,dx = _xyz(c_v1y,c_v1z,c_v2y,c_v2z,c_v3y,c_v3z,c_v4y,c_v4z,c_nx)
    coly,dy = _xyz(c_v1z,c_v1x,c_v2z,c_v2x,c_v3z,c_v3x,c_v4z,c_v4x,c_ny)
    colz,dz = _xyz(c_v1x,c_v1y,c_v2x,c_v2y,c_v3x,c_v3y,c_v4x,c_v4y,c_nz)        

    col,data = np.r_[colx,coly,colz], np.r_[dx,dy,dz]
    v13 = (X[c_v3]-X[c_v1]).reshape(-1,3,order='F')
    v24 = (X[c_v4]-X[c_v2]).reshape(-1,3,order='F')
    r = np.cross(v13,v24).flatten('F')
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N)) 
    return H,r

def con_cross_product2(X,c_v,c_v1,c_v3,c_n,N): 
    "N = (v1-v) x (v3-v) =  v1 x v3 - v1 x v + v3 x v"
    num = int(len(c_v1)/3)
    arr = np.arange(num)
    one = np.ones(num)
    c_x,c_y,c_z = c_v[:num],c_v[num:2*num],c_v[2*num:]
    c_1x,c_1y,c_1z = c_v1[:num],c_v1[num:2*num],c_v1[2*num:]
    c_3x,c_3y,c_3z = c_v3[:num],c_v3[num:2*num],c_v3[2*num:]
    c_nx,c_ny,c_nz = c_n[:num],c_n[num:2*num],c_n[2*num:]
    row = np.r_[np.tile(arr,7),np.tile(arr,7)+num,np.tile(arr,7)+2*num]
        
    def _xyz(c_1y,c_1z,c_3y,c_3z,c_y,c_z,c_nx):
        "n_x=1y*3z-1z*3y-1y*z+1z*y+3y*z-3z*y"
        cx = np.r_[c_1y,c_1z,c_3y,c_3z,c_y,c_z,c_nx]
        x1,x2 = X[c_3z]-X[c_z],X[c_y]-X[c_3y]
        x3,x4 = X[c_z]-X[c_1z],X[c_1y]-X[c_y]
        x5,x6 = X[c_1z]-X[c_3z],X[c_3y]-X[c_1y]
        dx = np.r_[x1,x2,x3,x4,x5,x6,-one] ## note one or -one
        return cx, dx
    colx,dx = _xyz(c_1y,c_1z,c_3y,c_3z,c_y,c_z,c_nx)
    coly,dy = _xyz(c_1z,c_1x,c_3z,c_3x,c_z,c_x,c_ny)
    colz,dz = _xyz(c_1x,c_1y,c_3x,c_3y,c_x,c_y,c_nz)        
    col,data = np.r_[colx,coly,colz], np.r_[dx,dy,dz]
    v1 = (X[c_v1]-X[c_v]).reshape(-1,3,order='F')
    v2 = (X[c_v3]-X[c_v]).reshape(-1,3,order='F')
    r = np.cross(v1,v2).flatten('F')
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N)) 
    return H,r

def con_cross_product3(X,c_v,c_v1,c_v3,c_l,c_n,N): 
    "l*N = (v1-v) x (v3-v) =  v1 x v3 - v1 x v + v3 x v"
    num = int(len(c_v1)/3)
    arr = np.arange(num)
    c_x,c_y,c_z = c_v[:num],c_v[num:2*num],c_v[2*num:]
    c_1x,c_1y,c_1z = c_v1[:num],c_v1[num:2*num],c_v1[2*num:]
    c_3x,c_3y,c_3z = c_v3[:num],c_v3[num:2*num],c_v3[2*num:]
    c_nx,c_ny,c_nz = c_n[:num],c_n[num:2*num],c_n[2*num:]
    row = np.r_[np.tile(arr,8),np.tile(arr,8)+num,np.tile(arr,8)+2*num]
        
    def _xyz(c_1y,c_1z,c_3y,c_3z,c_y,c_z,c_nx,c_l):
        "l*n_x=1y*3z-1z*3y-1y*z+1z*y+3y*z-3z*y"
        cx = np.r_[c_1y,c_1z,c_3y,c_3z,c_y,c_z,c_nx,c_l]
        x1,x2 = X[c_3z]-X[c_z],X[c_y]-X[c_3y]
        x3,x4 = X[c_z]-X[c_1z],X[c_1y]-X[c_y]
        x5,x6 = X[c_1z]-X[c_3z],X[c_3y]-X[c_1y]
        dx = np.r_[x1,x2,x3,x4,x5,x6,-X[c_l],-X[c_nx]]
        return cx, dx
    colx,dx = _xyz(c_1y,c_1z,c_3y,c_3z,c_y,c_z,c_nx,c_l)
    coly,dy = _xyz(c_1z,c_1x,c_3z,c_3x,c_z,c_x,c_ny,c_l)
    colz,dz = _xyz(c_1x,c_1y,c_3x,c_3y,c_x,c_y,c_nz,c_l)        
    col,data = np.r_[colx,coly,colz], np.r_[dx,dy,dz]
    v1 = (X[c_v1]-X[c_v]).reshape(-1,3,order='F')
    v2 = (X[c_v3]-X[c_v]).reshape(-1,3,order='F')
    r = np.cross(v1,v2).flatten('F')-X[c_n]*X[np.tile(c_l,3)]
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N)) 
    
    Hn,rn = con_unit(X, c_n, num, N)
    H = sparse.vstack((H,Hn))
    r = np.r_[r,rn]
    return H,r

def con_dependent_vector(X,c_a,c_b,c_t,N):
    """ three variables: a, b,t
    t x (a-b) = 0
    <==> t1*(a-b)2=t2*(a-b)1; 
         t2*(a-b)3=t3*(a-b)2; 
         t1*(a-b)3=t3*(a-b)1
    """
    num = int(len(c_a)/3)
    arr = np.arange(num)
    c_ax,c_ay,c_az = c_a[:num], c_a[num:2*num], c_a[2*num:]
    c_bx,c_by,c_bz = c_b[:num], c_b[num:2*num], c_b[2*num:]
    c_tx,c_ty,c_tz = c_t[:num], c_t[num:2*num], c_t[2*num:]
    def _cross(c_ax,c_ay,c_bx,c_by,c_tx,c_ty):
        "(ax-bx) * ty = (ay-by) * tx"
        col = np.r_[c_ax,c_ay,c_bx,c_by,c_tx,c_ty]
        row = np.tile(arr,6)
        data = np.r_[X[c_ty],-X[c_tx],-X[c_ty],X[c_tx],-X[c_ay]+X[c_by],X[c_ax]-X[c_bx]]
        r = (X[c_ax]-X[c_bx])*X[c_ty] - (X[c_ay]-X[c_by])*X[c_tx]
        H = sparse.coo_matrix((data,(row,col)), shape=(num, N)) 
        return H,r
    H12,r12 = _cross(c_ax,c_ay,c_bx,c_by,c_tx,c_ty)
    H23,r23 = _cross(c_az,c_ay,c_bz,c_by,c_tz,c_ty)
    H13,r13 = _cross(c_ax,c_az,c_bx,c_bz,c_tx,c_tz)
    H = sparse.vstack((H12,H23,H13))
    r = np.r_[r12,r23,r13]
    return H,r

def con_dependent_vector2(X,c_a,c_b,c_i,c_j,N):
    """ 4 variables: a,b,i,j
    two vectors (a-b) // (i-j)
    (a-b) x (i-j) = 0 
    <==> (i-j)1*(a-b)2=(i-j)2*(a-b)1; 
         (i-j)2*(a-b)3=(i-j)3*(a-b)2; 
         (i-j)1*(a-b)3=(i-j)3*(a-b)1
    """
    num = int(len(c_a)/3)
    arr = np.arange(num)
    c_ax,c_ay,c_az = c_a[:num], c_a[num:2*num], c_a[2*num:]
    c_bx,c_by,c_bz = c_b[:num], c_b[num:2*num], c_b[2*num:]
    c_ix,c_iy,c_iz = c_i[:num], c_i[num:2*num], c_i[2*num:]
    c_jx,c_jy,c_jz = c_j[:num], c_j[num:2*num], c_j[2*num:]
    def _cross(c_ax,c_ay,c_bx,c_by,c_ix,c_iy,c_jx,c_jy):
        "(ax-bx) * (iy-jy) = (ay-by) * (ix-jx)"
        col = np.r_[c_ax,c_ay,c_bx,c_by,c_ix,c_iy,c_jx,c_jy]
        row = np.tile(arr,8)
        data = np.r_[X[c_iy]-X[c_jy],-X[c_ix]+X[c_jx]] #c_ax,c_ay
        data = np.r_[data, -X[c_iy]+X[c_jy],X[c_ix]+X[c_jx]] #c_bx,c_by
        data = np.r_[data, -X[c_ay]+X[c_by],X[c_ax]-X[c_bx]] #c_ix,c_iy
        data = np.r_[data, X[c_ay]-X[c_by],-X[c_ax]+X[c_bx]] #c_jx,c_jy
        r = (X[c_ax]-X[c_bx])*(X[c_iy]-X[c_jy]) - (X[c_ay]-X[c_by])*(X[c_ix]-X[c_jx])
        H = sparse.coo_matrix((data,(row,col)), shape=(num, N)) 
        return H,r
    H12,r12 = _cross(c_ax,c_ay,c_bx,c_by,c_ix,c_iy,c_jx,c_jy)
    H23,r23 = _cross(c_az,c_ay,c_bz,c_by,c_iz,c_iy,c_jz,c_jy)
    H13,r13 = _cross(c_ax,c_az,c_bx,c_bz,c_ix,c_iz,c_jx,c_jz)
    H = sparse.vstack((H12,H23,H13))
    r = np.r_[r12,r23,r13]
    return H,r    
    
def con_dependent_vector3(X,c_a,c_b,t,N):
    """ given vectors t(shape=[n,3]); 2 variables: a, b 
    t x (a-b) = 0
    <==> t1*(a-b)2=t2*(a-b)1; 
         t2*(a-b)3=t3*(a-b)2; 
         t1*(a-b)3=t3*(a-b)1
    """
    num = int(len(c_a)/3)
    arr = np.arange(num)
    c_ax,c_ay,c_az = c_a[:num], c_a[num:2*num], c_a[2*num:]
    c_bx,c_by,c_bz = c_b[:num], c_b[num:2*num], c_b[2*num:]
    tx,ty,tz = t.T
    def _cross(c_ax,c_ay,c_bx,c_by,tx,ty):
        "linear eqs: (ax-bx) * ty = (ay-by) * tx"
        col = np.r_[c_ax,c_ay,c_bx,c_by]
        row = np.tile(arr,4)
        data = np.r_[ty,-tx,-ty,tx]
        r = np.zeros(num)
        H = sparse.coo_matrix((data,(row,col)), shape=(num, N)) 
        return H,r
    H12,r12 = _cross(c_ax,c_ay,c_bx,c_by,tx,ty)
    H23,r23 = _cross(c_az,c_ay,c_bz,c_by,tz,ty)
    H13,r13 = _cross(c_ax,c_az,c_bx,c_bz,tx,tz)
    H = sparse.vstack((H12,H23,H13))
    r = np.r_[r12,r23,r13]
    return H,r

def con_bisecting_vector(X,c_n,c_a,c_b,N):
    """ three variables: n,a,b
    N x (a + b) = 0 = 0
    <==> n1*(a+b)2 = n2*(a+b)1; 
         n2*(a+b)3 = n3*(a+b)2; 
         n1*(a+b)3 = n3*(a+b)1
    """
    num = int(len(c_n)/3)
    arr = np.arange(num)
    c_ax,c_ay,c_az = c_a[:num], c_a[num:2*num], c_a[2*num:]
    c_bx,c_by,c_bz = c_b[:num], c_b[num:2*num], c_b[2*num:]
    c_tx,c_ty,c_tz = c_n[:num], c_n[num:2*num], c_n[2*num:]
    def _cross(c_ax,c_ay,c_bx,c_by,c_tx,c_ty):
        "(ax+bx) * ty = (ay+by) * tx"
        col = np.r_[c_ax,c_ay,c_bx,c_by,c_tx,c_ty]
        row = np.tile(arr,6)
        data = np.r_[X[c_ty],-X[c_tx],X[c_ty],-X[c_tx],-X[c_ay]-X[c_by],X[c_ax]+X[c_bx]]
        r = (X[c_ax]+X[c_bx])*X[c_ty] - (X[c_ay]+X[c_by])*X[c_tx]
        H = sparse.coo_matrix((data,(row,col)), shape=(num, N)) 
        return H,r
    H12,r12 = _cross(c_ax,c_ay,c_bx,c_by,c_tx,c_ty)
    H23,r23 = _cross(c_az,c_ay,c_bz,c_by,c_tz,c_ty)
    H13,r13 = _cross(c_ax,c_az,c_bx,c_bz,c_tx,c_tz)
    H = sparse.vstack((H12,H23,H13))
    r = np.r_[r12,r23,r13]
    return H,r    

    # -------------------------------------------------------------------------
    #                          Geometric Constraints (from Davide)
    # -------------------------------------------------------------------------

def con_normal_constraints(**kwargs):
    "represent unit normal"
    #w = kwargs.get('normal') * kwargs.get('geometric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    V = mesh.V
    F = mesh.F
    f = 3*V + np.arange(F)
    i = np.arange(F)
    i = np.hstack((i, i, i)) #row ==np.tile(i,3) == np.r_[i,i,i]
    j = np.hstack((f, F+f, 2*F+f)) #col ==np.r_[f,F+f,2*F+f]
    data = 2 * np.hstack((X[f], X[F+f], X[2*F+f])) #* w
    H = sparse.coo_matrix((data,(i,j)), shape=(F, N))
    r = ((X[f]**2 + X[F+f]**2 + X[2*F+f]**2) + 1) #* w
    return H,r

def con_planarity_constraints(**kwargs):
    "n*(vi-vj) = 0; Note: making sure normals is always next to V in X[V,N]"
    w = kwargs.get('planarity')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    V = mesh.V
    F = mesh.F
    f, v1, v2 = mesh.face_edge_vertices_iterators(order=True)
    K = f.shape[0]
    f = 3*V + f
    r = ((X[v2] - X[v1]) * X[f] + (X[V+v2] - X[V+v1]) * X[F+f]
         + (X[2*V+v2] - X[2*V+v1]) * X[2*F+f] ) * w
    v1 = np.hstack((v1, V+v1, 2*V+v1))
    v2 = np.hstack((v2, V+v2, 2*V+v2))
    f = np.hstack((f, F+f, 2*F+f))
    i = np.arange(K)
    i = np.hstack((i, i, i, i, i, i, i, i, i))
    j = np.hstack((f, v2, v1))
    data = 2 * np.hstack((X[v2] - X[v1], X[f], -X[f])) * w
    H = sparse.coo_matrix((data,(i,j)), shape=(K, N))
    Hn,rn = con_normal_constraints(**kwargs)
    H = sparse.vstack((H*w,Hn))
    r = np.r_[r*w,rn]
    return H,r

def con_edge_length_constraints(**kwargs):
    "(v1-v2)^2 - l^2 = 0"
    w = kwargs.get('edge_length') *kwargs.get('geometric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N1 = kwargs.get('N1')
    V = mesh.V
    E = mesh.E
    i = np.arange(E)
    e = N1 + i
    v1, v2 = mesh.edge_vertices()
    r = ( X[v1]**2 + X[v2]**2 - 2*X[v1]*X[v2]
        + X[V+v1]**2 + X[V+v2]**2 - 2*X[V+v1]*X[V+v2]
        + X[2*V+v1]**2 + X[2*V+v2]**2 - 2*X[2*V+v1]*X[2*V+v2]
        - X[e]**2 ) * w
    v1 = np.hstack((v1,V+v1,2*V+v1))
    v2 = np.hstack((v2,V+v2,2*V+v2))
    i = np.hstack((i,i,i,i,i,i,i))
    j = np.hstack((v1,v2,e))
    data = 2 * np.hstack((X[v1] - X[v2], X[v2] - X[v1], -X[e])) * w
    H = sparse.coo_matrix((data,(i,j)), shape=(E, N))
    #self.add_iterative_constraint(H, r, 'edge_length')
    return H*w,r*w

def con_area_constraints(**kwargs):
    """
    Huinote: to check!
    """
    import utilities
    w = kwargs.get('area') * kwargs.get('geometric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N2 = kwargs.get('N2')
    V = mesh.V
    F = mesh.F
    fi, v1, v2 = mesh.face_edge_vertices_iterators(order=True)
    v1a = np.hstack((V+v1,2*V+v1,v1))
    v1b = np.hstack((2*V+v1,v1,V+v1))
    v2a = np.hstack((V+v2,2*V+v2,v2))
    v2b = np.hstack((2*V+v2,v2,V+v2))
    r = 0.5 * (X[v1a] * X[v2b] - X[v1b] * X[v2a]) * w
    k = np.hstack((fi,F+fi,2*F+fi))
    r = utilities.sum_repeated(r,k)
    f = N2 + np.arange(F)
    i = np.hstack((k,k,k,k,np.arange(3*F)))
    j = np.hstack((v1a, v2b, v1b, v2a, f, F+f, 2*F+f))
    data1 = 0.5 * np.hstack((X[v2b], X[v1a], -X[v2a], -X[v1b]))
    data2 = - np.ones(3*F)
    data = np.hstack((data1, data2)) * w
    H = sparse.coo_matrix((data,(i,j)), shape=(3*F, N))
    #self.add_iterative_constraint(H, r, 'face_vector_area')
    return H*w,r*w

def con_vector_area_constraints(**kwargs):
    """
    Huinote: to check!
    """        
    w = kwargs.get('area') * kwargs.get('geometric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N2 = kwargs.get('N2')
    F = mesh.F
    f = N2 + np.arange(F)
    i = np.arange(F)
    i = np.hstack((i,i,i,i))
    j = np.hstack((f,F+f,2*F+f,3*F+f))
    data = 2 * np.hstack((X[f], X[F+f], X[2*F+f], - X[3*F+f])) * w
    H = sparse.coo_matrix((data,(i,j)), shape=(F, N))
    r = (X[f]**2 + X[F+f]**2 + X[2*F+f]**2 - X[3*F+f]**2) * w
    #self.add_iterative_constraint(H, r, 'face_area')
    return H*w,r*w

def con_circularity_constraints(**kwargs):
    "(v1-c)^2 = (v2-c)^2"
    w = kwargs.get('circularity')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N3 = kwargs.get('N3')
    V = mesh.V
    F = mesh.F
    f, v1, v2 = mesh.face_edge_vertices_iterators(order=True)
    K = f.shape[0]
    cx = N3 + np.array(f)
    cy = N3 + np.array(f) + F
    cz = N3 + np.array(f) + 2*F
    v1x = np.array(v1)
    v1y = np.array(V + v1)
    v1z = np.array(2*V + v1)
    v2x = np.array(v2)
    v2y = np.array(V + v2)
    v2z = np.array(2*V + v2)
    jx = np.hstack((v1x, v2x, cx))
    jy = np.hstack((v1y, v2y, cy))
    jz = np.hstack((v1z, v2z, cz))
    datax = np.hstack((X[v1x] - X[cx], -X[v2x] + X[cx], -X[v1x] + X[v2x]))
    rx = 0.5*X[v1x]**2 - 0.5*X[v2x]**2 -X[v1x]*X[cx] + X[v2x]*X[cx]
    datay = np.hstack((X[v1y] - X[cy], -X[v2y] + X[cy], -X[v1y] + X[v2y]))
    ry = 0.5*X[v1y]**2 - 0.5*X[v2y]**2 -X[v1y]*X[cy] + X[v2y]*X[cy]
    dataz = np.hstack((X[v1z] - X[cz], -X[v2z] + X[cz], -X[v1z] + X[v2z]))
    rz = 0.5*X[v1z]**2 - 0.5*X[v2z]**2 -X[v1z]*X[cz] + X[v2z]*X[cz]
    i = np.arange(K)
    i = np.hstack((i,i,i,i,i,i,i,i,i))
    j = np.hstack((jx, jy, jz))
    data = np.hstack((datax, datay, dataz)) * w
    r = (rx + ry + rz) * w
    H = sparse.coo_matrix((data,(i,j)), shape=(K, N))
    #self.add_iterative_constraint(H, r,'circularity')
    return H*w,r*w
