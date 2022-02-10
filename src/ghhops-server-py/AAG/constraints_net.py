# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:16:29 2021

@author: WANGH0M
"""

import numpy as np

from scipy import sparse

from constraints_basic import columnnew,\
    con_edge,con_unit,con_constl,con_equal_length,\
    con_constangle2,con_constangle,con_unit_vector,con_dependent_vector,\
    con_planarity,con_osculating_tangent,con_diagonal,\
    con_equal_opposite_angle,\
    con_dot,con_cross_product2,con_bisecting_vector,\
    con_normal_constraints, con_planarity_constraints,\
    con_unit_tangentplane_normal

# -------------------------------------------------------------------------
#                           common used net-constraints:
# -------------------------------------------------------------------------
   

    #--------------------------------------------------------------------------
    #                       isogonals:
    #--------------------------------------------------------------------------  
def con_unit_edge(rregular=False,**kwargs): 
    """ unit_edge / unit_diag_edge
    X += [l1,l2,l3,l4,ue1,ue2,ue3,ue4]
    (vi-v) = li*ui, ui**2=1, (i=1,2,3,4)
    """
    if kwargs.get('unit_diag_edge'):
        w = kwargs.get('unit_diag_edge')
        diag=True   
    elif kwargs.get('unit_edge'):
        w = kwargs.get('unit_edge')
        diag=False
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    V = mesh.V
    if diag:
        v,v1,v2,v3,v4 = mesh.rr_star_corner 
    elif rregular:
        v,v1,v2,v3,v4 = mesh.rr_star[mesh.ind_rr_star_v4f4].T
    else:
        #v,v1,v2,v3,v4 = mesh.ver_regular_star.T # default angle=90, non-orient
        v,v1,v2,v3,v4 = mesh.ver_star_matrix.T # oriented
    num = len(v)
    c_v = columnnew(v,0,V)
    c_v1 = columnnew(v1,0,V)
    c_v2 = columnnew(v2,0,V)
    c_v3 = columnnew(v3,0,V)
    c_v4 = columnnew(v4,0,V)
    
    arr = np.arange(num)
    c_l1 = N5-16*num + arr
    c_l2 = c_l1 + num
    c_l3 = c_l2 + num
    c_l4 = c_l3 + num
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    c_ue4 = columnnew(arr,N5-3*num,num)

    H1,r1 = con_edge(X,c_v1,c_v,c_l1,c_ue1,num,N)
    H2,r2 = con_edge(X,c_v2,c_v,c_l2,c_ue2,num,N)
    H3,r3 = con_edge(X,c_v3,c_v,c_l3,c_ue3,num,N)
    H4,r4 = con_edge(X,c_v4,c_v,c_l4,c_ue4,num,N)
    Hu1,ru1 = con_unit(X,c_ue1,num,N)
    Hu2,ru2 = con_unit(X,c_ue2,num,N)
    Hu3,ru3 = con_unit(X,c_ue3,num,N)
    Hu4,ru4 = con_unit(X,c_ue4,num,N)

    H = sparse.vstack((H1,H2,H3,H4,Hu1,Hu2,Hu3,Hu4))
    r = np.r_[r1,r2,r3,r4,ru1,ru2,ru3,ru4]
    return H*w,r*w

def con_orthogonal(diagmesh=False,**kwargs): # simpliest one, for auxetic-cmc-case
    """(v1-v3)*(v2-v4)=0, no auxilary variables
    """
    if kwargs.get('orthogonal'):
        w = kwargs.get('orthogonal')
    elif kwargs.get('orthogonal_diag'):
        w = kwargs.get('orthogonal_diag')
        diagmesh=True
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    V = mesh.V
    if diagmesh:
        "(v1-v3)*(v2-v4)=0"
        v,v1,v2,v3,v4 = mesh.rr_star_corner
    else:
        v0, vj, l = mesh.vertex_ring_vertices_iterators(order=True,
                                                     return_lengths=True)
        ind = np.in1d(v0, np.where(l == 4)[0])
        v0 = v0[ind]
        vj = vj[ind]
        v = v0[::4]
        v1,v2,v3,v4 = vj[::4],vj[1::4],vj[2::4],vj[3::4]
    c_v1 = columnnew(v1,0,V)
    c_v2 = columnnew(v2,0,V)
    c_v3 = columnnew(v3,0,V)
    c_v4 = columnnew(v4,0,V)
    col = np.r_[c_v1,c_v2,c_v3,c_v4]
    num = len(v)
    row = np.tile(np.arange(num),12)
    d1 = X[c_v2]-X[c_v4]
    d2 = X[c_v1]-X[c_v3]
    d3 = X[c_v4]-X[c_v2]
    d4 = X[c_v3]-X[c_v1]
    data = np.r_[d1,d2,d3,d4]
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    r = np.einsum('ij,ij->i',d1.reshape(-1,3, order='F'),d2.reshape(-1,3, order='F'))
    #self.add_iterative_constraint(H*w, r*w, name)
    return H*w,r*w

def con_orthogonal_midline(**kwargs):
    """ this method is almost the same as above, minor differences at boundary
    control quadfaces: two middle line are orthogonal to each other
    quadface: v1,v2,v3,v4 
    middle lins: e1 = (v1+v2)/2-(v3+v4)/2; e2 = (v2+v3)/2-(v4+v1)/2
    <===> e1 * e2 = 0 <==> (v1-v3)^2=(v2-v4)^2
    """
    w = kwargs.get('orthogonal')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    num = mesh.num_quadface
    v1,v2,v3,v4 = mesh.rr_quadface.T # in odrder
    c_v1 = columnnew(v1,0,mesh.V)
    c_v2 = columnnew(v2,0,mesh.V)
    c_v3 = columnnew(v3,0,mesh.V)
    c_v4 = columnnew(v4,0,mesh.V)
    H,r = con_equal_length(X,c_v1,c_v2,c_v3,c_v4,num,N)
    return H*w,r*w 

def con_isogonal(cos0,assign=False,**kwargs):
    """
    keep tangent crossing angle
    X += [lt1,lt2, ut1,ut2, cos]
    (ue1-ue3) =  lt1 * ut1, ut1**2 = 1
    (ue2-ue4) =  lt2 * ut2, ut2**2 = 1
    ut1 * ut2 = cos
    if assign:
        cos == cos0
    """
    w = kwargs.get('isogonal')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    N6 = kwargs.get('N6')
    
    num = mesh.num_regular
    arr = np.arange(num)
    c_l1 = N6-8*num-1 + arr
    c_l2 = c_l1+num
    c_ut1 = columnnew(arr,N6-6*num-1,num)
    c_ut2 = columnnew(arr,N6-3*num-1,num)
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    c_ue4 = columnnew(arr,N5-3*num,num)
    H1,r1 = con_edge(X,c_ue1,c_ue3,c_l1,c_ut1,num,N)
    H2,r2 = con_edge(X,c_ue2,c_ue4,c_l2,c_ut2,num,N)
    Hu1,ru1 = con_unit(X,c_ut1,num,N)
    Hu2,ru2 = con_unit(X,c_ut2,num,N)
    Ha,ra = con_constangle2(X,c_ut1,c_ut2,N6-1,num,N)
    H = sparse.vstack((H1,H2,Hu1,Hu2,Ha))
    r = np.r_[r1,r2,ru1,ru2,ra]
    if assign:
        H0,r0 = con_constl(np.array([N6-1],dtype=int),cos0,1,N)
        H = sparse.vstack((H, H0))
        r = np.r_[r,r0]
    #self.add_iterative_constraint(H*w, r*w, 'isogonal')
    #print('err:isogonal:',np.sum(np.square(H*X-r)))
    return H*w,r*w

def con_isogonal_diagnet(cos0,assign=False,**kwargs):
    """
    keep tangent crossing angle, of diagnal directions
    X += [lt1,lt2, ut1,ut2, cos]
    (ue1-ue3) =  lt1 * ut1, ut1**2 = 1
    (ue2-ue4) =  lt2 * ut2, ut2**2 = 1
    ut1 * ut2 = cos
    if assign:
        cos == cos0
    """
    w = kwargs.get('isogonal_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    N6 = kwargs.get('N6')
    
    num = len(mesh.ind_rr_star_v4f4)
    arr = np.arange(num)
    c_l1 = N6-8*num-1 + arr
    c_l2 = c_l1+num
    c_ut1 = columnnew(arr,N6-6*num-1,num)
    c_ut2 = columnnew(arr,N6-3*num-1,num)
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    c_ue4 = columnnew(arr,N5-3*num,num)
    H1,r1 = con_edge(X,c_ue1,c_ue3,c_l1,c_ut1,num,N)
    H2,r2 = con_edge(X,c_ue2,c_ue4,c_l2,c_ut2,num,N)
    Hu1,ru1 = con_unit(X,c_ut1,num,N)
    Hu2,ru2 = con_unit(X,c_ut2,num,N)
    Ha,ra = con_constangle2(X,c_ut1,c_ut2,N6-1,num,N)
    H = sparse.vstack((H1,H2,Hu1,Hu2,Ha))
    r = np.r_[r1,r2,ru1,ru2,ra]
    if assign:
        H0,r0 = con_constl(np.array([N6-1],dtype=int),cos0,1,N)
        H = sparse.vstack((H, H0))
        r = np.r_[r,r0]
    #self.add_iterative_constraint(H*w, r*w, 'isogonal_diagnet')
    return H*w,r*w

def con_isogonal_checkerboard_based(cos0,assign=False,**kwargs):
    """
    quadface: diagonal crossing angle
    X += [ld1,ld2, ud1,ud2]
    1. (v1-v3) = ld1*ud1, ud1**2=1
    2. (v2-v4) = ld2*ud2, ud2**2=1
    3. ud1*ud2 == cos0
    """
    w = kwargs.get('isogonal_ck_based')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N10 = kwargs.get('N10')
    
    V = mesh.V
    num = mesh.num_quadface
    numl = N10-8*num-1
    numud = N10-6*num-1
    arr = np.arange(num)
    c_ld1 = numl+arr
    c_ld2 = numl+num+arr
    v1,v2,v3,v4 = mesh.rr_quadface.T # in odrder
    c_v1 = np.r_[v1,V+v1,2*V+v1] # [x,y,z]
    c_v2 = np.r_[v2,V+v2,2*V+v2] # [x,y,z]
    c_v3 = np.r_[v3,V+v3,2*V+v3] # [x,y,z]
    c_v4 = np.r_[v4,V+v4,2*V+v4] # [x,y,z]
    c_ud1 = np.r_[numud+arr,numud+num+arr,numud+2*num+arr]
    c_ud2 = c_ud1+3*num

    He1,re1 = con_edge(X,c_v1,c_v3,c_ld1,c_ud1,num,N)
    He2,re2 = con_edge(X,c_v2,c_v4,c_ld2,c_ud2,num,N)
    Hu1,ru1 = con_unit(X,c_ud1,num,N)
    Hu2,ru2 = con_unit(X,c_ud2,num,N)
    Ha,ra = con_constangle2(X,c_ud1,c_ud2,N10-1,num,N)

    H = sparse.vstack((He1,He2,Hu1,Hu2,Ha*10))
    r = np.r_[re1,re2,ru1,ru2,ra*10]
    if assign:
        H0,r0 = con_constl(np.array([N10-1],dtype=int),cos0,1,N)
        H = sparse.vstack((H, H0))
        r = np.r_[r,r0]
    
    #self.add_iterative_constraint(H*w, r*w, 'isogonal_ck_based')   
    return H*w,r*w

def con_isogonal_quadface_based(cos0,assign=False,halfdiag=True,**kwargs):
    """
    quadface: midedge point edge vectors
    X += [ld1,ld2, ud1,ud2]
    1. (v2+v3-v1-v4) = 2* ld1*ud1, ud1**2=1
    2. (v3+v4-v1-v2) = 2* ld2*ud2, ud2**2=1
    3. ud1*ud2 == cos0
    """
    w = kwargs.get('isogonal_face_based')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N10 = kwargs.get('N10')
    V = mesh.V

    if halfdiag:
        ib,ir = mesh.vertex_check_ind
        _,v1,v2,v3,v4 = mesh.rr_star.T
        v1,v2,v3,v4 = v1[ib],v2[ib],v3[ib],v4[ib]
        num = len(v1)
    else:
        num = mesh.num_quadface
        v1,v2,v3,v4 = mesh.rr_quadface.T # in odrder
        
    numl = N10-8*num-1
    numud = N10-6*num-1
    arr = np.arange(num)
    c_ld1 = numl+arr
    c_ld2 = numl+num+arr            
        
    c_v1 = np.r_[v1,V+v1,2*V+v1] # [x,y,z]
    c_v2 = np.r_[v2,V+v2,2*V+v2] # [x,y,z]
    c_v3 = np.r_[v3,V+v3,2*V+v3] # [x,y,z]
    c_v4 = np.r_[v4,V+v4,2*V+v4] # [x,y,z]
    c_ud1 = np.r_[numud+arr,numud+num+arr,numud+2*num+arr]
    c_ud2 = c_ud1+3*num

    def _edge(c_ld1,c_ud1,dddd):
        "(v2+v3-v1-v4) = 2* ld1*ud1, ud1**2=1"
        ld1 = X[c_ld1]
        ud1 = X[c_ud1]
        row = np.tile(np.arange(3*num),6)
        col = np.r_[c_v1,c_v2,c_v3,c_v4,np.tile(c_ld1,3),c_ud1]
        data = np.r_[dddd,-2*ud1,-2*np.tile(ld1,3)]
        r = -2*np.tile(ld1,3)*ud1
        H = sparse.coo_matrix((data,(row,col)), shape=(3*num, N))
        return H,r
    
    a3 = np.ones(3*num)
    d1 = np.r_[-a3,a3,a3,-a3]
    d2 = np.r_[-a3,-a3,a3,a3]
    He1,re1 = _edge(c_ld1,c_ud1,d1)
    He2,re2 = _edge(c_ld2,c_ud2,d2)
    Hu1,ru1 = con_unit(X,c_ud1,num,N)
    Hu2,ru2 = con_unit(X,c_ud2,num,N)
    Ha,ra = con_constangle2(X,c_ud1,c_ud2,N10-1,num,N)
    H = sparse.vstack((He1,He2,Hu1,Hu2,Ha))
    r = np.r_[re1,re2,ru1,ru2,ra]
    
    if assign:
        H0,r0 = con_constl(np.array([N10-1],dtype=int),cos0,1,N)
        H = sparse.vstack((H, H0))
        r = np.r_[r,r0]
        
    #self.add_iterative_constraint(H*w, r*w, 'isogonal_face_based')
    return H*w,r*w

def con_unequal_two_neighbouring_edges(v012,eps,**kwargs):
    """ oriented edge1,edge2 l1>=l2 <==> l1^2-l2^2*(1+eps)=s^2
                            (v1-v)^2-(v2-v)^2*(1+eps) = s^2
    """
    w = kwargs.get('nonsymmetric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nnonsym = kwargs.get('Nnonsym')
    num = len(v012[0])
    c_s = Nnonsym-num+np.arange(num)
    c_v  = columnnew(v012[0],0, mesh.V)
    c_v1 = columnnew(v012[1],0, mesh.V)
    c_v2 = columnnew(v012[2],0, mesh.V)
    col = np.r_[c_v,c_v1,c_v2,c_s]
    row = np.tile(np.arange(num),10)
    X0,X1,X2,Xs = X[c_v],X[c_v1],X[c_v2],X[c_s]
    data = np.r_[-2*(X1-X0)+2*(X2-X0)*(1+eps),2*(X1-X0),-2*(X2-X0)*(1+eps),-2*Xs]
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    E1,E2 = (X1-X0).reshape(-1,3,order='F'),(X2-X0).reshape(-1,3,order='F')
    r = np.linalg.norm(E1,axis=1)**2-np.linalg.norm(E2,axis=1)**2*(1+eps)
    r -= Xs**2
    return H*w,r*w

def con_nonsquare_quadface(v012,il12,eps,**kwargs):
    """ oriented edge1,edge2 l1 > l2 or l1<l2.
        <==> (l1-l2)^2 = s^2 + eps
            l1**2 = (v1-v0)^2; l2**2 = (v2-v0)^2
    v012 := [v0,v1,v2]
    il12 := [il1, il2]
    """
    w = kwargs.get('nonsymmetric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nnonsym = kwargs.get('Nnonsym')
    
    c_v  = columnnew(v012[0],0, mesh.V)
    c_v1 = columnnew(v012[1],0, mesh.V)
    c_v2 = columnnew(v012[2],0, mesh.V)
    num = len(il12[0])
    c_l1 = Nnonsym-mesh.E-num + il12[0]
    c_l2 = Nnonsym-mesh.E-num + il12[1]
    c_s = Nnonsym-num + np.arange(num)
    Xl1,Xl2,Xs = X[c_l1],X[c_l2],X[c_s]
    def _ratio():
        col = np.r_[c_l1,c_l2,c_s]
        row = np.tile(np.arange(num),3)
        data = np.r_[2*(Xl1-Xl2),-2*(Xl1-Xl2),-2*Xs]
        H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
        r = (Xl1-Xl2)**2-Xs**2 + np.ones(num)*eps
        return H,r
    def _edge(c_l1,c_v0,c_v1):
        "l1**2 = (v1-v0)^2"
        col = np.r_[c_v0,c_v1,c_l1]
        row = np.tile(np.arange(num),7)
        data = 2*np.r_[-X[c_v1]+X[c_v0],X[c_v1]-X[c_v0],-X[c_l1]]
        H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
        r = np.linalg.norm((X[c_v1]-X[c_v0]).reshape(-1,3,order='F'),axis=1)**2
        r -= X[c_l1]**2
        return H,r
    H1,r1 = _ratio()
    H2,r2 = _edge(c_l1,c_v,c_v1)
    H3,r3 = _edge(c_l2,c_v,c_v2)
    H = sparse.vstack((H1, H2, H3))
    r = np.r_[r1,r2,r3]
    return H*w,r*w

def con_ctrlnet_symmetric_1_diagpoly(another_poly_direction=False,**kwargs):
    """ ctrl-quadmesh + 1diagonal form a web:
        three families of polylines satisfy symmetric condtion:
        ut1,ut2 (unit tangnets of control polylines); ud1 (unit tangent of diagonal)
        ut1 and ut2 symmetric to ud1
        <==>
            ud1 * (ut1-ut2) = 0;
            (v1-v3) = l1 * ut1; (v2-v4) = l2 * ut2; (va-vc) = lac * ud1
            ut1^2=1; ut2^2=1; ut1^2=1;
    X = [lt1,lt2,ut1,ut2; lac,ud1]   ##len=1+1+3+3+1+3
    """
    w = kwargs.get('ctrlnet_symmetric_1diagpoly')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    num = len(mesh.ind_rr_star_v4f4)
    arr,arr3 = np.arange(num), np.arange(3*num)
    Ncds = kwargs.get('Ncds')-12*num
    c_lt1,c_t1 = Ncds+arr, Ncds+2*num+arr3
    c_lt2,c_t2 = c_lt1+num, c_t1+3*num
    c_ld1,c_d1 = Ncds+8*num+arr,Ncds+9*num+arr3
    
    _,v1,v2,v3,v4 = mesh.rr_star[mesh.ind_rr_star_v4f4].T
    _,va,vb,vc,vd = mesh.rr_star_corner# in diagonal direction
    c_v1 = columnnew(v1,0,mesh.V)
    c_v2 = columnnew(v2,0,mesh.V)
    c_v3 = columnnew(v3,0,mesh.V)
    c_v4 = columnnew(v4,0,mesh.V)
    if another_poly_direction:
        c_va = columnnew(vb,0,mesh.V)
        c_vc = columnnew(vd,0,mesh.V)
    else:
        c_va = columnnew(va,0,mesh.V)
        c_vc = columnnew(vc,0,mesh.V)
    H1,r1 = con_edge(X,c_v1,c_v3,c_lt1,c_t1,num,N)
    H2,r2 = con_edge(X,c_v2,c_v4,c_lt2,c_t2,num,N)
    H3,r3 = con_edge(X,c_va,c_vc,c_ld1,c_d1,num,N)
    
    Hu1,ru1 = con_unit(X,c_t1,num,N)
    Hu2,ru2 = con_unit(X,c_t2,num,N)
    Hu3,ru3 = con_unit(X,c_d1,num,N)
    Hs,rs = con_planarity(X,c_t1,c_t2,c_d1,num,N)
    H = sparse.vstack((H1, H2, H3, Hu1,Hu2,Hu3,Hs))
    r = np.r_[r1,r2,r3,ru1,ru2,ru3,rs]
    return H*w,r*w

def con_chebyshev(l0,assign=False,**kwargs):
    """
    keeping all edge_length equal
            (Vi-Vj)^2 = l^2
    if assign:
            l == l0
    """
    w = kwargs.get('chebyshev')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N8 = kwargs.get('N8')
    V = mesh.V
    vi, vj = mesh.vertex_ring_vertices_iterators(order=True)
    num = len(vi)
    numl = N8-1
    c_l = np.tile(numl, num)
    c_vi = columnnew(vi,0,V)
    c_vj = columnnew(vj,0,V)
    data1 = X[c_vi]
    data2 = X[c_vj]
    col = np.r_[c_vi, c_vj, c_l]
    data = 2*np.r_[data1-data2, data2-data1, -X[c_l]]
    row = np.tile(np.arange(num),7)
    r = np.einsum('ij,ij->i',(data1-data2).reshape(-1,3, order='F'),(data1-data2).reshape(-1,3, order='F')) - X[c_l]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    if assign:
        Hl,rl = con_constl(np.array([numl],dtype=int),np.array([l0]),1,N)
        H = sparse.vstack((H, Hl))
        r = np.r_[r,rl]
    return H*w, r*w
    #--------------------------------------------------------------------------
    #                       A-net:
    #-------------------------------------------------------------------------- 
def _con_anet(X,w,c_n,c_v,c_v1,c_v2,c_v3,c_v4,N):
    "vn*(vi-v)=0; vn**2=1"
    num = int(len(c_v)/3)
    H1,r1 = con_planarity(X,c_v,c_v1,c_n,num,N)
    H2,r2 = con_planarity(X,c_v,c_v2,c_n,num,N)
    H3,r3 = con_planarity(X,c_v,c_v3,c_n,num,N)
    H4,r4 = con_planarity(X,c_v,c_v4,c_n,num,N)
    Hn,rn = con_unit(X,c_n,num,N)
    H = sparse.vstack((H1,H2,H3,H4,Hn))
    r = np.r_[r1,r2,r3,r4,rn]
    return H*w, r*w
    
def con_anet(rregular=False,checker_weight=1,id_checker=None,pitch=1,**kwargs): #TODO
    """ based on con_unit_edge()
    X += [ni]
    ni * (vij - vi) = 0
    """
    w = kwargs.get('Anet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nanet = kwargs.get('Nanet')
    
    if rregular:
        v,v1,v2,v3,v4 = mesh.rr_star[mesh.ind_rr_star_v4f4].T
        num=len(mesh.ind_rr_star_v4f4)
    else:
        num = mesh.num_regular
        v,v1,v2,v3,v4 = mesh.ver_regular_star.T
        
    c_n = Nanet-3*num+np.arange(3*num)
    c_v  = columnnew(v ,0,mesh.V)
    c_v1 = columnnew(v1,0,mesh.V)
    c_v2 = columnnew(v2,0,mesh.V)
    c_v3 = columnnew(v3,0,mesh.V)
    c_v4 = columnnew(v4,0,mesh.V)
    
    if rregular and checker_weight<1:
        "at red-rr-vs, smaller weight"
        wr = checker_weight
        iblue,ired = id_checker
        ib = columnnew(iblue,0,len(mesh.ind_rr_star_v4f4))
        ir = columnnew(ired,0,len(mesh.ind_rr_star_v4f4))  
        Hb,rb = _con_anet(X,w,c_n[ib],c_v[ib],c_v1[ib],c_v2[ib],c_v3[ib],c_v4[ib],N)
        Hr,rr = _con_anet(X,wr,c_n[ir],c_v[ir],c_v1[ir],c_v2[ir],c_v3[ir],c_v4[ir],N)
        H = sparse.vstack((Hb,Hr))
        r = np.r_[rb,rr]  
    else:
        "all rr-vs, same weight"
        H,r = _con_anet(X,w,c_n,c_v,c_v1,c_v2,c_v3,c_v4,N)

    if kwargs.get('normal_bar'):
        Nbar = kwargs.get('Nbar')
        if pitch<0:
            c_nbar = Nbar-3*num+np.arange(3*num)-1
            annnbar = [c_v,c_n,c_nbar,Nbar-1]
        else:
            c_nbar = Nbar-3*num+np.arange(3*num)
            annnbar = [c_v,c_n,c_nbar]
        return H,r, annnbar
    return H,r
     
def con_anet_diagnet(checker_weight=1,id_checker=None,
                     assign_crpc_ratio=1,pitch=1,**kwargs):
    "based on con_unit_edge(diag=True); X += [ni]; ni * (vij - vi) = 0"
    w = kwargs.get('Anet_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nanet = kwargs.get('Nanet')
    
    #c_v,c_v1,c_v2,c_v3,c_v4 = mesh.get_vs_diagonal_v(index=False)
    v,v1,v2,v3,v4 = mesh.rr_star_corner
    c_v  = columnnew(v ,0,mesh.V)
    c_v1 = columnnew(v1,0,mesh.V)
    c_v2 = columnnew(v2,0,mesh.V)
    c_v3 = columnnew(v3,0,mesh.V)
    c_v4 = columnnew(v4,0,mesh.V)
    
    num = int(len(c_v)/3)
    c_n = Nanet-3*num+np.arange(3*num)

    if checker_weight<1:
        "at red-rr-vs, smaller weight"
        wr = checker_weight
        iblue,ired = id_checker
        ib = columnnew(iblue,0,len(mesh.ind_rr_star_v4f4))
        ir = columnnew(ired,0,len(mesh.ind_rr_star_v4f4))  
        Hb,rb = _con_anet(X,w,c_n[ib],c_v[ib],c_v1[ib],c_v2[ib],c_v3[ib],c_v4[ib],N)
        Hr,rr = _con_anet(X,wr,c_n[ir],c_v[ir],c_v1[ir],c_v2[ir],c_v3[ir],c_v4[ir],N)
        H = sparse.vstack((Hb,Hr))
        r = np.r_[rb,rr]  
    else:
        "all rr-vs, same weight"
        H,r = _con_anet(X,w,c_n,c_v,c_v1,c_v2,c_v3,c_v4,N)


    annnbar = None
    if kwargs.get('normal_bar'):
        N10 = kwargs.get('N10')
        Nbar = kwargs.get('Nbar')
        if pitch<0:
            c_nbar = Nbar-3*num+np.arange(3*num)-1
            annnbar = [c_v,c_n,c_nbar,Nbar-1]
        else:
            c_nbar = Nbar-3*num+np.arange(3*num)
            annnbar = [c_v,c_n,c_nbar]
        return H*w,r*w,annnbar
    if kwargs.get('CRPC'):
        """
        quadface: diagonal crossing angle
        no additional varibalse; related with e1,e2,given ratio a
        a family of constraints:
            (1-a) e1*e2 - a-1=0 <==> e1*e2 = (1+a) / (1-a) === cos0
        """   
        num = mesh.num_quadface
        numud = N10-6*num-1
        arr = np.arange(num)
        c_ud1 = np.r_[numud+arr,numud+num+arr,numud+2*num+arr]
        c_ud2 = c_ud1+3*num
        col = np.r_[c_ud1,c_ud2]
        row = np.tile(arr,6)
        data = np.r_[X[c_ud2],X[c_ud1]]
        rr = np.einsum('ij,ij->i',X[c_ud1].reshape(-1,3, order='F'),X[c_ud2].reshape(-1,3, order='F'))
        a = assign_crpc_ratio
        rr += np.ones(num)*(1+a)/(1-a)
        Hr = sparse.coo_matrix((data,(row,col)), shape=(num, N))
        H = sparse.vstack((H,Hr))
        r = np.r_[r,rr]
        return H*w,r*w,annnbar
    return H,r


    #--------------------------------------------------------------------------
    #                       S-net:
    #--------------------------------------------------------------------------  
def con_snet(orientrn,pitch=None,**kwargs):
    w = kwargs.get('Snet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nsnet = kwargs.get('Nsnet')
 
    V = mesh.V
    X = X
    numv = mesh.num_regular
    v0,v1,v2,v3,v4 = mesh.rr_star.T
    c_v0 = columnnew(v0,0,V)
    c_v1 = columnnew(v1,0,V)
    c_v2 = columnnew(v2,0,V)
    c_v3 = columnnew(v3,0,V)
    c_v4 = columnnew(v4,0,V)
    arr1 = np.arange(numv)
    arr3 = np.arange(3*numv)
    _n1 = Nsnet-11*numv
    c_squ, c_a = _n1+np.arange(5*numv),_n1+5*numv+arr1
    c_b,c_c,c_d,c_e = c_a+numv,c_a+2*numv,c_a+3*numv,c_a+4*numv
    c_a_sqr = c_a+5*numv

    def _con_v_square(c_squ):
        "[v;v1,v2,v3,v4]=[x,y,z], X[c_squ]=x^2+y^2+z^2"
        row_v = np.tile(arr1,3)
        row_1 = row_v+numv
        row_2 = row_v+2*numv
        row_3 = row_v+3*numv
        row_4 = row_v+4*numv
        row = np.r_[row_v,row_1,row_2,row_3,row_4,np.arange(5*numv)]
        col = np.r_[c_v0,c_v1,c_v2,c_v3,c_v4,c_squ]
        dv = 2*np.r_[X[c_v0]]
        d1 = 2*np.r_[X[c_v1]]
        d2 = 2*np.r_[X[c_v2]]
        d3 = 2*np.r_[X[c_v3]]
        d4 = 2*np.r_[X[c_v4]]
        data = np.r_[dv,d1,d2,d3,d4,-np.ones(5*numv)]
        H = sparse.coo_matrix((data,(row,col)), shape=(5*numv, N))
        def xyz(c_i):
            c_x = c_i[:numv]
            c_y = c_i[numv:2*numv]
            c_z = c_i[2*numv:]
            return np.r_[X[c_x]**2+X[c_y]**2+X[c_z]**2]
        r = np.r_[xyz(c_v0),xyz(c_v1),xyz(c_v2),xyz(c_v3),xyz(c_v4)]
        return H,r
    def _con_pos_a(c_a,c_a_sqr):
        "a>=0 <---> a_sqr^2 - a = 0"
        row = np.tile(arr1,2)
        col = np.r_[c_a_sqr, c_a]
        data = np.r_[2*X[c_a_sqr], -np.ones(numv)]
        r = X[c_a_sqr]**2
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    def _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e):
        """normalize the sphere equation,
        convinent for computing/represent distance\normals
        ||df|| = b^2+c^2+d^2-4ae=1
        """
        row = np.tile(arr1,5)
        col = np.r_[c_a,c_b,c_c,c_d,c_e]
        data = 2*np.r_[-2*X[c_e],X[c_b],X[c_c],X[c_d],-2*X[c_a]]
        r = X[c_b]**2+X[c_c]**2+X[c_d]**2-4*X[c_a]*X[c_e]+np.ones(numv)
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    def _con_sphere(c_squ,c_a,c_b,c_c,c_d,c_e):
        "a(x^2+y^2+z^2)+(bx+cy+dz)+e=0"
        row = np.tile(arr1,9)
        def __sphere(c_vi,c_sq):
            c_x = c_vi[:numv]
            c_y = c_vi[numv:2*numv]
            c_z = c_vi[2*numv:]
            col = np.r_[c_x,c_y,c_z,c_sq,c_a,c_b,c_c,c_d,c_e]
            data = np.r_[X[c_b],X[c_c],X[c_d],X[c_a],X[c_sq],X[c_x],X[c_y],X[c_z],np.ones(numv)]
            r = X[c_b]*X[c_x]+X[c_c]*X[c_y]+X[c_d]*X[c_z]+X[c_a]*X[c_sq]
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        H0,r0 = __sphere(c_v0,c_squ[:numv])
        H1,r1 = __sphere(c_v1,c_squ[numv:2*numv])
        H2,r2 = __sphere(c_v2,c_squ[2*numv:3*numv])
        H3,r3 = __sphere(c_v3,c_squ[3*numv:4*numv])
        H4,r4 = __sphere(c_v4,c_squ[4*numv:])
        H = sparse.vstack((H0,H1,H2,H3,H4))
        r = np.r_[r0,r1,r2,r3,r4]
        return H,r
    def _con_const_radius(c_a,c_r):
        "2*ai * r = 1 == df"
        c_rr = np.tile(c_r, numv)
        row = np.tile(arr1,2)
        col = np.r_[c_a, c_rr]
        data = np.r_[X[c_rr], X[c_a]]
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        r = X[c_rr] * X[c_a] + 0.5*np.ones(numv)
        return H,r
    def _con_anet(c_a):
        row = arr1
        col = c_a
        data = np.ones(numv)
        r = np.zeros(numv)
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    def _con_orient(c_n,c_o):
        "n0x*nx+n0y*ny+n0z*nz-x_orient^2 = 0"
        row = np.tile(arr1,4)
        col = np.r_[c_n, c_o]
        data = np.r_[orientrn.flatten('F'), -2*X[c_o]]
        r = -X[c_o]**2
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
        
    H0,r0 = _con_v_square(c_squ)
    H1,r1 = _con_pos_a(c_a,c_a_sqr)
    Hn,rn = _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e)
    Hs,rs = _con_sphere(c_squ,c_a,c_b,c_c,c_d,c_e)
    H = sparse.vstack((H0,H1,Hn,Hs))
    r = np.r_[r0,r1,rn,rs]
        
    if kwargs.get('Snet_orient'):
        w1 = kwargs.get('Snet_orient')
        Ns_n = kwargs.get('Ns_n')
        c_n = Ns_n-4*numv+arr3
        c_n_sqr = Ns_n-numv+arr1
        Ho,ro = _con_orient(c_n,c_n_sqr)
        H = sparse.vstack((H, Ho * w1))
        r = np.r_[r, ro * w1]
    if kwargs.get('Snet_constR'):
        w2 = kwargs.get('Snet_constR')
        Ns_r = kwargs.get('Ns_r')
        c_r = np.array([Ns_r-1],dtype=int)
        Hr,rr = _con_const_radius(c_a,c_r)
        H = sparse.vstack((H, Hr * w2))
        r = np.r_[r, rr * w2]
    if kwargs.get('Snet_anet'):
        w3 = kwargs.get('Snet_anet')
        Ha,ra = _con_anet(c_a)
        H = sparse.vstack((H, Ha * w3))
        r = np.r_[r, ra * w3]

    if kwargs.get('normal_bar'):
        """cen-an=n*r; (n^2=1, not necessary)
        cen = -(B,C,D)/2A, r is computed from last iteration
        2A*(r* nx + anx) + B = 0
        2A*(r* ny + any) + C = 0
        2A*(r* nz + anz) + D = 0
        """
        Nbar = kwargs.get('Nbar')
        annnbar = None
        if pitch<0:
            c_n = Nbar-6*numv+np.arange(3*numv)-1
            c_nbar = Nbar-3*numv+np.arange(3*numv)-1
            annnbar = [c_v0,c_n,c_nbar,Nbar-1]
        else:
            c_n = Nbar-6*numv+np.arange(3*numv)
            c_nbar = Nbar-3*numv+np.arange(3*numv)
            annnbar = [c_v0,c_n,c_nbar]
        
        cen = -np.c_[X[c_b]/X[c_a],X[c_c]/X[c_a],X[c_d]/X[c_a]]/2
        rad1 = np.linalg.norm(cen-X[c_v0].reshape(-1,3,order='F'),axis=1)
        rad2 = np.linalg.norm(cen-X[c_v1].reshape(-1,3,order='F'),axis=1)
        rad3 = np.linalg.norm(cen-X[c_v2].reshape(-1,3,order='F'),axis=1)
        rad4 = np.linalg.norm(cen-X[c_v3].reshape(-1,3,order='F'),axis=1)
        rad5 = np.linalg.norm(cen-X[c_v4].reshape(-1,3,order='F'),axis=1)
        radii = (rad1+rad2+rad3+rad4+rad5)/5
        
        def _normal(c_a,c_b,c_anx,c_nx):
            row = np.tile(np.arange(numv),4)
            col = np.r_[c_a,c_b,c_anx,c_nx]
            one = np.ones(numv)
            data = np.r_[2*(radii*X[c_nx]+X[c_anx]),one,2*X[c_a],2*radii*X[c_a]]
            r = 2*X[c_a]*(radii*X[c_nx]+X[c_anx])
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        Hb,rb = _normal(c_a,c_b,c_v0[:numv],c_n[:numv])
        Hc,rc = _normal(c_a,c_c,c_v0[numv:2*numv],c_n[numv:2*numv])
        Hd,rd = _normal(c_a,c_d,c_v0[2*numv:],c_n[2*numv:])
        Hn,rn = con_unit(X,c_n,numv,N)
        H = sparse.vstack((H, Hb, Hc, Hd, Hn))
        r = np.r_[r, rb, rc, rd, rn]
        return H*w,r*w,annnbar
    #self.add_iterative_constraint(H * w, r * w, 'Snet')
    return H*w,r*w

def con_snet_diagnet(assign_crpc_ratio,pitch=None,
                     ck1=False,ck2=False,is_sub=True,
                     **kwargs):
    w = kwargs.get('Snet_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nsnet = kwargs.get('Nsnet')
    
    X = X
    numv = len(mesh.ind_rr_star_v4f4)
    
    if ck1:
        numv = len(mesh.ind_ck_rr_vertex[0])
    elif ck2:
        numv = len(mesh.ind_ck_rr_vertex[1])
        
    arrv1 = np.arange(numv)

    c_v,c_cen1,c_cen2,c_cen3,c_cen4 = mesh.get_vs_diagonal_v(ck1=ck1,ck2=ck2,index=False)
    c_cen = [c_cen1,c_cen2,c_cen3,c_cen4]
    _n1 = Nsnet-11*numv
    c_squ, c_a = _n1+np.arange(5*numv),_n1+5*numv+arrv1
    c_b,c_c,c_d,c_e = c_a+numv,c_a+2*numv,c_a+3*numv,c_a+4*numv
    c_a_sqr = c_a+5*numv

    def _con_v_square(c_v,c_cen,c_squ):
        "[v;c1,c2,c3,c4]=[x,y,z], X[c_squ]=x^2+y^2+z^2"
        c_cen1,c_cen2,c_cen3,c_cen4 = c_cen
        row_v = np.tile(arrv1,3)
        row_1 = row_v+numv
        row_2 = row_v+2*numv
        row_3 = row_v+3*numv
        row_4 = row_v+4*numv
        row = np.r_[row_v,row_1,row_2,row_3,row_4,np.arange(5*numv)]
        col = np.r_[c_v,c_cen1,c_cen2,c_cen3,c_cen4,c_squ]
        dv = 2*np.r_[X[c_v]]
        d1 = 2*np.r_[X[c_cen1]]
        d2 = 2*np.r_[X[c_cen2]]
        d3 = 2*np.r_[X[c_cen3]]
        d4 = 2*np.r_[X[c_cen4]]
        data = np.r_[dv,d1,d2,d3,d4,-np.ones(5*numv)]
        H = sparse.coo_matrix((data,(row,col)), shape=(5*numv, N))
        def xyz(c_i):
            c_x = c_i[:numv]
            c_y = c_i[numv:2*numv]
            c_z = c_i[2*numv:]
            return np.r_[X[c_x]**2+X[c_y]**2+X[c_z]**2]
        r = np.r_[xyz(c_v),xyz(c_cen1),xyz(c_cen2),xyz(c_cen3),xyz(c_cen4)]
        return H,r
    def _con_pos_a(c_a,c_a_sqr):
        "a>=0 <---> a_sqr^2 - a = 0"
        row = np.tile(arrv1,2)
        col = np.r_[c_a_sqr, c_a]
        data = np.r_[2*X[c_a_sqr], -np.ones(numv)]
        r = X[c_a_sqr]**2
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    def _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e):
        """normalize the sphere equation,
        convinent for computing/represent distance\normals
        ||df|| = b^2+c^2+d^2-4ae=1
        """
        row = np.tile(arrv1,5)
        col = np.r_[c_a,c_b,c_c,c_d,c_e]
        data = 2*np.r_[-2*X[c_e],X[c_b],X[c_c],X[c_d],-2*X[c_a]]
        r = X[c_b]**2+X[c_c]**2+X[c_d]**2-4*X[c_a]*X[c_e]+np.ones(numv)
        H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
        return H,r
    def _con_sphere(c_v,c_cen,c_squ,c_a,c_b,c_c,c_d,c_e):
        "a(x^2+y^2+z^2)+(bx+cy+dz)+e=0"
        c_cen1,c_cen2,c_cen3,c_cen4 = c_cen
        row = np.tile(arrv1,9)
        def __sphere(c_vi,c_sq):
            c_x = c_vi[:numv]
            c_y = c_vi[numv:2*numv]
            c_z = c_vi[2*numv:]
            col = np.r_[c_x,c_y,c_z,c_sq,c_a,c_b,c_c,c_d,c_e]
            data = np.r_[X[c_b],X[c_c],X[c_d],X[c_a],X[c_sq],X[c_x],X[c_y],X[c_z],np.ones(numv)]
            r = X[c_b]*X[c_x]+X[c_c]*X[c_y]+X[c_d]*X[c_z]+X[c_a]*X[c_sq]
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        H0,r0 = __sphere(c_v,c_squ[:numv])
        H1,r1 = __sphere(c_cen1,c_squ[numv:2*numv])
        H2,r2 = __sphere(c_cen2,c_squ[2*numv:3*numv])
        H3,r3 = __sphere(c_cen3,c_squ[3*numv:4*numv])
        H4,r4 = __sphere(c_cen4,c_squ[4*numv:])
        H = sparse.vstack((H0,H1,H2,H3,H4))
        r = np.r_[r0,r1,r2,r3,r4]
        return H,r   
    
    H0,r0 = _con_v_square(c_v,c_cen,c_squ)
    H1,r1 = _con_pos_a(c_a,c_a_sqr)
    Hn,rn = _con_sphere_normalization(c_a,c_b,c_c,c_d,c_e)
    Hs,rs = _con_sphere(c_v,c_cen,c_squ,c_a,c_b,c_c,c_d,c_e)
    H = sparse.vstack((H0,H1,Hn,Hs))
    r = np.r_[r0,r1,rn,rs]

    def _con_normal(c_n):
        cen = -np.c_[X[c_b]/X[c_a],X[c_c]/X[c_a],X[c_d]/X[c_a]]/2
        rad1 = np.linalg.norm(cen-X[c_v].reshape(-1,3,order='F'),axis=1)
        rad2 = np.linalg.norm(cen-X[c_cen1].reshape(-1,3,order='F'),axis=1)
        rad3 = np.linalg.norm(cen-X[c_cen2].reshape(-1,3,order='F'),axis=1)
        rad4 = np.linalg.norm(cen-X[c_cen3].reshape(-1,3,order='F'),axis=1)
        rad5 = np.linalg.norm(cen-X[c_cen4].reshape(-1,3,order='F'),axis=1)
        radii = (rad1+rad2+rad3+rad4+rad5)/5
        
        def _normal(c_a,c_b,c_anx,c_nx):
            row = np.tile(np.arange(numv),4)
            col = np.r_[c_a,c_b,c_anx,c_nx]
            one = np.ones(numv)
            data = np.r_[2*(radii*X[c_nx]+X[c_anx]),one,2*X[c_a],2*radii*X[c_a]]
            r = 2*X[c_a]*(radii*X[c_nx]+X[c_anx])
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, N))
            return H,r
        Hb,rb = _normal(c_a,c_b,c_v[:numv],c_n[:numv])
        Hc,rc = _normal(c_a,c_c,c_v[numv:2*numv],c_n[numv:2*numv])
        Hd,rd = _normal(c_a,c_d,c_v[2*numv:],c_n[2*numv:])
        Hn,rn = con_unit(X,c_n,numv,N)
        H = sparse.vstack((Hb, Hc, Hd, Hn))
        r = np.r_[rb, rc, rd, rn]  
        return H,r
        
    if kwargs.get('normal_bar'):
        Nbar = kwargs.get('Nbar')
        annnbar = None
        if pitch<0:
            c_n = Nbar-6*numv+np.arange(3*numv)-1 # unit n
            c_nbar = Nbar-3*numv+np.arange(3*numv)-1 # n_bar
            annnbar = [c_v,c_n,c_nbar,Nbar-1]
        else:
            c_n = Nbar-6*numv+np.arange(3*numv) # unit n
            c_nbar = Nbar-3*numv+np.arange(3*numv) # n_bar
            annnbar = [c_v,c_n,c_nbar]
        Hn,rn = _con_normal(c_n)
        H = sparse.vstack((H, Hn))
        r = np.r_[r, rn]        
        return H*w,r*w,annnbar
    
    if kwargs.get('snet_geodesic'):
        Nbar = kwargs.get('Nbar')
        Ns_bi = kwargs.get('Ns_bi')
        if kwargs.get('normal_bar'):
            "note: here has already include Hn,rn, below add twice"
            c_n = Nbar-6*numv+np.arange(3*numv) # unit n
            c_bi1 = Ns_bi-6*numv+np.arange(3*numv)
            c_bi2 = c_bi1+3*numv
        else:
            c_n = Ns_bi-9*numv+np.arange(3*numv)
            c_bi1 = c_n+3*numv
            c_bi2 = c_bi1+3*numv
        H1,r1 = con_cross_product2(X,c_v,c_cen1,c_cen3,c_bi1,N)
        H2,r2 = con_cross_product2(X,c_v,c_cen2,c_cen4,c_bi2,N)
        H3,r3 = con_dot(X,c_bi1,c_n,N)
        H4,r4 = con_dot(X,c_bi2,c_n,N)
        Hn,rn = _con_normal(c_n)
        H = sparse.vstack((H,H1,H2,H3,H4,Hn))
        r = np.r_[r,r1,r2,r3,r4,rn]  
        
    if kwargs.get('Snet_gi_t'): # no use now!!!
        """
        quadface: diagonal crossing angle
        X += [ld1,ld2, ud1,ud2, cos00] -- GI-net tangent
        1. (v1-v3) = ld1*ud1, ud1**2=1
        2. (v2-v4) = ld2*ud2, ud2**2=1
        3. ud1*ud2 == cos00
        4. (a^2-1)*A*B-(1+a^2)A+(1+a^2)B+1-a^2=0 (ti, gi-ti, given a)
            A:=cos0; B:=cos00
        """           
        N10 = kwargs.get('N10')
        Ns_n = kwargs.get('Ns_n')
        Ns_git = kwargs.get('Ns_git')

        V = mesh.V
        X = X
        v1,v2,v3,v4 = mesh.rr_quadface.T # in odrder
        if is_sub:
            "normal from rr-vertex, tangent from inner-quadface" 
            inn,_ = mesh.get_rr_quadface_boundaryquad_index()
            v1,v2,v3,v4 = v1[inn],v2[inn],v3[inn],v4[inn]
        num = len(v1)
        numl = Ns_git-8*num-1
        numud = Ns_git-6*num-1
        numn = Ns_n - 3*V
        arr = np.arange(num)
        c_ld1 = numl+arr
        c_ld2 = numl+num+arr
        
        "HERE VERTEX FROM UNIT-NORMAL"
        #c_alln = numn + np.arange(3*V)
        #X[c_alln] = -mesh.vertex_normals().flatten('F')
        subn = mesh.rr_star_corner[0]
        c_subn = columnnew(subn,numn,V)
        Hn,rn = _con_normal(c_subn)
 
        expn = np.setdiff1d(np.arange(V), subn)
        c_expn = columnnew(expn,numn,V)
        X[c_expn] = -mesh.vertex_normals()[expn].flatten('F')

        c_v1 = numn+np.r_[v1,V+v1,2*V+v1] # [x,y,z]
        c_v2 = numn+np.r_[v2,V+v2,2*V+v2] # [x,y,z]
        c_v3 = numn+np.r_[v3,V+v3,2*V+v3] # [x,y,z]
        c_v4 = numn+np.r_[v4,V+v4,2*V+v4] # [x,y,z]
        c_ud1 = np.r_[numud+arr,numud+num+arr,numud+2*num+arr]
        c_ud2 = c_ud1+3*num
        
        He1,re1 = con_edge(X,c_v1,c_v3,c_ld1,c_ud1,num,N)
        He2,re2 = con_edge(X,c_v2,c_v4,c_ld2,c_ud2,num,N)
        Hu1,ru1 = con_unit(X,c_ud1,num,N)
        Hu2,ru2 = con_unit(X,c_ud2,num,N)
        Ha,ra = con_constangle2(X,c_ud1,c_ud2,Ns_git-1,num,N)

        if True:
            "1 eq.: (a^2-1)*A*B-(1+a^2)A+(1+a^2)B+1-a^2=0"
            a = assign_crpc_ratio
            c_cos0,c_cos00 = N10-1, Ns_git-1
            col = np.array([c_cos0,c_cos00],dtype=int)
            row = np.zeros(2)
            d1,d2 = (a**2-1)*X[c_cos00]-a**2-1, (a**2-1)*X[c_cos0]+a**2+1
            data = np.array([d1,d2])
            rpc = np.array([(a**2-1)*X[c_cos0]*X[c_cos00]+a**2-1])
            Hpc = sparse.coo_matrix((data,(row,col)), shape=(1, N))
            
        H = sparse.vstack((H,Hn,He1,He2,Hu1,Hu2,Ha,Hpc))
        r = np.r_[r,rn,re1,re2,ru1,ru2,ra,rpc]
        # if assign:
        #     "maybe not useful"
        #     H0,r0 = con_constl(np.array([Ns_git-1],dtype=int),cos00,1,N)
        #     H = sparse.vstack((H, H0))
        #     r = np.r_[r,r0]
     #print('n:', np.sum(np.square((Hn*X)-rn)))
        # print('e1:', np.sum(np.square((He1*X)-re1)))
        # print('u1:', np.sum(np.square((Hu1*X)-ru1)))
        # print('a:', np.sum(np.square((Ha*X)-ra)))
        # print('pc:', np.sum(np.square((Hpc*X)-rpc)))
     #print('all:', np.sum(np.square((H*X)-r)))

    if kwargs.get('CRPC'):
        """
        quadface: diagonal crossing angle
        no additional varibalse; related with e1,e2,given ratio a
        a family of constraints:
            (1+a) e1*e2 + a-1=0 <==> e1*e2 = (1-a) / (1+a) === cos0
        """   
        num = mesh.num_quadface
        numud = N10-6*num-1
        arr = np.arange(num)
        c_ud1 = np.r_[numud+arr,numud+num+arr,numud+2*num+arr]
        c_ud2 = c_ud1+3*num
        col = np.r_[c_ud1,c_ud2]
        row = np.tile(arr,6)
        data = np.r_[X[c_ud2],X[c_ud1]]
        rr = np.einsum('ij,ij->i',X[c_ud1].reshape(-1,3, order='F'),X[c_ud2].reshape(-1,3, order='F'))
        a = assign_crpc_ratio
        rr += np.ones(num)*(1-a)/(1+a)
        Hr = sparse.coo_matrix((data,(row,col)), shape=(num, N))
        H = sparse.vstack((H,Hr))
        r = np.r_[r,rr]
   
    #self.add_iterative_constraint(H * w, r * w, 'Snet_diagnet')   
    return H*w,r*w

    #--------------------------------------------------------------------------
    #                       G-net:
    #--------------------------------------------------------------------------   

def con_1geodesic(polyline_direction=False,**kwargs): 
    """ still depends on the angle condition at vertex-star
    default direction: e1*e2-e3*e4=0;
    """
    w = kwargs.get('Geodesic')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    
    num = mesh.num_regular
    arr = np.arange(num)
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    c_ue4 = columnnew(arr,N5-3*num,num)      
    if polyline_direction:
        H,r = con_equal_opposite_angle(X,c_ue2,c_ue3,c_ue4,c_ue1,num,N)
    else:
        H,r = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue3,c_ue4,num,N)
    return H*w,r*w

def _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4,N):
    num = int(len(c_ue1)/3)
    H1,r1 = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue3,c_ue4,num,N)
    H2,r2 = con_equal_opposite_angle(X,c_ue2,c_ue3,c_ue4,c_ue1,num,N)
    H, r = sparse.vstack((H1, H2)), np.r_[r1,r2]
    return H*w, r*w

def con_gnet(rregular=False,checker_weight=1,id_checker=None,**kwargs):
    """
    based on con_unit_edge(diag=False)
    e1*e2-e3*e4=0; e2*e3-e1*e4=0
    """
    w = kwargs.get('Gnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    
    if rregular:
        "function same as below:con_gnet_diagnet"
        num=len(mesh.ind_rr_star_v4f4)
    else:
        num = mesh.num_regular
    arr = np.arange(num)
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    c_ue4 = columnnew(arr,N5-3*num,num)     
    
    if rregular and checker_weight<1:
        "at red-rr-vs, smaller weight"
        wr = checker_weight
        iblue,ired = id_checker
        ib = columnnew(iblue,0,len(mesh.ind_rr_star_v4f4))
        ir = columnnew(ired,0,len(mesh.ind_rr_star_v4f4))  
        Hb,rb = _con_gnet(X,w,c_ue1[ib],c_ue2[ib],c_ue3[ib],c_ue4[ib],N)
        Hr,rr = _con_gnet(X,wr,c_ue1[ir],c_ue2[ir],c_ue3[ir],c_ue4[ir],N)
        H = sparse.vstack((Hb,Hr))
        r = np.r_[rb,rr]  
    else:
        "all rr-vs, same weight"
        H,r = _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4,N)
    
    return H,r

def con_gnet_diagnet(checker_weight=1,id_checker=None,**kwargs):
    """
    based on con_unit_edge(diag=True)
    e1*e2-e3*e4=0; e2*e3-e1*e4=0
    """
    w = kwargs.get('Gnet_diagnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    
    num = len(mesh.ind_rr_star_v4f4)
    arr = np.arange(num)
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    c_ue4 = columnnew(arr,N5-3*num,num)        
    
    if checker_weight<1:
        "at red-rr-vs, smaller weight"
        wr = checker_weight
        iblue,ired = id_checker
        ib = columnnew(iblue,0,len(mesh.ind_rr_star_v4f4))
        ir = columnnew(ired,0,len(mesh.ind_rr_star_v4f4))  
        Hb,rb = _con_gnet(X,w,c_ue1[ib],c_ue2[ib],c_ue3[ib],c_ue4[ib],N)
        Hr,rr = _con_gnet(X,wr,c_ue1[ir],c_ue2[ir],c_ue3[ir],c_ue4[ir],N)
        H = sparse.vstack((Hb,Hr))
        r = np.r_[rb,rr]  
    else:
        "all rr-vs, same weight"
        H,r = _con_gnet(X,w,c_ue1,c_ue2,c_ue3,c_ue4,N)
    
    return H,r
   
def con_dog(rregular=False,**kwargs):
    """
    based on con_unit_edge() & con_gnet()
    e1*e2-e2*e3=0
    """
    w = kwargs.get('DOG')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    
    if rregular:
        num=len(mesh.ind_rr_star_v4f4)
    else:
        num = mesh.num_regular
    arr = np.arange(num)
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    #c_ue4 = columnnew(arr,N5-3*num,num)        
    H,r = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue2,c_ue3,num,N)
    return H*w,r*w

def con_gonet(rregular=False,is_direction24=False,**kwargs): 
    """ GEODESIC PARALLEL COORDINATES
    based on con_unit_edge() & con_1geodesic
    orthogonal: (e1-e3)*(e2-e4) = 0
    if direction: 
        geodesic: e1*e2-e1*e4=0;  e2*e3-e3*e4=0; 
    else:
        geodesic: e1*e2-e2*e3=0;  e3*e4-e4*e1=0;
    """
    w = kwargs.get('GOnet')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    N5 = kwargs.get('N5')
    
    if rregular:
        num=len(mesh.ind_rr_star_v4f4)
    else:
        num = mesh.num_regular
        
    arr = np.arange(num)
    c_ue1 = columnnew(arr,N5-12*num,num)
    c_ue2 = columnnew(arr,N5-9*num,num)
    c_ue3 = columnnew(arr,N5-6*num,num)
    c_ue4 = columnnew(arr,N5-3*num,num) 
  
    if is_direction24:
        H1,r1 = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue2,c_ue3,num,N)
        H2,r2 = con_equal_opposite_angle(X,c_ue3,c_ue4,c_ue4,c_ue1,num,N)
    else:
        H1,r1 = con_equal_opposite_angle(X,c_ue1,c_ue2,c_ue1,c_ue4,num,N)
        H2,r2 = con_equal_opposite_angle(X,c_ue2,c_ue3,c_ue3,c_ue4,num,N)
                
    row = np.tile(arr,12)
    col = np.r_[c_ue1,c_ue2,c_ue3,c_ue4]
    data = np.r_[X[c_ue2]-X[c_ue4],X[c_ue1]-X[c_ue3],X[c_ue4]-X[c_ue2],X[c_ue3]-X[c_ue1]]
    H3 = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    r3 = np.einsum('ij,ij->i',(X[c_ue1]-X[c_ue3]).reshape(-1,3, order='F'),(X[c_ue2]-X[c_ue4]).reshape(-1,3, order='F'))
    H = sparse.vstack((H1, H2, H3))
    r = np.r_[r1, r2, r3]        
    #print('err:gonet:',np.sum(np.square(H*X-r)))
    return H*w,r*w

def con_Voss(**kwargs):
    "conjugate geodesic net: planar quads with equal opposite angles"
    H1,r1 = con_normal_constraints(**kwargs)
    H2,r2 = con_planarity_constraints(**kwargs)
    H3,r3 = con_gnet(**kwargs)
    H = sparse.vstack((H1,H2,H3))
    r = np.r_[r1,r2,r3]
    return H,r

    #--------------------------------------------------------------------------
    #                       DGPC:
    #--------------------------------------------------------------------------  
def con_dgpc(rregular=False,polyline_direction=False,**kwargs):
    """main difference here is using patch_matrix to represent all vertices
    based on con_unit_edge() & con_gonet
    equal parallel_circle_direction edges
    each row: (vi-vj)^2 - lij^2 = 0
    """    
    w = kwargs.get('DGPC')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Ndgpc = kwargs.get('Ndgpc')
    
    rm = mesh.patch_matrix
    if polyline_direction:
        rm = rm.T
    nrow,ncol = rm.shape    
    vi,vj = rm[:,:-1].flatten(), rm[:,1:].flatten()
    c_vi = columnnew(vi ,0,mesh.V)
    c_vj = columnnew(vj ,0,mesh.V)
    c_l = (Ndgpc-nrow+np.arange(nrow)).repeat(ncol-1)
    H,r = con_diagonal(X,c_vi,c_vj,c_l,nrow*(ncol-1),N)
    return H*w,r*w

    #--------------------------------------------------------------------------
    #                       AAG / GGA-net:
    #--------------------------------------------------------------------------  
def _con_agnet_liouville(c_geo,num,angle=90,is_angle=False,**kwargs):
    """ X +=[ll1,ll2,ll3,ll4,u1,u2; lu1,tu1]
          +=[lu2,tu2; lla,llc,g1, lg1,tg1, c] 
    orthgonal A-net & constant angle with diagonal crv.
    2asymptotics: v1 -- v -- v3   &   v2 -- v -- v4

    orthogonal: u1 = l1**2*(V3-V0) - l3**2*(V1-V0)
                u2 = l2**2*(V4-V0) - l4**2*(V2-V0)
                u1 * v1 = 0 (no need to be unit)

    1geodesic:   a --- v --- c
                g1 = la**2*(Vc-V0) - lc**2*(Va-V0)   
                
    const.angle: u1 = tu1 * lu1
                 g1 = tg1 * lg1
                 tu1 * tg1 = const.       
    """
    X = kwargs.get('X')
    N = kwargs.get('N')
    Noscut = kwargs.get('Noscut')
    
    c_v,c_v1,c_v3,c_lu1,c_tu1,c_lla,c_llc,c_lg,c_g1,c_lg1,c_tg1,c_c = c_geo
    H2,r2 = con_osculating_tangent(X,c_v,c_v1,c_v3,c_lla,c_llc,c_lg,c_g1,num,N)
    "Unit 1asym tangent vector u1 = tu1 * lu1 :"
    c_u1 = Noscut-10*num+4*num+np.arange(3*num)
    H3,r3 = con_unit_vector(X,c_u1,c_tu1,c_lu1,num,N)
    "Unit 1geo tangent vector g1 = tg1 * lg1 :"
    H4,r4 = con_unit_vector(X,c_g1,c_tg1,c_lg1,num,N)
    "Constant angle with 1geo and 1asym crv.: "
    H5,r5 = con_constangle2(X,c_tu1,c_tg1,c_c,num,N) 
    H = sparse.vstack((H2,H3,H4,H5))
    r = np.r_[r2,r3,r4,r5]
    if is_angle:
        cos0 = np.cos(angle/180.0*np.pi)
        H0,r0 = con_constl(np.array([c_c],dtype=int),cos0,1,N)
        Ha,ra = con_constangle(X,c_tu1,c_tg1,cos0,num,N)
        H = sparse.vstack((H, H0, Ha))
        r = np.r_[r,r0,ra]
    return H,r

def _con_agnet_planar_geodesic(ver_poly_strip,strong=False,**kwargs):
    """ X +=[ni]
    along each i-th geodesic: ni * (vij-vik) = 0; k=j+1,j=0,...
    refer: self.get_poly_strip_normal()
    if strong:
        ni * anet_n = 0
    """
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Ndgeo = kwargs.get('Ndgeo')
    Ndgeopc = kwargs.get('Ndgeopc')

    iall,iind = ver_poly_strip
    num = len(iall)
    arr = Ndgeopc-3*num+np.arange(3*num)
    c_nx,c_ny,c_nz = arr[:num],arr[num:2*num],arr[2*num:3*num]

    col=row=data=r = np.array([])
    k,i = 0,0
    for iv in iall:
        va,vb = iv[:-1],iv[1:]
        m = len(va)
        c_a = columnnew(va,0,mesh.V)
        c_b = columnnew(vb,0,mesh.V)
        c_ni = np.r_[np.tile(c_nx[i],m),np.tile(c_ny[i],m),np.tile(c_nz[i],m)]
        coli = np.r_[c_a,c_b,c_ni]
        rowi = np.tile(np.arange(m),9) + k
        datai = np.r_[X[c_ni],-X[c_ni],X[c_a]-X[c_b]]
        ri = np.einsum('ij,ij->i',X[c_ni].reshape(-1,3,order='F'),(X[c_a]-X[c_b]).reshape(-1,3,order='F'))
        col = np.r_[col,coli]
        row = np.r_[row,rowi]
        data = np.r_[data,datai]
        r = np.r_[r,ri]
        k += m
        i += 1
    H = sparse.coo_matrix((data,(row,col)), shape=(k, N))
    H1,r1 = con_unit(X,arr,num,N)
    H = sparse.vstack((H,H1))
    r = np.r_[r,r1]
    
    if strong:
        "planar_geodesic = PC crv. if strong: normal_planar=PQ"
        num = len(mesh.ind_rr_star_v4f4)
        move = Ndgeo-6*num 
        col=row=data= np.array([])
        k,i = 0,0
        for iv in iind:
            iv=np.array(iv)
            m = len(iv)
            c_an = move+np.r_[iv,iv+num,iv+2*num]
            c_ni = np.r_[np.tile(c_nx[i],m),np.tile(c_ny[i],m),np.tile(c_nz[i],m)]
            coli = np.r_[c_an,c_ni]
            rowi = np.tile(np.arange(m),6) + k
            datai = np.r_[X[c_ni],X[c_an]]
            ri = np.einsum('ij,ij->i',X[c_ni].reshape(-1,3,order='F'),X[c_an].reshape(-1,3,order='F'))
            col = np.r_[col,coli]
            row = np.r_[row,rowi]
            data = np.r_[data,datai]
            r = np.r_[r,ri]
            k += m
            i += 1
        H0 = sparse.coo_matrix((data,(row,col)), shape=(k, N))  
        H = sparse.vstack((H,H0))
    return H,r

def con_anet_geodesic(ver_poly_strip,another_poly_direction=False,
                      checker_weight=1,id_checker=None,
                      **kwargs):
    """Anet(Gnet) with diagonal geodesic/asymptotic project:
                        d   4   c
                        1   v   3
                        a   2   b          
    if AAG:
        control net (v,1,2,3,4) is Anet, (a-v-c or b-v-d) is geodesic
    elif GAA:
        diagonal net (v,a,b,c,d) is Anet, (1-v-3 or 2-v-4) is geodesic
    elif GGA:
        control net (v,1,2,3,4) is Gnet, (a-v-c or b-v-d) is asymptotic
    elif AGG:
        diagonal net (v,a,b,c,d) is Gnet, (1-v-3 or 2-v-4) is asymptotic
 
    if AAG/GAA:
        X += [ni]+[Ni]; 
            ni: vertex-normal from Anet;
            Ni: osculating normal of geodesic
        <==> from Anet/Anet_diagnet: ni*(vi-v)=0,(i=1,2,3,4), ni^2=1;
             *geodesic: Ni=(Vc-V) x (Va-V); ni * Ni = 0
    elif GGA/AGG:
        X += [Ni,No1,No2]; 
            Ni: vertex-normal of Gnet;
            No1,No2: two oscualting normals of G-net
        <==> Way1 (guess has problem):
                Gnet/Gnet_diagnet: li*ei=vi-v,ei^2=1 (i=1,2,3,4);
                bisecting: ni*(e1-e3)= ni*(e2-e4)=0; ni^2=1; 
                asymptotic: ni*(va-v)=ni*(vc-v)=0
            *Way2 (using this):
                *Ni^2=1; Ni*No1,No2,va-v,vc-v=0;
                *No1=(vc-v) x (va-v), No2=(vd-v) x (vb-v)
    elif AAGG/GGAA:
        X += [ni] + [No1,No2];
            ni: vertex-normal from Anet;
            No1,No2: two oscualting normals of G-net
        <==> *No1=(vc-v) x (va-v), No2=(vd-v) x (vb-v); ni*No1,No2=0

    # from constraints_net import con_unit_edge, con_anet,con_anet_diagnet,
    # con_gnet,con_gnet_diagnet

    checker constraints: 
        blue for hard constraint; 
        red for soft..with lower checker_weight   
    id_checker=[iblue,ired]; len(id_checker)==len(ind_rr_star_v4f4)
    rr_star[ind_rr_star_v4f4][id_checker[0]] =\in= vblue
    rr_star[ind_rr_star_v4f4][id_checker[1]] =\in= vred
    """
    w_aag = kwargs.get('AAGnet')
    w_gaa = kwargs.get('GAAnet')
    w_gga = kwargs.get('GGAnet')
    w_agg = kwargs.get('AGGnet')
    w_aagg = kwargs.get('AAGGnet')
    w_ggaa = kwargs.get('GGAAnet')
    
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nanet = kwargs.get('Nanet') # for AAG, AAGG
    Ndgeo = kwargs.get('Ndgeo')

    num=len(mesh.ind_rr_star_v4f4)
    arr3 = np.arange(3*num)
    
    if id_checker is not None:
        "if checker_weight<1, checker_weight for red-vertex; w1,..,w6 for blue"
        iblue,ired = id_checker
        ib = columnnew(iblue,0,len(mesh.ind_rr_star_v4f4))
        ir = columnnew(ired,0,len(mesh.ind_rr_star_v4f4))    
    
    v,v1,v2,v3,v4 = mesh.rr_star[mesh.ind_rr_star_v4f4].T
    v,va,vb,vc,vd = mesh.rr_star_corner# in diagonal direction
    c_v = columnnew(v,0,mesh.V)
    c_1 = columnnew(v1,0,mesh.V)
    c_2 = columnnew(v2,0,mesh.V)
    c_3 = columnnew(v3,0,mesh.V)
    c_4 = columnnew(v4,0,mesh.V)     
    c_a = columnnew(va,0,mesh.V)
    c_b = columnnew(vb,0,mesh.V)
    c_c = columnnew(vc,0,mesh.V)
    c_d = columnnew(vd,0,mesh.V)
    c_n = Nanet-3*num+np.arange(3*num) # for AAG, AAGG

    def _1geo(X,w,c_v,c_a,c_c,c_an,c_on):
        "on = (Vc-V) x (Va-V); an * on = 0"
        H1,r1 = con_cross_product2(X,c_v,c_c,c_a,c_on,N)
        H2,r2 = con_dot(X,c_an,c_on,N)
        H = sparse.vstack((H1,H2))
        r = np.r_[r1,r2]
        return H*w, r*w
    def _1asym(X,w,c_v,c_a,c_c,c_n):
        "*asymptotic: ni*(va-v)=ni*(vc-v)=0"
        num = int(len(c_v)/3)
        H1,r1 = con_planarity(X,c_v,c_a,c_n,num,N)
        H2,r2 = con_planarity(X,c_v,c_c,c_n,num,N)
        Hu,ru = con_unit(X,c_n,num,N)
        H = sparse.vstack((H1,H2,Hu))
        r = np.r_[r1,r2,ru]
        return H*w, r*w

    def _gga(X,w,c_v,c_g1,c_g2,c_g3,c_g4,c_l,c_r,c_n,c_on1,c_on2):
        Ha,ra = _1asym(X,w,c_v,c_l,c_r,c_n)
        Ho1,ro1 = _1geo(X,w,c_v,c_g1,c_g3,c_n,c_on1)
        Ho2,ro2 = _1geo(X,w,c_v,c_g2,c_g4,c_n,c_on2)
        H = sparse.vstack((Ha,Ho1,Ho2))
        r = np.r_[ra,ro1,ro2]  
        return H,r
    def _aagg(X,w,c_v,c_g1,c_g2,c_g3,c_g4,c_a1,c_a2,c_a3,c_a4,c_n,c_on1,c_on2):
        Ha1,ra1 = _1asym(X,w,c_v,c_a1,c_a3,c_n)
        Ha2,ra2 = _1asym(X,w,c_v,c_a2,c_a4,c_n)
        Ho1,ro1 = _1geo(X,w,c_v,c_g1,c_g3,c_n,c_on1)
        Ho2,ro2 = _1geo(X,w,c_v,c_g2,c_g4,c_n,c_on2)
        H = sparse.vstack((Ha1,Ha2,Ho1,Ho2))
        r = np.r_[ra1,ra2,ro1,ro2]
        return H,r

    wr = checker_weight
    if w_aag or w_gaa:
        """X += [ni]+[Ni]; 
            ni: vertex-normal from Anet;
            Ni: osculating normal of geodesic
        <==> from Anet/Anet_diagnet: ni*(vi-v)=0,(i=1,2,3,4), ni^2=1;
             *geodesic: Ni=(Vc-V) x (Va-V); ni * Ni = 0
        """
        wag = max(w_aag,w_gaa)
        c_on = Ndgeo-3*num + arr3
        
        if w_aag:
            if another_poly_direction:
                c_l,c_r = c_b, c_d
            else:
                c_l,c_r = c_a, c_c
        elif w_gaa:
            if another_poly_direction:
                c_l,c_r = c_2, c_4
            else:
                c_l,c_r = c_1, c_3 
        
        if checker_weight<1:
            "at red-rr-vs, smaller weight"
            Hb,rb = _1geo(X,wag,c_v[ib],c_l[ib],c_r[ib],c_n[ib],c_on[ib])
            Hr,rr = _1geo(X,wr,c_v[ir],c_l[ir],c_r[ir],c_n[ir],c_on[ir])    
            H = sparse.vstack((Hb,Hr))
            r = np.r_[rb,rr]  
        else:
            "all rr-vs, same weight"
            H,r = _1geo(X,wag,c_v,c_l,c_r,c_n,c_on)
        
    elif w_gga or w_agg:
        """X += [Ni,No1,No2]; 
            Ni: vertex-normal of Anet;
            No1,No2: two oscualting normals of G-net
        <==>Ni^2=1; Ni*No1,No2,va-v,vc-v=0;
            No1=(vc-v) x (va-v), No2=(vd-v) x (vb-v)
        """
        wag = max(w_gga,w_agg)
        c_n = Ndgeo-9*num + arr3
        c_on1,c_on2 = c_n + 3*num,c_n + 6*num
        
        if w_gga:
            c_g1,c_g2,c_g3,c_g4 = c_1,c_2,c_3,c_4
            if another_poly_direction:
                c_l,c_r = c_b, c_d
            else:
                c_l,c_r = c_a, c_c
        elif w_agg:
            c_g1,c_g2,c_g3,c_g4 = c_a,c_b,c_c,c_d
            if another_poly_direction:
                c_l,c_r = c_2, c_4
            else:
                c_l,c_r = c_1, c_3 
        
        if checker_weight<1:
            "at red-rr-vs, smaller weight"
            Hb,rb = _gga(X,wag,c_v[ib],c_g1[ib],c_g2[ib],c_g3[ib],c_g4[ib],c_l[ib],c_r[ib],c_n[ib],c_on1[ib],c_on2[ib])
            Hr,rr = _gga(X,wr,c_v[ir],c_g1[ir],c_g2[ir],c_g3[ir],c_g4[ir],c_l[ir],c_r[ir],c_n[ir],c_on1[ir],c_on2[ir])
            H = sparse.vstack((Hb,Hr))
            r = np.r_[rb,rr]  
        else:
            "all rr-vs, same weight"
            H,r = _gga(X,wag,c_v,c_g1,c_g2,c_g3,c_g4,c_l,c_r,c_n,c_on1,c_on2)

    elif w_aagg or w_ggaa:
        """ X += [ni] + [No1,No2];
            ni: vertex-normal from Anet;
            No1,No2: two oscualting normals of G-net
        <==> *No1=(vc-v) x (va-v), No2=(vd-v) x (vb-v); ni*No1,No2=0
        """
        wag = max(w_aagg,w_ggaa)
        c_on1 = Ndgeo-6*num + arr3
        c_on2 = c_on1 + 3*num
        
        if w_aagg:
            c_g1,c_g2,c_g3,c_g4 = c_a,c_b,c_c,c_d ##different from above
            c_a1,c_a2,c_a3,c_a4 = c_1,c_2,c_3,c_4
        elif w_ggaa:
            c_g1,c_g2,c_g3,c_g4 = c_1,c_2,c_3,c_4 ##different from above
            c_a1,c_a2,c_a3,c_a4 = c_a,c_b,c_c,c_d
            
        if checker_weight<1:
            "at red-rr-vs, smaller weight"
            Hb,rb = _aagg(X,wag,c_v[ib],c_g1[ib],c_g2[ib],c_g3[ib],c_g4[ib],
                          c_a1[ib],c_a2[ib],c_a3[ib],c_a4[ib],
                          c_n[ib],c_on1[ib],c_on2[ib])
            Hr,rr = _aagg(X,wr,c_v[ir],c_g1[ir],c_g2[ir],c_g3[ir],c_g4[ir],
                          c_a1[ir],c_a2[ir],c_a3[ir],c_a4[ir],
                          c_n[ir],c_on1[ir],c_on2[ir])
            H = sparse.vstack((Hb,Hr))
            r = np.r_[rb,rr]   
        else:
            "all rr-vs, same weight"
            H,r = _aagg(X,wag,c_v,c_g1,c_g2,c_g3,c_g4,c_a1,c_a2,c_a3,c_a4,
                        c_n,c_on1,c_on2)

    w5 = kwargs.get('agnet_liouville') # no need now.
    w6 = kwargs.get('planar_geodesic') # no need now.
    Ndgeoliou = kwargs.get('Ndgeoliou')
    if w5: # no need now.
        "X +=[lu1,tu1; lla,llc,g1, lg1,tg1]"
        arr = np.arange(num)
        n = Ndgeoliou - 13*num -1
        c_lu1 = n+arr
        c_tu1 = n+num+arr3
        c_lla = n+4*num+arr
        c_llc = c_lla+num
        c_g1 = n+6*num+arr3
        c_lg1 = n+9*num+arr
        c_tg1 = n+10*num+arr3
        c_const = Ndgeoliou - 1
        #c_geo = [c_lu1,c_tu1,c_lla,c_llc,c_g1,c_lg1,c_tg1,c_const]
        
        if w_gaa:
            if another_poly_direction:
                c_geo = [c_v,c_2,c_4,c_lu1,c_tu1,c_lla,c_llc,c_g1,c_lg1,c_tg1,c_const]
            else:
                c_geo = [c_v,c_1,c_3,c_lu1,c_tu1,c_lla,c_llc,c_g1,c_lg1,c_tg1,c_const]
        elif w_aag:
            if another_poly_direction:
                c_geo = [c_v,c_b,c_d,c_lu1,c_tu1,c_lla,c_llc,c_g1,c_lg1,c_tg1,c_const]
            else:
                c_geo = [c_v,c_a,c_c,c_lu1,c_tu1,c_lla,c_llc,c_g1,c_lg1,c_tg1,c_const]
            
        H0,r0 = _con_agnet_liouville(c_geo,num,**kwargs)
        H = sparse.vstack((H,H0))
        r = np.r_[r,r0]
    if w6: # no need now.
        H0,r0 = _con_agnet_planar_geodesic(ver_poly_strip,**kwargs)
        H = sparse.vstack((H,H0))
        r = np.r_[r,r0]
    return H,r


def con_AGnet(is_ag_or_ga=True,is_ortho=False,
              is_const_r=False,is_unique_r=False,**kwargs):
    """ based on pre-defined osculating_tangent: con_osculating_tangents()
        X +=[ll1,ll2,ll3,ll4,lt1,lt2,t1,t2]
        v1-v-v3: 
            lt*t = l1**2*(V3-V0) - l3**2*(V1-V0)
            t^2=1
        <===>
            ll1 (= l1**2) = (V1-V0)^2
            ll3 (= l3**2) = (V3-V0)^2
            ll1 * (v3-v0) - ll3 * (v1-v0) - t*lt = 0
            t^2=1
            
        asymptotic v1-v-v3; geodesic v2-v-v4; 
        X += [surfN; ogN]
        unit surfN // principalnormal of geodesic _|_ edges of asymptotic
        constraints:
            1. surfN^2=1
            2. surfN * t2 = 0;
            3. surfN * (v1-v) = 0
            4. surfN * (v3-v) = 0
            5. ogN^2=1
            6. ogN * (v2-v) = 0
            7. ogN * (v4-v) = 0
            8. ogN * surfN = 0
            if ortho.
                 t1 * t2 = 0
            if const.r. 
                X+=[Ri],each geodesic assigned Ri=ri^2, or 1 whole R=const.r^2
                (v1-v)^2 = 4*[(v1-v)*surfN/|v1-v|]^2 *r^2
                <==> ll1 = 4*[]^2 * r^2
                <==> ll1^2 = 4* C *R, C:= [(v1-v)*surfN]^2
    """
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Noscut = kwargs.get('Noscut')
    Nag = kwargs.get('Nag')
    wag = kwargs.get('AGnet')
    #igeo = mesh.igeopoly#TODO
    num=len(mesh.ind_rr_star_v4f4)
    arr,arr3 = np.arange(num),np.arange(3*num)
    if is_const_r or is_unique_r:
        if is_const_r:
            pass
            # k = len(igeo)
            # c_ri = Nag-k+np.arange(k)
            # c_srfN = Nag-6*num+arr3-k
            # c_ogN = Nag-4*num+arr3-k
        elif is_unique_r:
            c_r = Nag-1
            c_srfN = Nag-6*num+arr3-1
            c_ogN = Nag-4*num+arr3-1
    else:
        c_srfN = Nag-6*num+arr3
        c_ogN = Nag-3*num+arr3
    
    n = Noscut - 12*num
    c_ll1,c_ll2 = n+arr,n+arr+num
    c_t1,c_t2 = n+6*num+arr3, n+9*num+arr3
    v,v1,v2,v3,v4 = mesh.rr_star[mesh.ind_rr_star_v4f4].T
    c_v = columnnew(v,0,mesh.V)
    c_1 = columnnew(v1,0,mesh.V)
    c_2 = columnnew(v2,0,mesh.V)
    c_3 = columnnew(v3,0,mesh.V)
    c_4 = columnnew(v4,0,mesh.V)

    if is_ag_or_ga:
        "asy(1-v-3), geo(2-v-4)"
        c1,c2,c3,c4 = c_1,c_2,c_3,c_4
    else:
        "asy(2-v-4), geo(1-v-3)"
        c1,c2,c3,c4 = c_2,c_1,c_4,c_3
        c_ll1,c_ll2 = c_ll2,c_ll1
        c_t1,c_t2 = c_t2,c_t1
 
    def _AG():
        "surfN^2=1"
        H1,r1 = con_unit(X,c_srfN,num,N)
        "surfN * t2 = 0;"
        H2,r2 = con_dot(X,c_t2,c_srfN,N)
        "surfN*(v1-v)=0; surfN*(v3-v)=0;"
        H3,r3 = con_planarity(X,c_v,c1,c_srfN,num,N)
        H4,r4 = con_planarity(X,c_v,c3,c_srfN,num,N)
        "ogN^2=1; ogN*(v2-v)=0; ogN*(v4-v)=0"
        H5,r5 = con_unit(X,c_ogN,num,N)
        H6,r6 = con_planarity(X,c_v,c2,c_ogN,num,N)
        H7,r7 = con_planarity(X,c_v,c4,c_ogN,num,N)
        "ogN * surfN = 0"
        H8,r8 = con_dot(X,c_srfN,c_ogN,N)
        H = sparse.vstack((H1,H2,H3,H4,H5,H6,H7,H8))
        r = np.r_[r1,r2,r3,r4,r5,r6,r7,r8]
        #print('err:1:',np.sum(np.square(H1*X-r1)))
        #print('err:2:',np.sum(np.square(H2*X-r2)))
        print('err:3:',np.sum(np.square(H3*X-r3)))
        print('err:4:',np.sum(np.square(H4*X-r4)))
        # print('err:5:',np.sum(np.square(H5*X-r5)))
        # print('err:6:',np.sum(np.square(H6*X-r6)))
        # print('err:7:',np.sum(np.square(H7*X-r7)))
        # print('err:8:',np.sum(np.square(H8*X-r8)))
        return H,r
    
    H,r = _AG()
    
    if is_ortho:
        "t1 * t2 = 0"
        Ho,ro = con_dot(X,c_t1,c_t2,N)
        H = sparse.vstack((H,Ho))
        r = np.r_[r,ro]
        #print('err:o:',np.sum(np.square(Ho*X-ro)))
    if is_const_r or is_unique_r:
        if is_const_r:
            "num_m is the num of geodesic"
            pass
        elif is_unique_r:
            "ll1^2 = 4* C *R, C:= [(v1-v)*surfN]^2"
            VV1 = (X[c1]-X[c_v]).reshape(-1,3,order='F')
            srfN = X[c_srfN].reshape(-1,3,order='F')
            C = np.einsum('ij,ij->i',VV1,srfN)**2
            col = np.r_[c_ll1,np.ones(num,dtype=int)*c_r]
            row = np.tile(arr,2)
            data = np.r_[2*X[c_ll1],-4*C]
            rr = X[c_ll1]**2
            Hr = sparse.coo_matrix((data,(row,col)), shape=(num, N))
            print('err:r:',np.sum(np.square(Hr*X-rr)))
        H = sparse.vstack((H,Hr))
        r = np.r_[r,rr]
    return H*wag,r*wag

def con_singular_Anet_diag_geodesic(singular_polylist,ind_anet,**kwargs):
    "for singular A-net, 1family diagonals are geodesic"
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Ndgeo = kwargs.get('Ndgeo')
    Nanet = kwargs.get('Nanet')
    vl,vc,vr = singular_polylist
    c_l = columnnew(vl,0,mesh.V)
    c_v = columnnew(vc,0,mesh.V)
    c_r = columnnew(vr,0,mesh.V)
    num = len(vc)
    arr3 = np.arange(3*num)
    c_on = Ndgeo-3*num + arr3
    num_anet=len(mesh.ind_rr_star_v4f4)
    #c_anet_n = columnnew(ind_anet,Nanet-3*num_anet,num_anet)
    c_anet_n = np.r_[ind_anet,ind_anet+num_anet,ind_anet+2*num_anet]+Nanet-3*num_anet
    #print(len(c_on),len(c_anet_n),num,num_anet)
    def _1geo(c_v,c_a,c_c,c_an,c_on):
        "based on control-net==A-net"
        "an:Anet-normal; on:Osculating-normal"
        "on=(Vc-V)x(Va-V)"
        H1,r1 = con_cross_product2(X,c_v,c_a,c_c,c_on,N)
        "an*on=0"
        H2,r2 = con_dot(X,c_an,c_on,N)
        H = sparse.vstack((H1,H2))
        r = np.r_[r1,r2]
        return H,r       
    H,r = _1geo(c_v,c_l,c_r,c_anet_n,c_on)
    return H,r

def con_diag_1_asymptotic_or_geodesic(singular_polylist=None,
                                      ind_rrv=None,
                                      another_poly_direction=False,
                                      is_asym_or_geod = True,
                                      **kwargs):
    """ normal at vs from control-mesh two polylines tangents: t1 x t2 // N
    default direction: va-v-vc; 
                 else: vb-v-vd
    common:
        <==> (v3-v1) x (v4-v2) = un * l; un^2=1         
    if asymptotic: 
        X += [v4N] ##len=[3]
        <==> uN * (va-v) = uN * (vc-v) == 0
    elif geodesic:
        X += [v4N,la,lc,ea,ec] ##len=[3+1+1+3+3]
        <==> uN x (ea+ec) == 0; 
            (va-v) = la * ea; (vc-v) = lc * ec; ea^2=1; ec^2=1;
    """
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    num = len(mesh.ind_rr_star_v4f4)
    arr,arr3 = np.arange(num), np.arange(3*num)    
    
    v,v1,v2,v3,v4 = mesh.rr_star[mesh.ind_rr_star_v4f4].T
    c_v = columnnew(v,0,mesh.V)
    c_v1 = columnnew(v1,0,mesh.V)
    c_v2 = columnnew(v2,0,mesh.V)
    c_v3 = columnnew(v3,0,mesh.V)
    c_v4 = columnnew(v4,0,mesh.V)  
    
    if singular_polylist is not None:
        vl,vc,vr = singular_polylist
        c_v0 = columnnew(vc,0,mesh.V)
        c_vl = columnnew(vl,0,mesh.V)
        c_vr = columnnew(vr,0,mesh.V)
        ind = ind_rrv
        ind3 = columnnew(ind,0,num)
        if is_asym_or_geod:
            "uN * (va-v) = uN * (vc-v) == 0"
            w = kwargs.get('diag_1_asymptotic')
            Ncd = kwargs.get('Ncd')-4*num
            c_l,c_n = Ncd+arr, Ncd+num+arr3
            H1,r1 = con_planarity(X,c_vl,c_v0,c_n[ind3],len(ind),N)#change
            H2,r2 = con_planarity(X,c_vr,c_v0,c_n[ind3],len(ind),N)#change
            H = sparse.vstack((H1,H2))
            r = np.r_[r1,r2]
        else:
            "uN x (ea+ec) == 0;(va-v) = la * ea; (vc-v) = lc * ec; ea^2=1; ec^2=1;"
            w = kwargs.get('diag_1_geodesic')
            num_ind = len(ind)#new
            arr_ind,arr3_ind = np.arange(num_ind),np.arange(3*num_ind)#new
            Ncd = kwargs.get('Ncd')-4*num-8*num_ind#change
            c_l,c_n = Ncd+arr, Ncd+num+arr3
            c_la,c_ea = Ncd+4*num+arr_ind, Ncd+4*num+2*num_ind+arr3_ind#change
            c_lc,c_ec = c_la+num_ind, c_ea+3*num_ind#change
            
            H1,r1 = con_edge(X,c_vl,c_v0,c_la,c_ea,num_ind,N)#change
            H2,r2 = con_edge(X,c_vr,c_v0,c_lc,c_ec,num_ind,N)#change
            #print('err:edge:',np.sum(np.square(H1*X-r1)))
            Hu1,ru1 = con_unit(X,c_ea,num_ind,N)#change
            Hu2,ru2 = con_unit(X,c_ec,num_ind,N)#change
            #print('err:unit:',np.sum(np.square(Hu1*X-ru1)))
            Hb,rb = con_bisecting_vector(X,c_n[ind3],c_ea,c_ec,N)#change
            #print('err:geod:',np.sum(np.square(Hb*X-rb)))
            H = sparse.vstack((H1,H2,Hu1,Hu2,Hb))
            r = np.r_[r1,r2,ru1,ru2,rb]        
    else:
        _,va,vb,vc,vd = mesh.rr_star_corner# in diagonal direction
        if another_poly_direction:
            c_va = columnnew(vb,0,mesh.V)
            c_vc = columnnew(vd,0,mesh.V)
        else:
            c_va = columnnew(va,0,mesh.V)
            c_vc = columnnew(vc,0,mesh.V)    
        if is_asym_or_geod:
            "uN * (va-v) = uN * (vc-v) == 0"
            w = kwargs.get('diag_1_asymptotic')
            Ncd = kwargs.get('Ncd')-4*num
            c_l,c_n = Ncd+arr, Ncd+num+arr3
            H1,r1 = con_planarity(X,c_va,c_v,c_n,num,N)
            H2,r2 = con_planarity(X,c_vc,c_v,c_n,num,N)
            H = sparse.vstack((H1,H2))
            r = np.r_[r1,r2]
        else:
            "uN x (ea+ec) == 0;(va-v) = la * ea; (vc-v) = lc * ec; ea^2=1; ec^2=1;"
            w = kwargs.get('diag_1_geodesic')
            Ncd = kwargs.get('Ncd')-12*num
            c_la,c_ea = Ncd+4*num+arr, Ncd+6*num+arr3
            c_lc,c_ec = c_la+num, c_ea+3*num
            c_l,c_n = Ncd+arr, Ncd+num+arr3
            H1,r1 = con_edge(X,c_va,c_v,c_la,c_ea,num,N)
            H2,r2 = con_edge(X,c_vc,c_v,c_lc,c_ec,num,N)
            Hu1,ru1 = con_unit(X,c_ea,num,N)
            Hu2,ru2 = con_unit(X,c_ec,num,N)
            Hb,rb = con_bisecting_vector(X,c_n,c_ea,c_ec,N)
            H = sparse.vstack((H1,H2,Hu1,Hu2,Hb))
            r = np.r_[r1,r2,ru1,ru2,rb]

    "uN * l = (v3-v1) x (v4-v2)"
    H0,r0 = con_unit_tangentplane_normal(X,c_v1,c_v2,c_v3,c_v4,c_n,c_l,N)  
    H = sparse.vstack((H,H0))
    r = np.r_[r,r0]
    return H*w,r*w
    
def con_polyline_ruling(switch_diagmeth=False,**kwargs):
    """ X +=[ni]
    along each i-th polyline: ti x (vij-vik) = 0; k=j+1,j=0,...
    refer: self._con_agnet_planar_geodesic(),self.get_poly_strip_ruling_tangent()
    """
    w = kwargs.get('ruling')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Nruling = kwargs.get('Nruling')
    
    iall = mesh.get_both_isopolyline(diagpoly=switch_diagmeth) # interval is random
    num = len(iall)
    arr = Nruling-3*num+np.arange(3*num)
    c_tx,c_ty,c_tz = arr[:num],arr[num:2*num],arr[2*num:3*num]

    alla=allb = np.array([],dtype=int)
    alltx=allty=alltz = np.array([],dtype=int)
    i = 0
    for iv in iall:
        "t x (a-b) = 0"
        va,vb = iv[:-1],iv[1:]
        alla = np.r_[alla,va]
        allb = np.r_[allb,vb]
        m = len(va)
        alltx = np.r_[alltx,np.tile(c_tx[i],m)]
        allty = np.r_[allty,np.tile(c_ty[i],m)]
        alltz = np.r_[alltz,np.tile(c_tz[i],m)]
        i += 1
    c_a = columnnew(alla,0,mesh.V)
    c_b = columnnew(allb,0,mesh.V)
    c_ti = np.r_[alltx,allty,alltz]
    H,r = con_dependent_vector(X,c_a,c_b,c_ti,N)
    H1,r1 = con_unit(X,arr,num,N)
    H = sparse.vstack((H,H1))
    r = np.r_[r,r1]
    #self.add_iterative_constraint(H * w, r * w, 'ruling')    
    return H*w, r*w
 
def con_osculating_tangents(diagnet=False,is_ortho=False,**kwargs):
    """X +=[ll1,ll2,ll3,ll4,lt1,lt2,t1,t2]
    t1,t2 are built from _con_osculating_tangent
    if orthogonal: t1 * t2 = 0
    """
    w = kwargs.get('oscu_tangent')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Noscut = kwargs.get('Noscut')
    
    if diagnet:
        v,va,vb,vc,vd = mesh.rr_star_corner# in diagonal direction
        c_v = columnnew(v,0,mesh.V)
        c_1 = columnnew(va,0,mesh.V)
        c_2 = columnnew(vb,0,mesh.V)
        c_3 = columnnew(vc,0,mesh.V)
        c_4 = columnnew(vd,0,mesh.V)
    else:
        v,v1,v2,v3,v4 = mesh.rr_star[mesh.ind_rr_star_v4f4].T
        c_v = columnnew(v,0,mesh.V)
        c_1 = columnnew(v1,0,mesh.V)
        c_2 = columnnew(v2,0,mesh.V)
        c_3 = columnnew(v3,0,mesh.V)
        c_4 = columnnew(v4,0,mesh.V)         
    num = len(v)
    arr,arr3 = np.arange(num),np.arange(3*num)
    n = Noscut - 12*num
    c_ll1 = n+arr
    c_ll2,c_ll3,c_ll4 = c_ll1+num, c_ll1+2*num, c_ll1+3*num
    c_lt1,c_lt2 = c_ll1+4*num, c_ll1+5*num
    c_t1,c_t2 = n+6*num+arr3, n+9*num+arr3
    H1,r1 = con_osculating_tangent(X,c_v,c_1,c_3,c_ll1,c_ll3,c_lt1,c_t1,num,N)
    H2,r2 = con_osculating_tangent(X,c_v,c_2,c_4,c_ll2,c_ll4,c_lt2,c_t2,num,N)
    H = sparse.vstack((H1,H2))
    r = np.r_[r1,r2]
    if is_ortho:
        H3,r3 = con_dot(X,c_t1,c_t2,N)
        H = sparse.vstack((H,H3))*1
        r = np.r_[r,r3]*1
        ##print('ortho:',np.sum(np.square(H*X-r)))  
    #print('err:T:',np.sum(np.square(H*X-r)))
    return H*w, r*w

def con_planar_1familyof_polylines(ver_poly_strip,is_parallxy_n=False,**kwargs):
    """ refer: _con_agnet_planar_geodesic(ver_poly_strip,strong=True,**kwargs)
    X +=[ni]
    along each i-th polyline: ni * (vij-vik) = 0; k=j+1,j=0,...
    refer: self.get_poly_strip_normal()
    """
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    N = kwargs.get('N')
    Npp = kwargs.get('Npp')

    iall = ver_poly_strip
    num = len(iall)
    arr = Npp-3*num+np.arange(3*num)
    c_nx,c_ny,c_nz = arr[:num],arr[num:2*num],arr[2*num:3*num]

    col=row=data=r = np.array([])
    k,i = 0,0
    for iv in iall:
        va,vb = iv[:-1],iv[1:]
        m = len(va)
        c_a = columnnew(va,0,mesh.V)
        c_b = columnnew(vb,0,mesh.V)
        c_ni = np.r_[np.tile(c_nx[i],m),np.tile(c_ny[i],m),np.tile(c_nz[i],m)]
        coli = np.r_[c_a,c_b,c_ni]
        rowi = np.tile(np.arange(m),9) + k
        datai = np.r_[X[c_ni],-X[c_ni],X[c_a]-X[c_b]]
        ri = np.einsum('ij,ij->i',X[c_ni].reshape(-1,3,order='F'),(X[c_a]-X[c_b]).reshape(-1,3,order='F'))
        col = np.r_[col,coli]
        row = np.r_[row,rowi]
        data = np.r_[data,datai]
        r = np.r_[r,ri]
        k += m
        i += 1
    H = sparse.coo_matrix((data,(row,col)), shape=(k, N))
    H1,r1 = con_unit(X,arr,num,N)
    H = sparse.vstack((H,H1))
    r = np.r_[r,r1]
    
    if is_parallxy_n:
        "variable normals are parallel to xy plane: n[2]=0"
        row = np.arange(num)
        data = np.ones(num)
        col = c_nz
        r0 = np.zeros(num)
        H0 = sparse.coo_matrix((data,(row,col)), shape=(num, N))  
        H = sparse.vstack((H,H0))
        r = np.r_[r,r0]
    return H,r    
    #--------------------------------------------------------------------------
    #                       Quad-Gi:
    #--------------------------------------------------------------------------  
def con_quad_gi_vn(**kwargs):
    """
    sig-2020-isometry(Caigui):
        (given unit nf = (v2-v0) x (v3-v1), and vi, getting nj)
        csym(f) =⟨n3 −n1,v2 −v0⟩−⟨v3 −v1,n2 −n0⟩=0
        cnorm,j(f)=⟨nj +nj+1,nf⟩−2=0, (j=0,...,3)
    """      
    w = kwargs.get('quad_gi_vn')
    mesh = kwargs.get('mesh')
    N = kwargs.get('N')
    Ngi_quad = kwargs.get('Ngi_quad')
    
    Vi = mesh.vertices
    V = mesh.V
    numvn = Ngi_quad-3*V
    num = mesh.num_quadface
    
    vi = mesh.quadface
    iv0,iv1,iv2,iv3 = vi[::4],vi[1::4],vi[2::4],vi[3::4]
    nf = np.cross(Vi[iv2]-Vi[iv0],Vi[iv3]-Vi[iv1])
    nf = nf / np.linalg.norm(nf,axis=1)[:,None]
    nf = nf.flatten('F')
    
    v0 = Vi[iv0].flatten('F')       
    v1 = Vi[iv1].flatten('F') 
    v2 = Vi[iv2].flatten('F') 
    v3 = Vi[iv3].flatten('F') 
    
    c_v0 = np.r_[iv0,V+iv0,2*V+iv0]    
    c_v1 = np.r_[iv1,V+iv1,2*V+iv1] 
    c_v2 = np.r_[iv2,V+iv2,2*V+iv2] 
    c_v3 = np.r_[iv3,V+iv3,2*V+iv3]  
    
    c_n0 = numvn + c_v0  
    c_n1 = numvn + c_v1 
    c_n2 = numvn + c_v2  
    c_n3 = numvn + c_v3  
    
    def _con_norm(c_nj0,c_nj1):
        col = np.r_[c_nj0,c_nj1]
        row = np.tile(np.arange(num),6)
        data = np.r_[nf,nf]
        r = np.ones(num)*2
        H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
        return H,r
    
    H1,r1 = _con_norm(c_n0,c_n1)
    H2,r2 = _con_norm(c_n1,c_n2)
    H3,r3 = _con_norm(c_n2,c_n3)
    H4,r4 = _con_norm(c_n3,c_n0)
    
    row5 = np.tile(np.arange(num),12)
    col5 = np.r_[c_n0,c_n1,c_n2,c_n3]
    data5 = np.r_[v3-v1,v0-v2,v1-v3,v2-v0]
    H5 = sparse.coo_matrix((data5,(row5,col5)), shape=(num, N))
    r5 = np.zeros(num)
    H = sparse.vstack((H1, H2, H3, H4, H5))
    r = np.r_[r1,r2,r3,r4,r5]
    
    #self.add_iterative_constraint(H, r, 'quad_gi_vn')
    #print('err-quad_gi_vn:',np.sum(np.square(H*X-r)))
    return H*w,r*w