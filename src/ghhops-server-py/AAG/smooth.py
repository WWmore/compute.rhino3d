# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import print_function

from __future__ import division
#---------------------------------------------------------------------------
import numpy as np
from scipy import sparse
try:
    from pypardiso import spsolve
except:
    from scipy.sparse.linalg import spsolve
# -----------------------------------------------------------------------------
try:
    from constraints_basic import con_fair_midpoint0
except:
    pass
#-----------------------------------------------------------------------------
'''_'''

__author__ = 'Hui Wang'

#------------------------------------------------------------------------------
#                                 GENERATION
#------------------------------------------------------------------------------

def fair_vertices_or_vectors(Vertices,itera=10,efair=0.005,ee=0.001,is_fix=False):
    """
    given vertices(vectors)
    efair: fairness weight
    return same type, but after fairness-optimization
    """
    def matrix_fair(iva,ivb,ivc,num,var,efair):
        """midpoint: 2Q2 = Q1+Q3;
        """
        c_va = np.r_[iva, num+iva, 2*num+iva]
        c_vb = np.r_[ivb, num+ivb, 2*num+ivb]
        c_vc = np.r_[ivc, num+ivc, 2*num+ivc]
        K = con_fair_midpoint0(c_vb,c_va,c_vc,var)
        return efair * K    
    X = Vertices.flatten('F')
    num,var = len(Vertices),len(X)
    iva=np.arange(num-2)
    ivb,ivc = iva+1, iva+2
    K = matrix_fair(iva,ivb,ivc,num,var,efair)
    I = sparse.eye(var,format='coo')*ee**2
    r = np.zeros(K.shape[0])
    if is_fix:
        "fix two endpoint"
        vfix = np.r_[Vertices[0],Vertices[-1]] ##[x,y,z,x,y,z]
        col = np.array([0,num,2*num,num-1,2*num-1,3*num-1],dtype=int)
        data = np.ones(6)
        row = np.arange(6)
        F = sparse.coo_matrix((data,(row,col)), shape=(6, var)) 
        K = sparse.vstack((K,F*10))
        r = np.r_[r,vfix*10]
    n = 0
    opt_num = 100
    while n < itera and opt_num>1e-7 and opt_num<1e+6:
        X = spsolve(K.T*K+I, K.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
        n += 1
        opt_num = np.sum(np.square((K*X)))
    #print('fair-vectors:',n, '%.2g' %opt_num)
    return X.reshape(-1,3,order='F')

    
    
def intersectLine(V1=[0,0,0],N1=[0,0,1],V2=[0,1,0],N2=[0,1,1],N3=[1,1,1]):
    """
    from given normal Ni and a vertex Vi to compute the tangent plane
    from two planes to get intersected line(ruling): a point and its unit vector N3//N1XN2
    from (V1+V2)/2 and N3 to get another plane
    from these three plane to get an intersected Point P
    together P and N3 is the ruling
    """
#    #N3 = np.cross(N1,N2) / np.linalg.norm(np.cross(N1,N2))
#    V3 = [(V1[0]+V2[0])/2.0,(V1[1]+V2[1])/2.0,(V1[2]+V2[2])/2.0]
#    x = Symbol('x')
#    y = Symbol('y')
#    z = Symbol('z')
#    ss=solve([N1[0]*(x-V1[0])+N1[1]*(y-V1[1])+N1[2]*(z-V1[2]), \
#       N2[0]*(x-V2[0])+N2[1]*(y-V2[1])+N2[2]*(z-V2[2]), \
#       N3[0]*(x-V3[0])+N3[1]*(y-V3[1])+N3[2]*(z-V3[2])],[x, y, z])
#    point = [float(ss[x]),float(ss[y]),float(ss[z])]
#    #ruling = N3
#    return point

    V3 = [(V1[0]+V2[0])/2.0,(V1[1]+V2[1])/2.0,(V1[2]+V2[2])/2.0]
    A = np.array([N1,N2,N3])
    b = np.array([np.dot(N1,V1),np.dot(N2,V2),np.dot(N3,V3)])
    point = spsolve(A,b)
    return point

# note that the ss[x] is sympy.core.numbers
# should transver it to float, otherwise can't comput np.linalg.norm()



#import numpy as np
#from scipy.linalg import spsolve
#a = np.array([[3, 1, -2], [1, -1, 4], [2, 0, 3]])
#b = np.array([5, -2, 2.5])
#x = spsolve(a, b)
#print(x)

