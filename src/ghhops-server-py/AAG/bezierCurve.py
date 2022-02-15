# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import print_function

from __future__ import division
#---------------------------------------------------------------------------

__author__ = 'Hui'

#---------------------------------------------------------------------------

import numpy as np

from scipy.special import comb

#---------------------------------------------------------------------------


def bernstein_poly(i, n, t, reverse=True):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    if reverse:
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
    else:
        return comb(n, i) * ( (1 - t)**(n-i) ) * t**i

# from control_points to curve_points:
def bezier_curve(points, nTimes=500, is_poly=False, is_crv=False):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    
    Hui Note: given points, the returned crv points in reverse direction:
                Q[0] == points[-1]; Q[-1] == points[0]
    """
    if is_poly:
        from geometrylab.geometry import Polyline
        return Polyline(points,closed=False)
    else:
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        zPoints = np.array([p[2] for p in points])
    
        t = np.linspace(0.0, 1.0, nTimes)
    
        polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        zvals = np.dot(zPoints, polynomial_array)
        Q = np.flip(np.c_[xvals,yvals,zvals],axis=0)

        if is_crv:
            crv = Polyline(Q,closed=False)
            return crv
        else:
            return Q


def berstein_matrix(n):
 #bezierM = lambda ts: np.matrix([[bernstein_poly(i,5,t) for i in range(6)] for t in ts])
    ts = np.linspace(0.0, 1.0, n)
    m = []
    for t in ts:
        m.append([bernstein_poly(i,n-1,t) for i in range(n)])
    m.reverse()
    return np.array(m)

# from curve_points to control_points:
def bezier_controlPoints(P):
    n = len(P)
    M = berstein_matrix(n)
    Q = np.dot(np.linalg.inv(M), P)
    return Q

def bezier_degree_evaluation(P, deg_evl):
    "Qi = i/(n+1) * P_{i-1} + (1-i/(n+1)) * P_i"
    "only from current curves' control_points_P to get evaluated control_points_Q"
    "deg_evl > len(P)-1 = n"
    num = len(P) # num=4, control points, P0,P1,P2,P3
    n = num-1    # degree=3, deg_evl=5

    Q = np.zeros((deg_evl+1,3))
    Q[0,:] = P[0,:]
    Q[-1,:] = P[-1,:]

    while n < deg_evl:   # 3<5
        for j in range(n): # 0,1,2
            i = j+1        # 1,2,3
            Q[i,:] = float(i)/(n+1)*P[i-1,:] + (1-float(i)/(n+1))*P[i,:]
        n += 1 # 4
        P = np.vstack((Q[:n,:], P[-1,:]))

    return Q

# def bezier_derivative(points, nTimes=100, d=1, is_poly=False, is_crv=False): #Hui add:
#     if d==1:
#         "first derivative"
#         nPoints = len(points)-1
#         xPoints = np.array([p[0] for p in points])
#         yPoints = np.array([p[1] for p in points])
#         zPoints = np.array([p[2] for p in points])
        
#         dx = xPoints[1:]-xPoints[:-1]
#         dy = yPoints[1:]-yPoints[:-1]
#         dz = zPoints[1:]-zPoints[:-1]
    
#         t = np.linspace(0.0, 1.0, nTimes)
    
#         polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
#         xvals = np.dot(dx, polynomial_array) * len(points)
#         yvals = np.dot(dy, polynomial_array) * len(points)
#         zvals = np.dot(dz, polynomial_array) * len(points)
#         Q = np.flip(np.c_[xvals,yvals,zvals],axis=0)

#     if is_crv:
#         from geometrylab.geometry import Polyline
#         crv = Polyline(Q,closed=False)
#         return crv
#     else:
#         return Q
    
def bezier_tangent_atparameters(ctrl_points,ti):
    "len(ti)>=1"
    n = len(ctrl_points) #==6
    dP = ctrl_points[1:,:] - ctrl_points[:-1,:]
    arr1 = np.array([bernstein_poly(i,n-2,ti,False) for i in range(0, n-1)]) # len=n-1

    xvals = np.dot(dP[:,0], arr1)
    yvals = np.dot(dP[:,1], arr1)
    zvals = np.dot(dP[:,2], arr1)
    Q = np.flip(np.c_[xvals,yvals,zvals],axis=0)     
    T = Q / np.linalg.norm(Q,axis=1)[:,None]
    return T


def bezier_curvature_ataparameter(ctrl_points,ti,surfN=None,is_absolute=True):
    "P'(ti), P''(ti), P'''(ti) ,  ti=1/5, 2/5, 3/5, 4/5"
    n = len(ctrl_points) #==6
    n1,n2,n3 = n-1, n-2, n-3
    
    P = ctrl_points
    dP = P[1:,:] - P[:-1,:]
    ddP = dP[1:,:] - dP[:-1,:]
    dddP = ddP[1:,:] - ddP[:-1,:]

    arr  = np.array([bernstein_poly(i,n -1,ti,False) for i in range(0, n)])  # len=n
    arr1 = np.array([bernstein_poly(i,n1-1,ti,False) for i in range(0, n1)]) # len=n-1
    arr2 = np.array([bernstein_poly(i,n2-1,ti,False) for i in range(0, n2)]) # len=n-2
    arr3 = np.array([bernstein_poly(i,n3-1,ti,False) for i in range(0, n3)]) # len=n-3
    
    Pi = np.sum(P*arr,axis=0)
    dP = np.sum(dP*arr1,axis=0) * comb(n-1, 1)
    ddP = np.sum(ddP*arr2,axis=0) * comb(n-1, 2)
    dddP = np.sum(dddP*arr3,axis=0) * comb(n-1, 3)
    
    T = dP / np.linalg.norm(dP)
    k = np.linalg.norm(np.cross(dP,ddP)) / np.linalg.norm(dP)**3
    tau = np.dot(np.cross(dP,ddP), dddP) / np.linalg.norm(np.cross(dP,ddP))**2
    kg,kn,d = None,None,None
    if surfN is not None:
        kg = np.dot(np.cross(surfN,dP),ddP) / np.linalg.norm(dP)**3
        kn = np.dot(surfN,ddP) / np.linalg.norm(dP)**2
        if is_absolute:
            k,kg,kn,tau = np.abs(k),np.abs(kg),np.abs(kn),np.abs(tau)
        S = np.cross(surfN,T)
        d = tau*T-kn*S+kg*surfN
        d = d / np.linalg.norm(d)
    return Pi, T, [k,tau,kg,kn,d]

def bezier_curvature_atequalparameters(ctrl_points,is_asy_or_geo=True,num=5):
    "DENSE: P'(ti), P''(ti), P'''(ti) ,  ti=1/5, 2/5, 3/5, 4/5"
    n = len(ctrl_points)-1#==6-1
    n1,n2,n3 = n-1, n-2, n-3
    ti = np.arange(num+1) / num  #### ti==[0, 0.2, 0.4, 0.6, 0.8, 1] 
    P = ctrl_points
    dP = P[1:,:] - P[:-1,:]
    ddP = dP[1:,:] - dP[:-1,:]
    dddP = ddP[1:,:] - ddP[:-1,:]

    arr  = np.array([bernstein_poly(i,n,ti,False) for i in range(0, n+1)])  # len=6
    arr1 = np.array([bernstein_poly(i,n1,ti,False) for i in range(0, n1+1)]) # len=5
    arr2 = np.array([bernstein_poly(i,n2,ti,False) for i in range(0, n2+1)]) # len=4
    arr3 = np.array([bernstein_poly(i,n3,ti,False) for i in range(0, n3+1)]) # len=3

    xvals = np.dot(P[:,0], arr)
    yvals = np.dot(P[:,1], arr)
    zvals = np.dot(P[:,2], arr)
    #Pi = np.flip(np.c_[xvals,yvals,zvals],axis=0)
    Pi = np.c_[xvals,yvals,zvals]
    
    xvals = np.dot(dP[:,0], arr1)
    yvals = np.dot(dP[:,1], arr1)
    zvals = np.dot(dP[:,2], arr1)
    T = np.c_[xvals,yvals,zvals]*n
    E1 = T / np.linalg.norm(T,axis=1)[:,None]

    xvals = np.dot(ddP[:,0], arr2)
    yvals = np.dot(ddP[:,1], arr2)
    zvals = np.dot(ddP[:,2], arr2)
    dT = np.c_[xvals,yvals,zvals]*n*(n-1)
    E2 = dT / np.linalg.norm(dT,axis=1)[:,None] ##NOTE: infact, for general t, E2 is not _|_  E1, but E3//E1XE2

    xvals = np.dot(dddP[:,0], arr3)
    yvals = np.dot(dddP[:,1], arr3)
    zvals = np.dot(dddP[:,2], arr3)
    ddT = np.c_[xvals,yvals,zvals]*n*(n-1)*(n-2)
    
    k = np.linalg.norm(np.cross(T,dT),axis=1) / np.linalg.norm(T,axis=1)**3
    tau = np.einsum('ij,ij->i',np.cross(T,dT), ddT) / np.linalg.norm(np.cross(T,dT),axis=1)**2

    tau = np.abs(tau) ##IF NOT ABS, WILL CHANGE THE DIRECTION
    #print('tau:[min,mean,max]=','%.2g' % np.min(tau),'%.2g' % np.mean(tau),'%.2g' % np.max(tau))

    E3 = np.cross(E1,E2)
    E3 = E3 / np.linalg.norm(E3,axis=1)[:,None]
    d = E1*tau[:,None]+E3*k[:,None]
    d = d / np.linalg.norm(d,axis=1)[:,None]
    
    if is_asy_or_geo:
        "asymptotic: surf_normal = binormal _|_ osculating_normal"
        srfN = E3 ## len==len(ti)
    else:
        "geodesic: surf_normal = osculating_normal"
        srfN = E2    
        
    kg = np.einsum('ij,ij->i',np.cross(srfN,T),dT) / np.linalg.norm(T,axis=1)**3
    kn = np.einsum('ij,ij->i',srfN,dT) / np.linalg.norm(T,axis=1)**2
    
    kg,kn = np.abs(kg), np.abs(kn)
    
    return Pi,[E1,E2,E3],[kg,kn,k,tau,d]
