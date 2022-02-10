#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np
# -----------------------------------------------------------------------------

'''Frenet frame.py: (t,n,b) from any given three lists of points (v,v1,v3)
    l1 = (v1-v)^2; l3 = (v3-v)^2
   tangent t:= (v1-v)*l3**2-(v3-v)*l1**2; t=t/|t|, (in osculating plane)
   binormal b:= (v1-v) x (v3-v); b=b/|b|,          (orthogonal to osc.plane)
   principal normal n:= b x t.                     (in osculating plane)
   
   inscribecircle center & radius
'''

__author__ = 'Hui'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                   Frame
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class FrenetFrame(object):

    def __init__(self, v=[0,0,0],v1=[1,0,0],v3=[0,1,0]):

        self.v = v # CENTER

        self.v1 = v1 # LEFT

        self.v3 = v3 # RIGHT
        
        # self.t = [1,-1,0]
        
        # self.b = [0,0,1]
        
        # self.n = [1,1,0]
        
        # self.center = [0.5,0.5,0]
        
        # self.radius = np.sqrt(2)/2

    @property
    def N(self):
        return self._v.shape[0]

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        v = np.array(v, dtype=np.float)
        if len(v.shape) == 1:
            v = np.array([v])
        if v.shape[1] != 3:
            raise ValueError('the vertex v must be a (n,3) array')
        self._v = v

    @property
    def v1(self):
        return self._v1

    @v1.setter
    def v1(self, v1):
        v1 = np.array(v1, dtype=np.float)
        if len(v1.shape) == 1:
            v1 = np.array([v1])
        if v1.shape[1] != 3:
            raise ValueError('the vertex v1 must be a (n,3) array')
        if v1.shape[0] != self.N and v1.shape[0] == 1:
            v1 = np.tile(v1, (self.N, 1))
        elif v1.shape[0] != self.N:
            raise ValueError('dimesnsion mismatch')
        self._v1 = v1

    @property
    def v3(self):
        return self._v3

    @v3.setter
    def v3(self, v3):
        v3 = np.array(v3, dtype=np.float)
        if len(v3.shape) == 1:
            v3 = np.array([v3])
        if v3.shape[1] != 3:
            raise ValueError('the vertex v3 must be a (n,3) array')
        if v3.shape[0] != self.N and v3.shape[0] == 1:
            v3 = np.tile(v3, (self.N, 1))
        elif v3.shape[0] != self.N:
            raise ValueError('dimesnsion mismatch')
        self._v3 = v3

    @property
    def t(self):
        V0,V1,V3 = self.v, self.v1, self.v3
        l1 = np.linalg.norm(V1-V0,axis=1)
        l3 = np.linalg.norm(V3-V0,axis=1)
        t1 = (V3-V0)*(l1**2)[:,None] - (V1-V0)*(l3**2)[:,None]
        t1 = t1 / np.linalg.norm(t1,axis=1)[:,None]
        return t1

    @property
    def b(self):
        V0,V1,V3 = self.v, self.v1, self.v3
        b1 = np.cross(V1-V0, V3-V0)
        b1 = b1 / np.linalg.norm(b1,axis=1)[:,None]
        return b1
    
    @property
    def n(self):
        n1 = np.cross(self.b, self.t)
        n1 = n1 / np.linalg.norm(n1,axis=1)[:,None]
        return n1
    
    @property 
    def radius(self):
        if False:
            "(v+r*n-v1)^2 = (v+r*n-v3)^2 <==> r=(v1^2-v3^2+2vv3-2vv1)/(2(v1-v3)*n)"
            r = np.linalg.norm(self.v1,axis=1)**2 - np.linalg.norm(self.v3,axis=1)**2
            r += 2*np.einsum('ij,ij->',self.v3-self.v1,self.v)
            r /= 2*np.einsum('ij,ij->',self.v1-self.v3,self.n)
        else:
            "(v+r*n-v1)^2=r^2<==>r = (v-v1)^2/[2n*(v1-v)]"
            r = np.linalg.norm(self.v-self.v1,axis=1)**2 
            r /= 2*np.einsum('ij,ij->',self.v1-self.v,self.n)
        return r
    
    @property
    def center(self):
        return self.v + self.radius * self.n


    def frame(self):
        return (self.t,self.n,self.b)