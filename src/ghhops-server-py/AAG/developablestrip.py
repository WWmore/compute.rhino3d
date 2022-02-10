# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:43:19 2021

@author: WANGH0M
"""

#---------------------------------------------------------------------------
import numpy as np
from scipy import sparse
try:
    from pypardiso import spsolve
except:
    from scipy.sparse.linalg import spsolve
from meshpy import Mesh
#-----------------------------------------------------------------------------
try:
    from constraints_basic import con_fair_midpoint0
except:
    pass
#-----------------------------------------------------------------------------

class StraightStrip(object):

    def __init__(self, an, unitN, E1, ruling, num_seg, width, nmlist,
                 efair=0.005,itera=50,ee=0.001):

        self.name = 'developable_strip'

        self.Vl = an ## left boundary crv points
        
        self.uN = unitN
        
        self.El = E1 ## unit tangent of left boundary-crv
        
        self.Vr = an + unitN * width ## initial right boundary crv points; N is unit
        
        self.ruling = ruling ##len=allnum
        
        self.width = width
        
        self.nmlist = nmlist ## num of each strip's vertices
        
        self.allnum = len(an) ## all num; len(an)==len(N)==len(El1)
        
        self.segnum = num_seg ## a num of segments between two vertices
        
        self._left_edgelength = None
        
        self.id_va=self._id_vb=self.id_vc = None ## left, center, right
        
        self.id_vend0 = self.id_vend1 = None ### 2endpoints of each strip
        
        self._id_vertex = None ### endpoints of each dense-segment

        self._efair = efair
        
        self._itera = itera
        
        self._ee = ee
        
        self._X = None
                
    @property
    def type(self):
        return 'DevelopableStrip'

    @property
    def left_edgelength(self):
        if self._left_edgelength is None:
            self.get_left_edgelength()
        return self._left_edgelength

    @property
    def id_vb(self):
        "prelimanary: len(segment) > 2 "
        if self._id_vb is None:
            self.get_left_edgelength()
        return self._id_vb

    @property
    def id_vertex(self):
        if self._id_vertex is None:
            self.get_left_edgelength()
        return self._id_vertex
    
    # -------------------------------------------------------------------------
    #                            CONSTRAINTS:
    # -------------------------------------------------------------------------
    def get_strip(self,centerline_symmetric=False):
        X = self.optimize()
        self.Vr = X[:3*self.allnum].reshape(-1,3,order='F')
        
        if centerline_symmetric:
            Vl = 1.5*self.Vl-0.5*self.Vr
            Vr = 0.5*self.Vl+0.5*self.Vr
        else:
            Vl,Vr = self.Vl,self.Vr
            
        allV, allF = np.zeros(3), np.zeros(4,dtype=int)
        mmm = 0
        numv = 0
        for num in self.nmlist:
            srr = mmm + np.arange(num)
            P1234 = np.vstack((Vl[srr], Vr[srr]))
            arr1 = numv + np.arange(num)
            arr2 = num + arr1
            flist = np.c_[arr1[:-1],arr1[1:],arr2[1:],arr2[:-1]]
            allV, allF = np.vstack((allV,P1234)), np.vstack((allF,flist))
            numv = len(allV)-1
            mmm += num
        sm = Mesh()
        sm.make_mesh(allV[1:],allF[1:])    
        return sm       
    
    def optimize(self):
        efair,itera,ee= self._efair, self._itera, self._ee
        X = self.initial()
        var = len(X)
        K = self.matrix_fair(var,efair)
        I = sparse.eye(var,format='coo')*ee**2
        n = 0
        #m=3
        opt_num, opt = 100, 100
        while n < itera and (opt_num>1e-6 or opt>1e-6) and opt_num<1e+6:
            H,r,opt = self.con_straight_strip(X,efair,var)
            X = spsolve(H.T*H+K.T*K+I, H.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
            n += 1
            opt_num = np.sum(np.square((H*X)-r))
        print(n, '%.2g' %opt, '%.2g' %opt_num)
        return X
    
    def initial(self):
        """ X = [Vr; Er]
            v4  --- v3
             |       |
             |       |
            v1 ---- v2    
        Vr: right optimized vertices corresponding to Vl;
            (Vl1-Vl0) _|_ (Vr0-Vl0) _|_ (Vr1-Vr0); similar for another end point
        Er: unit_tangent_vector of right boundary curve; len==len(Vr)-2
             using roughly Er3-Er1 // Er at Vr2
        Er:= [Vr_1-Vr_0, Vc-Va, Vr_n-Vr_m]
        """
        Vr = self.Vr.flatten('F')
        # E1_end = self.El[self.id_vertex]
        # E3_end = self.uN[self.id_vertex]
        # E2 = np.cross(E3_end,E1_end)
        # return np.r_[Vr, E2.flatten('F')]
        return Vr

    def con_straight_strip(self, X, efair, var):
        """
        1. right edge-length is given: (Vr_up-Vr_dw)^2 == (Vl_up-Vl_dw)^2
        2. width is constant: (Vr-Vl)^2 == width^2
        3. ortho_width_vector: (Vr-Vl) * El = 0; (Vr-Vl) * Er = 0
                                El:= [Vl_1-Vl_0, El1, Vr_n-Vr_m]    
                                Er:= [Vr_1-Vr_0, Vc-Va, Vr_n-Vr_m]
        4. ruling only at vertices in rectify plane{E1,E3} :
                                ruling * (E3 X E1) = 0
        5. fairness: midpoint 2Vb = Va + Vc
        """
        vb = self.id_vb
        va,vc = self.id_va, self.id_vc
        c_va = np.r_[va, self.allnum+va, 2*self.allnum+va]
        c_vb = np.r_[vb, self.allnum+vb, 2*self.allnum+vb]
        #c_vc = np.r_[vc, self.allnum+vc, 2*self.allnum+vc]
        arr = np.arange(self.allnum)
        c_v = np.r_[arr,self.allnum+arr, 2*self.allnum+arr]
        def _con_edgelength():
            "(Vr_up-Vr_dw)^2 == (Vl_up-Vl_dw)^2"
            r = self.left_edgelength**2
            r += np.linalg.norm((X[c_va]-X[c_vb]).reshape(-1,3,order='F'),axis=1)**2
            col = np.r_[c_va,c_vb]
            data = 2*np.r_[X[c_va]-X[c_vb], X[c_vb]-X[c_va]]
            row = np.tile(np.arange(len(va)),6)
            H = sparse.coo_matrix((data,(row,col)), shape=(len(va), var))
            return H*2, r*2 
        def _con_width():
            "(Vr-Vl)^2 == width^2 <==> Vr^2-2Vl*Vr = -Vl^2+width^2"
            r = self.width**2
            r -= np.linalg.norm(self.Vl,axis=1)**2
            r += np.linalg.norm(X[:3*self.allnum].reshape(-1,3,order='F'),axis=1)**2
            col = c_v
            data = 2*np.r_[X[c_v]-self.Vl.flatten('F')]
            row = np.tile(np.arange(self.allnum),3)
            H = sparse.coo_matrix((data,(row,col)), shape=(self.allnum, var))
            return H*2, r*2 
        def _con_ortho_left():
            "(Vr-Vl) * El = 0 <==> Vr*El = Vl*El"
            r = np.einsum('ij,ij->i',self.Vl, self.El)
            col = c_v
            data = self.El.flatten('F')
            row = np.tile(np.arange(self.allnum),3)
            H = sparse.coo_matrix((data,(row,col)), shape=(self.allnum, var))
            return H, r  
        def _con_ortho_right(id_sub,id_a, id_c):
            """(Vr-Vl) * Er = 0, <=> Vr*(Vc-Va) - Vl*(Vc-Va) = 0
            """
            subVl =  self.Vl[id_sub]
            c_sub = np.r_[id_sub, self.allnum+id_sub, 2*self.allnum+id_sub]
            c_a = np.r_[id_a, self.allnum+id_a, 2*self.allnum+id_a]
            c_c = np.r_[id_c, self.allnum+id_c, 2*self.allnum+id_c]
            Va = X[c_a].reshape(-1,3,order='F')
            Vc = X[c_c].reshape(-1,3,order='F')
            r = np.einsum('ij,ij->i',X[c_sub].reshape(-1,3,order='F'), Vc-Va)
            col = np.r_[c_sub,c_a,c_c]
            d1 = X[c_c]-X[c_a]
            d2 = -X[c_sub]+subVl.flatten('F')
            d3 = X[c_sub]-subVl.flatten('F')
            data = np.r_[d1,d2,d3]
            row = np.tile(np.arange(len(id_sub)),9)
            H = sparse.coo_matrix((data,(row,col)), shape=(len(id_sub), var))
            return H *2, r  *2
        def _con_ruling():
            "ruling * (E3 X E1) = 0; given E1, ruling, variable E3=(Vr-Vl)"
            "<==>E1xVr*ruling = E1xVl*ruling"
            ind = self.id_vertex
            E1 = self.El[ind]
            Vl = self.Vl[ind]
            ruling = self.ruling[ind]
            r = np.einsum('ij,ij->i',np.cross(E1,Vl),ruling)
            row = np.tile(np.arange(len(ind)), 3)
            col = np.r_[ind,self.allnum+ind, 2*self.allnum+ind]
            d1 = ruling[:,1]*E1[:,2]-ruling[:,2]*E1[:,1]
            d2 = ruling[:,2]*E1[:,0]-ruling[:,0]*E1[:,2]
            d3 = ruling[:,0]*E1[:,1]-ruling[:,1]*E1[:,0]
            data = np.r_[d1,d2,d3]
            H = sparse.coo_matrix((data,(row,col)), shape=(len(ind), var))
            ##print(np.sum(np.square((H*X)-r)))
            return H*10,r*10
        H1,r1 = _con_edgelength()
        H2,r2 = _con_width()
        H3,r3 = _con_ortho_left()
        H4,r4 = _con_ortho_right(vb,va,vc)
        H5,r5 = _con_ortho_right(self.id_vend0,self.id_vend0,self.id_vend0+1)
        H6,r6 = _con_ortho_right(self.id_vend1,self.id_vend1,self.id_vend1-1)
        H7,r7 = _con_ruling()
        H = sparse.vstack((H1,H2,H3,H4,H5,H6,H7))
        r = np.r_[r1,r2,r3,r4,r5,r6,r7]
        opt = np.sum(np.square((H*X)-r))
        return H,r,opt

    def matrix_fair(self,var,efair):
        """midpoint: 2Q2 = Q1+Q3;
        """
        vb = self.id_vb
        va,vc = self.id_va, self.id_vc
        c_va = np.r_[va, self.allnum+va, 2*self.allnum+va]
        c_vb = np.r_[vb, self.allnum+vb, 2*self.allnum+vb]
        c_vc = np.r_[vc, self.allnum+vc, 2*self.allnum+vc]
        K = con_fair_midpoint0(c_vb,c_va,c_vc,var)
        return efair * K
    
        
    def get_left_edgelength(self):
        vb=vend0=vend1= np.array([],dtype=int)
        arr = np.array([],dtype=int)
        mmm = 0
        for num in self.nmlist:
            srr = mmm + np.arange(num)
            vend0 = np.r_[vend0, mmm]
            vend1 = np.r_[vend1, mmm+num-1]
            vb = np.r_[vb, srr[1:-1]]
            arr = np.r_[arr, np.where(srr%self.segnum==0)[0]]
            mmm += num
                            
        self._id_vb = vb    
        self.id_va = vb-1
        self.id_vc = vb+1
        self.id_vend0 = vend0
        self.id_vend1 = vend1
        self._id_vertex = arr
        self._left_edgelength = np.linalg.norm(self.Vl[vb-1]-self.Vl[vb],axis=1)     

    ### copied from quadring.py:
    def get_strip_from_rulings(self,an,ruling,row_list):
        "AG-NET: rectifying(tangent) planes along asymptotic(geodesic) crv."
        allV, allF = np.zeros(3), np.zeros(4,dtype=int)
        mmm = 0
        numv = 0
        for num in row_list:
            srr = mmm + np.arange(num)
            P1234 = np.vstack((an[srr], an[srr]+ruling[srr]))
            arr1 = numv + np.arange(num)
            arr2 = num + arr1
            flist = np.c_[arr1[:-1],arr1[1:],arr2[1:],arr2[:-1]]
            allV, allF = np.vstack((allV,P1234)), np.vstack((allF,flist))
            numv = len(allV)-1
            mmm += num
        sm = Mesh()
        sm.make_mesh(allV[1:],allF[1:])    
        return sm    