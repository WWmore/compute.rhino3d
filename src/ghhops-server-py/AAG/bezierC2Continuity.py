# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:33:37 2021

@author: wangh0m
"""

#---------------------------------------------------------------------------
import numpy as np

#import time

from scipy import sparse

try:
    from pypardiso import spsolve
except:
    from scipy.sparse.linalg import spsolve

from constraints_basic import con_edge, con_unit, con_fair_midpoint0,\
                                  con_fair_midpoint2
from bezierCurve import bezier_curve, bezier_curvature_atequalparameters
from smooth import fair_vertices_or_vectors

#-----------------------------------------------------------------------------

class BezierSpline(object):

    def __init__(self, degree=5, continuity=2,efair=0.005,itera=50,ee=0.001,
                 endpoints=None,midpoint=None,tangents=None,normals=None,curvatures=None):

        self.name = 'bezier_spline'

        self._degree = degree
        
        self._continuity = continuity

        self._endpoints = endpoints # only from inner vertices of valence 4
        
        self._midpoint = midpoint # Bezier crv interpolating these at t=0.5 =#(endpoints)-1

        self._tangents = tangents # =#(endpoints) ### E1 of Frenet (E1,E2,E3)

        self._normals = normals # =#(endpoints) ### binormal E3 of Frenet (E1,E2,E3)

        self._curvatures = curvatures # =#(endpoints)

        self._efair = efair
        
        self._itera = itera
        
        self._ee = ee
        
        self.sampling = 200
        
        self._inner_points = None
        

    def __str__(self):
        name = 'Bezier-spline curve: '
        info = 'degree = {}, number of points = {}'
        out = name + info.format(self.d, self.n)
        return out

    @property
    def type(self):
        return 'BezierSpline'
        
    @property
    def d(self):
        return int(self._degree)        

    @property
    def n(self):
        return self._endpoints.shape[0] #=len(tangents)=len(normals)=len(curvatures)        

    @property
    def endpoints_lr(self):
        if self.n==2:
            return [self._endpoints[0],self._endpoints[1]]
        else:
            return [self._endpoints[:-1],self._endpoints[1:]]

    @property
    def tangents_lr(self):
        if self.n==2:
            return [self._tangents[0],self._tangents[1]]
        else:
            return [self._tangents[:-1],self._tangents[1:]]

    @property
    def normals_lr(self):
        if self.n==2:
            return [self._normals[0],self._normals[1]]
        else:
            return [self._normals[:-1],self._normals[1:]]
    
    @property
    def curvatures_lr(self):
        if self.n==2:
            return [self._curvatures[0],self._curvatures[1]]
        else:
            return [self._curvatures[:-1],self._curvatures[1:]]

    @property
    def inner_points(self):
        if self._inner_points is None:
            self.optimization()
        return self._inner_points
# -------------------------------------------------------------------------
#                  optimization :            
# -------------------------------------------------------------------------
    def initial(self):
        """
        from given Pi,ti,Ni,ki, get initial-control-points:
        between endpoints Pi, there are 4 kinds of new points to be decided.
        set (1st-segment): Q0=P0, Q5=P1; Q1=la*t0, Q4=mu*t1;
                           Q2=(Q4-Q1)/3; Q3=2(Q4-Q1)/3; 
        variables X=[Q1,Q2,Q3,Q4]+ [a0,a1] + [t12,t34,l12,l34]
        """
        Pi,Ti = self._endpoints, self._tangents
        if self.n==2:
            Qj0, Qj5 = Pi[0], Pi[1]
            lj = np.linalg.norm(Qj5-Qj0)
            Qj1,Qj4 = Qj0 + Ti[0]*lj/5, Qj5 - Ti[1]*lj/5
            Qj2,Qj3 = (Qj4-Qj1)/3, (Qj4-Qj1)*2/3
            a0 = np.sqrt(np.abs(np.dot(Qj1-Qj0,Ti[0])))
            a1 = np.sqrt(np.abs(np.dot(Qj5-Qj4,Ti[1])))
            X = np.r_[Qj1,Qj2,Qj3,Qj4, a0, a1]  
            if self._curvatures is not None:
                " X += [t12,t34,l12,l34]"
                l12 = np.linalg.norm(Qj1-Qj2)
                l34 = np.linalg.norm(Qj3-Qj4)
                t12 = (Qj1-Qj2) / l12
                t34 = (Qj3-Qj4) / l34
                X = np.r_[X,t12, t34, l12, l34]            
        else:
            Qj0, Qj5 = Pi[:-1], Pi[1:]
            lj = np.linalg.norm(Qj5-Qj0,axis=1)
            Qj1 = Qj0 + Ti[:-1]*lj[:,None]/5
            Qj4 = Qj5 - Ti[1:]*lj[:,None]/5
            Qj2, Qj3 = (Qj4-Qj1)/3, (Qj4-Qj1)*2/3
            X = np.r_[Qj1.flatten('F'),Qj2.flatten('F'),Qj3.flatten('F'),Qj4.flatten('F')]
            a0 = np.sqrt(np.abs(np.einsum('ij,ij->i',Qj1-Qj0,Ti[:-1])))
            a1 = np.sqrt(np.abs(np.einsum('ij,ij->i',Qj5-Qj4,Ti[1:])))
            X = np.r_[X, a0, a1]
            if self._curvatures is not None:
                " X += [t12,t34,l12,l34]"
                l12 = np.linalg.norm(Qj1-Qj2,axis=1)
                l34 = np.linalg.norm(Qj3-Qj4,axis=1)
                t12 = (Qj1-Qj2) / l12[:,None]
                t34 = (Qj3-Qj4) / l34[:,None]
                X = np.r_[X,t12.flatten('F'), t34.flatten('F'),l12,l34]
        return X

    def optimization(self):
        ##start_time = time.time()

        efair,itera,ee= self._efair, self._itera, self._ee
        num = self.n-1
        X = self.initial()
        var = len(X)
        K = self.matrix_fair(num,var,efair)
        I = sparse.eye(var,format='coo')*ee**2
        n = 0
        opt_num, opt = 100, 100
        ### choose which one
        while n < itera and (opt_num>1e-6 or opt>1e-4) and opt_num<1e+6:
        #while n < itera and opt_num>1e-6 and opt_num<1e+6:
            #efair=efair/3 if n>int(itera/2) else efair
            H,r,opt = self.con_interpolation(X,efair)
            X = spsolve(H.T*H+K.T*K+I, H.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
            n += 1
            opt_num = np.sum(np.square((H*X)-r))
            #print('Opt Bezier-spline:',n, '%.2g' %opt, '%.2g' %opt_num)#'%.2g' %opt_num,
            #print('Opt Bezier-spline:',n, '%.2g' %opt_num)#'%.2g' %opt_num,
        
        #print(n, '%.2g' %opt, '%.2g' %opt_num, '%.2g s' %(time.time() - start_time))
        print(n, '%.2g' %opt)
               
        Q1 = X[:3*num].reshape(-1,3,order='F')
        Q2 = X[3*num:6*num].reshape(-1,3,order='F')
        Q3 = X[6*num:9*num].reshape(-1,3,order='F')
        Q4 = X[9*num:12*num].reshape(-1,3,order='F')
        self._inner_points = [Q1,Q2,Q3,Q4]
        return [Q1,Q2,Q3,Q4]
    
    def con_interpolation(self,X, efair):
        """ variables X=[Q1,Q2,Q3,Q4] + [|Q2-Q1|,|Q4-Q3|]
        1. C1: (Q1-Q0) x Tl = 0; (Q5-Q4) x Tr = 0;
        2. (Q1-Q0) * Tl = a0^2; (Q5-Q4) * Tr = a1^2;
        3. C1: Q4+Q1 = 2*Q5=2*Q0
        4. C2: Nl*(Q1-Q0)=Nl*(Q2-Q1)=0; Nr*(Q5-Q4)=Nr*(Q4-Q3)=0; 
        5. C2 (innner): Q15-2Q14+Q13=Q20-2Q21+Q22
        6. C2 (given k): kl = 4/5 |(Q1-Q0) x (Q2-Q1)|/|Q1-Q0|^3
               kr = 4/5 |(Q5-Q4) x (Q4-Q3)|/|Q4-Q3|^3
          <==> (Q1-Q0)^2 - 4sinl/(5kl) * |Q2-Q1| = 0,
               (Q5-Q4)^2 - 4sinr/(5kr) * |Q4-Q3| = 0
        7. C3: torsion continuity (switch)
            P-3Q4+3Q3-Q2 = P3-3P2+3P1-P
        """   
        var = len(X)
        num = self.n-1
        arr = np.arange(3*num)
        c_q1,c_q2,c_q3,c_q4 = arr,arr+3*num,arr+6*num,arr+9*num
        
        pl,pr = self.endpoints_lr
        tl,tr = self.tangents_lr
        nl,nr = self.normals_lr
        c_a0, c_a1 = np.arange(num)+12*num,np.arange(num)+13*num
        
        if self._curvatures is not None:
            kl,kr = self.curvatures_lr
            c_t12, c_t34 = arr+14*num, arr+17*num
            c_l12, c_l34 = np.arange(num)+20*num, np.arange(num)+21*num
        
            if self.n==2:
                crssl = np.cross((X[c_q2]-X[c_q1]),X[c_q1]-pl)
                sinl = np.linalg.norm(crssl)
                sinl = sinl / np.linalg.norm((X[c_q2]-X[c_q1]))
                sinl = sinl / np.linalg.norm(X[c_q1]-pl)
                
                crssr = np.cross((X[c_q4]-X[c_q3]),X[c_q4]-pr)
                sinr = np.linalg.norm(crssr)
                sinr = sinr / np.linalg.norm((X[c_q4]-X[c_q3]))
                sinr = sinr / np.linalg.norm(X[c_q4]-pr)
            else:    
                crssl = np.cross((X[c_q2]-X[c_q1]).reshape(-1,3,order='F'),X[c_q1].reshape(-1,3,order='F')-pl)
                sinl = np.linalg.norm(crssl, axis=1)
                sinl = sinl / np.linalg.norm((X[c_q2]-X[c_q1]).reshape(-1,3,order='F'), axis=1)
                sinl = sinl / np.linalg.norm(X[c_q1].reshape(-1,3,order='F')-pl, axis=1)
                
                crssr = np.cross((X[c_q4]-X[c_q3]).reshape(-1,3,order='F'),X[c_q4].reshape(-1,3,order='F')-pr)
                sinr = np.linalg.norm(crssr, axis=1)
                sinr = sinr / np.linalg.norm((X[c_q4]-X[c_q3]).reshape(-1,3,order='F'), axis=1)
                sinr = sinr / np.linalg.norm(X[c_q4].reshape(-1,3,order='F')-pr, axis=1)
            
        def _con_c1_base(pl,tl,c_q1):
            "(Q1-Q0) x Tl = 0 <==> Q1 x Tl = Q0 x Tl"
            "(cy-bz,az-cx,bx-ay)=(d,e,f)"
            r = np.cross(pl,tl).flatten('F')
            x,y,z = c_q1[:num], c_q1[num:2*num], c_q1[2*num:]
            a,b,c = tl.T
            col = np.r_[y,z,z,x,x,y]
            data = np.r_[c,-b,a,-c,b,-a]
            row1 = np.tile(np.arange(num),2)
            row2 = np.tile(np.arange(num)+num,2)
            row3 = np.tile(np.arange(num)+2*num,2)
            row = np.r_[row1,row2,row3]
            H = sparse.coo_matrix((data,(row,col)), shape=(3*num, var))
            return H,r
        
        def _con_c1(pl,c_q1,c_q4):
            """ 
            Ql4+Qr1 = 2*Ql5(==Qr0==pl)
            """
            if self.n<=2 or self._midpoint is not None and self.n<=3:
                null = np.zeros([0])
                H = sparse.coo_matrix((null,(null,null)), shape=(0,var))
                r = np.array([])
            else:          
                arr = np.arange(num-1)
                r = 2*pl[1:].flatten('F')
                row = np.tile(np.arange(3*num-3),2)
                c_q14 = c_q4[np.r_[arr,arr+num,arr+2*num]]
                c_q21 = c_q1[np.r_[arr+1,arr+num+1,arr+2*num+1]]
                col = np.r_[c_q14,c_q21]
                #print(self.n,num)
                data = np.ones(2*3*(num-1))
                H = sparse.coo_matrix((data,(row,col)), shape=(3*num-3, var))
            return H,r
        
        def _con_c1_2(pl,tl,c_q1,c_a0): # relation with given tangents
            "(Q1-Q0)*Tl=a0^2 <==> a0^2-Q1*Tl=-Q0*Tl"
            r = X[c_a0]**2
            if self.n==2:
                r -= np.dot(pl,tl)
            else:
                r -= np.einsum('ij,ij->i', pl,tl)
            row = np.tile(np.arange(num),4)
            col = np.r_[c_a0,c_q1]
            data = np.r_[X[c_a0], -tl.flatten('F')]
            H = sparse.coo_matrix((data,(row,col)), shape=(num, var))
            return H,r

        def _con_c2_1(pl,nl,c_q1): ##replaced by below
            "Nl*(Q1-Q0)=Nl*(Q2-Q0)=0 <==> Nl*Q1=Nl*Q0,Nl*Q2=Nl*Q0"
            if self.n==2:
                r = np.array([np.dot(pl,nl)])
            else:
                r = np.einsum('ij,ij->i', pl,nl)
            row = np.tile(np.arange(num),3)
            col = c_q1
            data = nl.flatten('F')
            H = sparse.coo_matrix((data,(row,col)), shape=(num, var))
            return H,r
        def _con_c2_11(pl,nl,c_q1,c_q2):
            "replace above as : Nl*(Q2-2Q1+Q0)=0 <==> Nl*Q2-2Nl*Q1= -Nl*Q0"
            if self.n==2:
                r = -np.array([np.dot(pl,nl)])
            else:
                r = -np.einsum('ij,ij->i', pl,nl)
            row = np.tile(np.arange(num),6)
            col = np.r_[c_q2,c_q1]
            data = np.r_[nl.flatten('F'),-2*nl.flatten('F')]
            H = sparse.coo_matrix((data,(row,col)), shape=(num, var))
            return H,r
        
        def _con_c2_2(c_q1,c_q2,c_q3,c_q4):
            "(innner): Q15-2Q14+Q13=Q20-2Q21+Q22; Q15-Q20=0"
            "2(Q14-Q21) = Q13-Q22 <==> Q13 - 2*Q14 + 2*Q21 - Q22=0"
            if self._midpoint is None and self.n<=2 or self._midpoint is not None and self.n<=3:
                null = np.zeros([0])
                H = sparse.coo_matrix((null,(null,null)), shape=(0,var))
                r = np.array([])
            else:                    
                arr = np.arange(num-1)
                r = np.zeros(3*num-3)
                row = np.tile(np.arange(3*num-3),4)
                c_q13 = c_q3[np.r_[arr,arr+num,arr+2*num]]
                c_q14 = c_q4[np.r_[arr,arr+num,arr+2*num]]
                c_q21 = c_q1[np.r_[arr+1,arr+num+1,arr+2*num+1]]
                c_q22 = c_q2[np.r_[arr+1,arr+num+1,arr+2*num+1]]
                col = np.r_[c_q13, c_q14, c_q21, c_q22]
                ones = np.ones(3*(num-1))
                data = np.r_[ones,-2*ones,2*ones,-ones]
                H = sparse.coo_matrix((data,(row,col)), shape=(3*(num-1), var))
            return H,r
        def _con_c2_3(pl,kl,c_q1,c_q2,c_t12,c_l12,sinl):
            "(Q1-Q0)^2 - 4sinl/(5kl) * |Q2-Q1| = 0"
            "Q1^2-2*Q1*Q0 - 4sinl/(5kl) * |Q2-Q1| = - Q0^2"
            if self.n==2:
                r = np.linalg.norm(X[c_q1])**2-np.linalg.norm(pl)**2
            else:
                r = np.linalg.norm(X[c_q1].reshape(-1,3,order='F'),axis=1)**2
                r -= np.linalg.norm(pl,axis=1)**2
            col = np.r_[c_q1, c_l12]
            data = np.r_[2*X[c_q1]-2*pl.flatten('F'), -4/5/kl*sinl]
            row = np.tile(np.arange(num),4)
            H = sparse.coo_matrix((data,(row,col)), shape=(num, var))
            H1,r1 = con_edge(X,c_q1,c_q2,c_l12,c_t12,num,var)
            H2,r2 = con_unit(X,c_t12,num,var)
            H = sparse.vstack((H,H1,H2))
            r = np.r_[r,r1,r2]
            return H,r
        def _con_c3(pl,c_q1,c_q2,c_q3,c_q4,is_strong=True):
            """ Q0-Q1-Q2-Q3-Q4-Q5==P==P0-P1-P2-P3-P4-P5
            Q10-Q11-Q12-Q13-Q14-Q15==P==Q20-Q21-Q22-Q23-Q24-Q25
            P-3Q4+3Q3-Q2 = P3-3P2+3P1-P
            <==>Q23+Q12 -3Q22-3Q13 +3Q21+3Q14 = 2P
            """
            if self._midpoint is None and self.n<=2 or self._midpoint is not None and self.n<=3:
                null = np.zeros([0])
                H = sparse.coo_matrix((null,(null,null)), shape=(0,var))
                r = np.array([])
            else:                    
                arr = np.arange(num-1)
                r = 2*pl[1:].flatten('F')
                row = np.tile(np.arange(3*num-3),6)
                c_q12 = c_q2[np.r_[arr,arr+num,arr+2*num]]
                c_q13 = c_q3[np.r_[arr,arr+num,arr+2*num]]
                c_q14 = c_q4[np.r_[arr,arr+num,arr+2*num]]
                c_q21 = c_q1[np.r_[arr+1,arr+num+1,arr+2*num+1]]
                c_q22 = c_q2[np.r_[arr+1,arr+num+1,arr+2*num+1]]
                c_q23 = c_q3[np.r_[arr+1,arr+num+1,arr+2*num+1]]
                col = np.r_[c_q23, c_q12, c_q22, c_q13, c_q21,c_q14]
                ones = np.ones(3*(num-1))
                data = np.r_[ones,ones,-3*ones,-3*ones,3*ones,3*ones]
                H = sparse.coo_matrix((data,(row,col)), shape=(3*(num-1), var))
                if is_strong:
                    """tau0==tau1:
                    dP = 5*(P1-P0)
                    ddP = 20*(P2-2*P1+P0)
                    dddP = 60*(P3-3*P2+3*P1-P0)
                    tau = np.einsum('ij,ij->i',np.cross(dP,ddP),dddP) 
                        / np.linalg.norm(np.cross(dP,ddP),axis=1)**2  
                    <==> A0*dddP0 = A1*dddP1; A0=np.cross(dP,ddP)/||^2
                    <==> A1*Q23+A0*Q12 -3A1*Q22-3A0*Q13 +3A1*Q21+3A0*Q14 = (A0+A1)*P
                    """
                    P13 = X[c_q13].reshape(-1,3,order='F')
                    P14 = X[c_q14].reshape(-1,3,order='F')
                    P21 = X[c_q21].reshape(-1,3,order='F')
                    P22 = X[c_q22].reshape(-1,3,order='F')
                    dp0,dp1 = pl[1:]-P14, P21-pl[1:]
                    ddp0,ddp1 = pl[1:]-2*P14+P13, P22-2*P21+pl[1:]
                    dpddp0,dpddp1 = np.cross(dp0,ddp0),np.cross(dp1,ddp1)
                    A0 = dpddp0*(np.linalg.norm(dpddp1,axis=1)**2)[:,None]*10**3 ## NEED TO SET
                    A1 = dpddp1*(np.linalg.norm(dpddp0,axis=1)**2)[:,None]*10**3 ## NEED TO SET
                    #print(A0,A1) ## 1E-6 VERY SMALL
                    col = np.r_[c_q23, c_q12, c_q22, c_q13, c_q21,c_q14]
                    a0,a1 = A0.flatten('F'),A1.flatten('F')
                    data = np.r_[a1,a0,-3*a1,-3*a0,3*a1,3*a0]
                    row = np.tile(np.arange(num-1),18) 
                    Hs = sparse.coo_matrix((data,(row,col)), shape=(num-1, var))
                    rs = np.einsum('ij,ij->i',A0+A1,pl[1:])
                    ###print(np.max(np.abs(rs)))
                    H = sparse.vstack((H,Hs))
                    r = np.r_[r,rs]
            return H,r
        

        def _con_interpolate_midpoint(pl,pr,c_q1,c_q2,c_q3,c_q4):
            """P(t=0.5)==(0.5)^5*[C0*Pl+C1*Q1+C2*Q2+C3*Q3+C4*Q4+C5*Pr]
               midpoint = 0.5^5*[Pl + 5*Q1 + 10*Q2 + 10*Q3 + 5*Q4 + Pr]
               <==>Q1+2*Q2+2*Q3+Q4 = (32*midpoint-Pl-Pr)/5
            """
            col = np.r_[c_q1,c_q2,c_q3,c_q4]
            row = np.tile(np.arange(3*num),4)
            one = np.ones(3*num)
            data = np.r_[one,2*one,2*one,one]
            r = (self._midpoint*32-pl-pr).flatten('F')/5
            H = sparse.coo_matrix((data,(row,col)), shape=(3*num, var))
            return H,r

        H11,r11 = _con_c1_base(pl, tl, c_q1)    
        H12,r12 = _con_c1_base(pr, tr, c_q4)  
        #H13,r13 = _con_c1_2(pl, tl, c_q1, c_a0)    
        #H14,r14 = _con_c1_2(pr, tr, c_q4, c_a1) 
        "1st derivative:"
        H15,r15 = _con_c1(pl,c_q1,c_q4)
        "tangent plane:"
        # H21,r21 = _con_c2_1(pl, nl, c_q1) ###NOTE: true from _con_c1_base
        # H22,r22 = _con_c2_1(pl, nl, c_q2)
        # H23,r23 = _con_c2_1(pr, nr, c_q3)
        # H24,r24 = _con_c2_1(pr, nr, c_q4) ###NOTE: true from _con_c1_base
        Ht0,rt0 = _con_c2_11(pl,nl,c_q1,c_q2)
        Ht1,rt1 = _con_c2_11(pr,nr,c_q4,c_q3)
        ##print('n:', np.sum(np.square((H21*X)-r21)))
        ##print('n:', np.sum(np.square((H22*X)-r22)))
        
        K02,r02 = con_fair_midpoint2(c_q1,c_q2,pl,var,efair)
        K35,r35 = con_fair_midpoint2(c_q4,c_q3,pr,var,efair)
        
        #H = sparse.vstack((H11,H12,H13,H14,H15,H21*10,H22*10,H23*10,H24*10,K02,K35))
        #r = np.r_[r11,r12,r13,r14,r15,r21*10,r22*10,r23*10,r24*10,r02,r35]
        H = sparse.vstack((H11,H12,H15*100,Ht0*10,Ht1*10,K02,K35))
        r = np.r_[r11,r12,r15*100,rt0*10,rt1*10,r02,r35]
        
        if self._midpoint is not None: ###NOTE: if use it, not enough dof
            Hm,rm = _con_interpolate_midpoint(pl,pr,c_q1,c_q2,c_q3,c_q4)
            H = sparse.vstack((H,Hm))
            r = np.r_[r,rm]
        if self._curvatures is not None: ###NOTE: no use now.
            H41,r41 = _con_c2_3(pl,kl,c_q1,c_q2,c_t12,c_l12,sinl)
            H42,r42 = _con_c2_3(pr,kr,c_q3,c_q4,c_t34,c_l34,sinr)
            H = sparse.vstack((H,H41,H42))
            r = np.r_[r,r41,r42]
        if self.n>2:
            "2nd derivative"
            H3, r3 = _con_c2_2(c_q1,c_q2,c_q3,c_q4)
            H = sparse.vstack((H,H3*100))
            r = np.r_[r,r3*100]
        if self._continuity==3:
            "tortion continuity; 3rd derivative"
            H3, r3 = _con_c3(pl,c_q1,c_q2,c_q3,c_q4,is_strong=False)
            H = sparse.vstack((H,H3*100))
            r = np.r_[r,r3*100]
        opt = np.sum(np.square((H*X)-r))
        return H,r,opt
    
    def matrix_fair(self,num,var,efair):
        """midpoint averaging of [Pl,Q1,Q2,Q3,Q4,Pr]:
           ONLY : 2Q2 = Q1+Q3;  2Q3 = Q2+Q4;
           (2Q1 = Q2+Pl; 2Q4=Q3+Pr) INSIDE H,r
        """
        arr = np.arange(3*num)
        c_q1,c_q2,c_q3,c_q4 = arr,arr+3*num,arr+6*num,arr+9*num
        K13 = con_fair_midpoint0(c_q2,c_q1,c_q3,var)
        K24 = con_fair_midpoint0(c_q3,c_q2,c_q4,var)
        K = sparse.vstack((K13,K24))
        return efair * K
    
# -------------------------------------------------------------------------
#                  return points + polygon + curve :            
# -------------------------------------------------------------------------

    def control_points(self,is_points=False,is_polygon=False,is_curve=False,
                       is_seg=False,is_Q1234=False,is_dense=False):
        Q1,Q2,Q3,Q4 = self.inner_points #self.optimization()
        from polyline import Polyline
        if is_Q1234:
            return np.hstack((Q1,Q2,Q3,Q4)).flatten().reshape(-1,3)
        pl,pr = self.endpoints_lr
        if self.n==2:
            pi = np.r_[pl,Q1[0],Q2[0],Q3[0],Q4[0],pr]
        else:
            pi = np.r_[np.hstack((pl,Q1,Q2,Q3,Q4)).flatten(),pr[-1]]
        Pi = pi.reshape(-1,3)
        if is_points:
            if is_seg:
                return Pi, Pi[:-1], Pi[1:]
            return Pi
        elif is_polygon:
            return Polyline(Pi)
        elif is_curve:
            if self.n==2:
                all_pi = bezier_curve(Pi, nTimes=self.sampling)
            else:
                sample_points = np.array([0,0,0])
                for i in range(self.n-1):
                    points = np.vstack((pl[i],Q1[i],Q2[i],Q3[i],Q4[i],pr[i]))
                    "Note: before bug is return bezier-crv list is reversed"
                    cp = bezier_curve(points, nTimes=self.sampling)
                    sample_points = np.vstack((sample_points,cp))
                all_pi = sample_points[1:]
            if is_seg:
                return all_pi, all_pi[:-1], all_pi[1:]
            return Polyline(all_pi,closed=False)

    def get_curvature(self,is_asy_or_geo=True,is_dense=False,num_div=5,
                      is_modify=False,is_smooth=0.0):
        """ from Bezier curves P(t) and  given surface normals n at vertices
            to get  t // P'(t), t' // P'', s = n x t
        if arc-length : k=|P''|, tau = det(P',P'',P''')/k^2, 
        elif general-parameter:
                        k = |p' x p''|/|p'|^3
                        tau = det(p',p'',p''')/|p' x p''|^2
        if asym: binormal == surf_normal
        elif geo: binormal == t x surf_normal
        """
        P1,P2,P3,P4 = self.inner_points
            
        if len(P1)==1:
            P1,P2,P3,P4 = P1[0],P2[0],P3[0],P4[0]
        P0,P5 = self.endpoints_lr
        uT0,uT5 = self.tangents_lr
        
        if is_asy_or_geo:
            "asym: given normal==surf_normal"
            "asymptotic: surf_normal = binormal _|_ osculating_normal"
            N0,N5 = self.normals_lr
            S0,S5 = np.cross(N0,uT0),np.cross(N5,uT5)
        else:
            "geo: given normal==binormal E3"
            "geodesic: surf_normal = osculating_normal"
            S0,S5 = self.normals_lr ##given normal_lr always from binormal E3
            N0,N5 = np.cross(S0,uT0),np.cross(S5,uT5)### here N refer // E2 //surfN

        def _end(P0,P1,P2,P3,N0,sign=1):
            dP = 5*(P1-P0) * sign
            ddP = 20*(P2-2*P1+P0)
            dddP = 60*(P3-3*P2+3*P1-P0) * sign
            k = np.linalg.norm(np.cross(dP,ddP))/np.linalg.norm(dP)**3
            tau = np.dot(np.cross(dP,ddP),dddP) / np.linalg.norm(np.cross(dP,ddP))**2
            E1 = dP / np.linalg.norm(dP)
            E2 = ddP / np.linalg.norm(ddP)
            E3 = np.cross(E1,E2)
            E3 = E3 / np.linalg.norm(E3)
            d = tau*E1+k*E3
            kg = np.dot(np.cross(N0,dP),ddP) / np.linalg.norm(dP)**3
            kn = np.dot(N0,ddP)/np.linalg.norm(dP)**2
            return kg,kn,k,tau,d
        
        def _ends(P0,P1,P2,P3,N0,sign=1):
            dP = 5*(P1-P0) * sign
            ddP = 20*(P2-2*P1+P0)
            dddP = 60*(P3-3*P2+3*P1-P0) * sign
            k = np.linalg.norm(np.cross(dP,ddP),axis=1)/np.linalg.norm(dP,axis=1)**3
            tau = np.einsum('ij,ij->i',np.cross(dP,ddP),dddP) / np.linalg.norm(np.cross(dP,ddP),axis=1)**2
            E1 = dP / np.linalg.norm(dP,axis=1)[:,None]
            E2 = ddP / np.linalg.norm(ddP,axis=1)[:,None]
            E3 = np.cross(E1,E2)
            E3 = E3 / np.linalg.norm(E3,axis=1)[:,None]
            d = E1*tau[:,None]+E3*k[:,None]
            kg = np.einsum('ij,ij->i',np.cross(N0,dP),ddP) / np.linalg.norm(dP,axis=1)**3
            kn = np.einsum('ij,ij->i',N0,ddP)/np.linalg.norm(dP,axis=1)**2
            ##print('%.2g' % np.max(np.abs(kn)))
            kg,kn = np.abs(kg), np.abs(kn)
            return kg,kn,k,tau,d
        
        if self.n==2:
            kg0,kn0,k0,tau0,d0 = _end(P0,P1,P2,P3,N0)
            kg1,kn1,k1,tau1,d1 = _end(P5,P4,P3,P2,N5,sign=-1)
            kg,kn = np.r_[kg0,kg1],np.r_[kn0,kn1]
            k,tau = np.r_[k0,k1], np.r_[tau0,tau1]
            if is_asy_or_geo:
                "asymptotic: binormal // surf-normal"
                if np.dot(N0,d0)<0:
                    d0 = -d0
                if np.dot(N5,d1)<0:
                    d1 = -d1
            else:
                "geodesic: binormal in tangent plane"
                if np.dot(S0,d0)<0:
                    d0 = -d0 
                if np.dot(S5,d1)<0:
                    d1 = -d1 
            d = np.vstack((d0,d1))
            d = d / np.linalg.norm(d,axis=1)[:,None]
        else:
            kg0,kn0,k0,tau0,d0 = _ends(P0,P1,P2,P3,N0)
            kg1,kn1,k1,tau1,d1 = _ends(P5,P4,P3,P2,N5,sign=-1)

            kg = np.r_[kg0[0],(kg0[1:]+kg1[:-1])/2,kg1[-1]]
            kn = np.r_[kn0[0],(kn0[1:]+kn1[:-1])/2,kn1[-1]]
            k = np.r_[k0[0],(k0[1:]+k1[:-1])/2,k1[-1]]
            tau = np.r_[tau0[0],(tau0[1:]+tau1[:-1])/2,tau1[-1]]

            if True:
                "orient binormal with surf-normal"
                if is_asy_or_geo:
                    "asymptotic: binormal // surf-normal"
                    i = np.where(np.einsum('ij,ij->i',N0,d0)<0)[0]
                    d0[i] = -d0[i]
                    j = np.where(np.einsum('ij,ij->i',N5,d1)<0)[0]
                    d1[j] = -d1[j]
                else: ## if comment, has big influence: along e1
                    "geodesic: binormal in tangent plane"
                    i = np.where(np.einsum('ij,ij->i',S0,d0)<0)[0]
                    d0[i] = -d0[i]
                    j = np.where(np.einsum('ij,ij->i',S5,d1)<0)[0]
                    d1[j] = -d1[j]
            
            #print(np.max(np.linalg.norm(d0[1:]-d1[:-1],axis=1)))
            #d = np.vstack((d0[0], d1[:-1], d1[-1]))
            d = np.vstack((d0[0], (d0[1:]+d1[:-1])/2, d1[-1]))
            d = d / np.linalg.norm(d,axis=1)[:,None]
            
        if is_dense:
            "get more curvature-data at several-parameters ti between endpoints"
            if self.n==2: ###NOTE DIDN'T CARE ABOUT THE IS_MODIFY + IS_SMOOTH, MAY HAS PROBLEM
                ctrlP = np.vstack((P0,P1,P2,P3,P4,P5))
                Pi,frmi,crvture = bezier_curvature_atequalparameters(ctrlP,
                                                                     is_asy_or_geo,
                                                                     num_div)
                E1,E2,E3 = frmi
                kgi,kni,ki,taui,di = crvture

                if is_asy_or_geo:
                    "reverse asymptotic direction"
                    if np.dot(N0,di[0])<0 and np.dot(N5,di[-1])<0:
                        di = -di
                    if True:
                        "reorient Frenet E3"
                        if np.dot(N0,E3[0])<0 and np.dot(N5,E3[-1])<0:
                            E3 = -E3
                else:
                    "geodesic"
                    if np.dot(S0,di[0])<0 and np.dot(S5,di[-1])<0:
                        di = -di
                        
                #--------------------------        
                if is_modify:
                    sign = 1
                    if is_asy_or_geo:
                        if np.dot(N0,di[0])<0 and np.dot(N5,di[-1])<0:
                            sign=-1    
                    di = di*sign
                    cos = np.abs(np.einsum('ij,ij->i',E1,di))    
                    #i = np.where(cos>np.mean(cos))[0]#
                    i = np.where(cos>0.8)[0]
                    di[i] = E3[i]
                if is_smooth:
                    weight=is_smooth ##NOTE: HERE IS AN INPUT NUMBER
                    di = fair_vertices_or_vectors(di,itera=10,efair=weight*1,is_fix=True)
                    di = di / np.linalg.norm(di,axis=1)[:,None]
                    E3 = fair_vertices_or_vectors(E3,itera=10,efair=weight*1,is_fix=True)
                    E3 = E3 / np.linalg.norm(E3,axis=1)[:,None]
                    E2 = np.cross(E3,E1) / np.linalg.norm(np.cross(E3,E1),axis=1)[:,None]
                #--------------------------    

                return kgi,kni,ki,taui,Pi,di,[E1,E2,E3]

            else:
                allkg=allkn =allk=alltau= np.array([])
                allP=allD = np.array([0,0,0])
                allE1=allE2=allE3 = np.array([0,0,0])
                for i in range(self.n-1):
                    "append for each line-segment"
                    ctrlP = np.vstack((P0[i],P1[i],P2[i],P3[i],P4[i],P5[i]))
                    Pi,frmi,crvture = bezier_curvature_atequalparameters(ctrlP,
                                                                         is_asy_or_geo,
                                                                         num_div)
                    E1,E2,E3 = frmi ###NOTE:E3_|_E1,E2, but E2 maynot _|_ E1
                    kgi,kni,ki,taui,di = crvture
                    
                    #--------------------------
                    if is_asy_or_geo:
                        "reverse asymptotic direction"
                        sign=1
                        if np.dot(N0[i],di[0])<0 and np.dot(N5[i],di[-1])<0:
                            sign=-1
                        di = di*sign
                        for ij,d in enumerate(di):
                            if np.dot(d,N0[i])<0 and np.dot(N5[i],d)<0:
                                di[ij] = -d  
                                
                        if True:
                            "reorient Frenet E3"
                            sign=1
                            if np.dot(N0[i],E3[0])<0 and np.dot(N5[i],E3[-1])<0:
                                sign=-1
                            E3 = E3*sign
                            for ij,e3 in enumerate(E3):
                                if np.dot(e3,N0[i])<0 and np.dot(N5[i],e3)<0:
                                    E3[ij] = -e3  

                        if is_modify:
                            if False: ###NOTE OPT FOR RUNLINGS
                                cos = np.abs(np.einsum('ij,ij->i',E1,di))    
                                #j = np.where(cos>np.mean(cos))[0]
                                j = np.where(cos>0.8)[0]
                                one = np.ones(len(j))
                                jk = np.where(np.einsum('ij,ij->i',di[j],E3[j])<0)[0]
                                one[jk] = -1
                                di[j] = E3[j] * one[:,None]
                            else: ###NOTE OPT FOR RUNLINGS
                                # if np.dot(N0[i],di[0])<0.9:##note to choose
                                #     di[0] = N0[i]
                                # if np.dot(N5[i],di[-1])<0.9:##note to choose
                                #     di[-1] = N5[i]
                                di[0],di[-1] = N0[i],N5[i]
                    else:
                        "reverse geodesic direction; given S//E3, N//E2(not_|_E1)"
                        sign=1
                        if np.dot(S0[i],di[0])<0 and np.dot(S5[i],di[-1])<0:
                            sign=-1
                        di = di*sign
                        for ij,d in enumerate(di):
                            if np.dot(d,S0[i])<0 and np.dot(S5[i],d)<0:
                                di[ij] = -d  
                                
                        if True:
                            "reorient Frenet E3 (should not E2)"
                            sign=1
                            if np.dot(S0[i],E3[0])<0 and np.dot(S5[i],E3[-1])<0:
                                sign=-1
                            E3 = E3*sign
                            for ij,e3 in enumerate(E3):
                                if np.dot(e3,S0[i])<0 and np.dot(S5[i],e3)<0:
                                    E3[ij] = -e3
                            E2 = np.cross(E3,E1)
                            E2 = E2 / np.linalg.norm(E2,axis=1)[:,None]
                        if is_modify:###NOTE OPT FOR RUNLINGS
                            # if np.dot(S0[i],di[0])<0.9:##note to choose
                            #     di[0] = S0[i]
                            # if np.dot(S5[i],di[-1])<0.9:##note to choose
                            #     di[-1] = S5[i]
                            di[0],di[-1] = S0[i],S5[i]
                    #--------------------------
                    
                    #--------------------------
                    if is_smooth:
                        weight = is_smooth
                        di = fair_vertices_or_vectors(di,itera=10,efair=weight*1,is_fix=True)
                        di = di / np.linalg.norm(di,axis=1)[:,None]
                        E3 = fair_vertices_or_vectors(E3,itera=10,efair=weight*1,is_fix=True)
                        E3 = E3 / np.linalg.norm(E3,axis=1)[:,None]
                        E2 = np.cross(E3,E1) / np.linalg.norm(np.cross(E3,E1),axis=1)[:,None]
                    #--------------------------
                    allkg = np.r_[allkg,kgi[:-1]]
                    allkn = np.r_[allkn,kni[:-1]]
                    allk = np.r_[allk,ki[:-1]]
                    alltau = np.r_[alltau,taui[:-1]]
                    allP = np.vstack((allP,Pi[:-1]))
                    allE1 = np.vstack((allE1,E1[:-1]))
                    allE2 = np.vstack((allE2,E2[:-1]))
                    allE3 = np.vstack((allE3,E3[:-1]))
                    allD = np.vstack((allD,di[:-1]))

                allkg = np.r_[allkg,kgi[-1]]
                allkn = np.r_[allkn,kni[-1]]
                allk = np.r_[allk,ki[-1]]
                alltau = np.r_[alltau,taui[-1]]
                allP = np.vstack((allP,Pi[-1]))
                allE1 = np.vstack((allE1,E1[-1]))
                allE2 = np.vstack((allE2,E2[-1]))
                allE3 = np.vstack((allE3,E3[-1]))
                allD = np.vstack((allD,di[-1]))
                return allkg,allkn,allk,alltau,allP[1:],allD[1:],[allE1[1:],allE2[1:],allE3[1:]]
        else:
            "only at endpoints"
            return kg,kn,k,tau,d    