# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:00:39 2021

@author: wangh0m
"""

#---------------------------------------------------------------------------
import numpy as np
#import time
from scipy import sparse
from scipy.optimize import fsolve#,root
from meshpy import Mesh
#-----------------------------------------------------------------------------

def unroll_multiple_strips(sm,num_list,dist=None,step=1.2,
                           coo=2,anchor=0,efair=0.005,
                           ee=0.001,itera=10,is_midaxis=False,
                           w_straight=0):
    """refer meshFunction.py/unfoldAllGeodesicStrips; based on unroll_1_strip
    Parameters
    ----------
    Vall: stack of each Vi of strips Vi=[Varr1;Varr2]
    anchor : one number of starting coordinates on x/y/z-axis
    step : control the distance between different strips
    coo : 0,1,2 ~ plane:y-z / x-z / x-y
    """
    #import time
    #start_time = time.time()

    Vall = sm.vertices
    allv = np.array([0,0,0])
    allf = np.array([0,0,0,0],dtype=int)
    move = 0
    istart = 0
    for num in num_list:
        Vi = Vall[istart:istart + 2*num]
        _,vlist,flist = unroll_1_strip(Vi,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
        if dist is None:
            v1,v2 = vlist[0], vlist[num]
            dist = np.linalg.norm(v2-v1)
        anchor += step*dist##NOTE: if +=step:const_controled_dist, if +=step*l, auto-dist
        allv = np.vstack((allv,vlist))
        allf = np.vstack((allf,flist+istart))
        move = num*2
        istart += move  
    allsm = Mesh()
    allsm.make_mesh(allv[1:],allf[1:])
    
    #print("---Unroll %.2g seconds ---" % (time.time() - start_time))
    return allsm

def unroll_1_strip(V,coo,anchor,efair,ee,itera,is_midaxis,w_straight):
    """forked from meshFunction.py / unrollGeodesicStrip
        strip is constructed by two array of points [Varr1; Varr2]
        meshV: [col_i; col_i+1]
        coo: plane x=0, y=0, z=0, (coo=0,1,2)
        
        v4  --- v3
         |       |
         |       |
        v1 ---- v2
    """
    # import time
    # start_time = time.time()
    
    sm, v,f = opt_unfold_quad(V,coo,anchor,efair,ee,itera,is_midaxis,w_straight)
    
    #print("---Unroll %.2g seconds ---" % (time.time() - start_time))
    
    return sm,v,f

def distortion_all_strips(mesh3d,mesh2d,edge=False,length=False,angle=False,area=False,dig_ctrl=False):
    v3d, v2d = mesh3d.vertices, mesh2d.vertices,
    flist = np.array(mesh3d.faces_list())
    v1,v2,v3,v4 = v3d[flist[:,0]],v3d[flist[:,1]],v3d[flist[:,2]],v3d[flist[:,3]]
    w1,w2,w3,w4 = v2d[flist[:,0]],v2d[flist[:,1]],v2d[flist[:,2]],v2d[flist[:,3]]
    el = mesh3d.mean_edge_length()    
    d11 = np.linalg.norm(v3-v1,axis=1)
    d12 = np.linalg.norm(v4-v2,axis=1)
    d21 = np.linalg.norm(w3-w1,axis=1)
    d22 = np.linalg.norm(w4-w2,axis=1)  
        
    if edge:
        i1,i2 = mesh3d.edge_vertices()
        dv = np.linalg.norm(v3d[i1]-v3d[i2],axis=1)
        dw = np.linalg.norm(v2d[i1]-v2d[i2],axis=1)
        err = (np.abs(dv-dw)) / el
        
    elif length:
        "diagonal length of control face"    
        err = (np.abs(d11-d21)+np.abs(d12-d22)) / el
        print('\n Isometry (length) error = (np.abs(d11-d21)+np.abs(d12-d22)) / el : \n')

    elif angle:
        ag1 = np.einsum('ij,ij->i',v3-v1,v4-v2)
        ag2 = np.einsum('ij,ij->i',w3-w1,w4-w2)
        cos1 = ag1/d11/d12
        cos2 = ag2/d21/d22
        try:
            err = np.abs((np.arccos(cos1)-np.arccos(cos2))*180/np.pi)
        except:
            err = np.zeros(len(cos1))
            for i in range(len(cos1)):
                if cos1[i]>1 or cos1[i]<-1 or cos2[i]>1 or cos2[i]<-1:
                    print(cos1[i],cos2[i])
                    continue
                else:
                    err[i] = np.abs((np.arccos(cos1[i])-np.arccos(cos2[i]))*180/np.pi)
        print('\n Isometry (angle) error=np.abs((np.arccos(cos1)-np.arccos(cos2))*180/np.pi) : \n')
        
    elif area:
        a1 = np.linalg.norm(np.cross(v2-v1,v4-v1),axis=1)
        a2 = np.linalg.norm(np.cross(w2-w1,w4-w1),axis=1)
        err = np.abs(a1-a2) / (el**2) / 2
        print('\n Isometry (area) error : \n')

    elif dig_ctrl:
        "diagonal length of checkerboard quad"
        d11 = np.linalg.norm((v2+v1)/2-(v3+v4)/2,axis=1)
        d12 = np.linalg.norm((v3+v2)/2-(v4+v1)/2,axis=1)
        d21 = np.linalg.norm((w2+w1)/2-(w3+w4)/2,axis=1)      
        d22 = np.linalg.norm((w3+w2)/2-(w4+w1)/2,axis=1)
        err = (np.abs(d11-d21)+np.abs(d12-d22))/2 / el

    print('   max = ','%.3g' % np.max(err),'\n')    
    return err
    
# -------------------------------------------------------------------------
#          Unroll optmization : ARAP - ISOMETRY CONSTRAINTS:
# -------------------------------------------------------------------------
def opt_unfold_quad(V,coo,anchor,efair,ee,itera,is_midaxis,w_straight):
    #start_time = time.time()
    
    num = int(len(V)/2)
    X,var,lh,ll,lr,l13,l24,fixp = initial_unfold(V,coo,anchor,is_midaxis,w_straight)
    n = 0
    K = matrix_fair(num,var,efair)
    I = sparse.eye(var,format='coo')*ee**2

    opt_num, opt = 100, 100
    "note should have opt-check, otherwise only run once for asymtotpicstrips"
    while n < itera and (opt_num>1e-6 or opt>1e-6) and opt_num<1e+6:
    #while n < itera and opt_num>0.00001 and opt_num<1e+6:
        H, r, opt = con_isometry(X,var,num,lh,ll,lr,l13,l24,coo,anchor,fixp,is_midaxis,w_straight)
        X = sparse.linalg.spsolve(H.T*H+K.T*K+I, H.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
        n += 1
        opt_num = np.sum(np.square((H*X)-r))
        #print('Optimize the unfolding mesh:',n, '%.2g' %opt, '%.2g' %opt_num)#'%.2g' %opt_num,
    
    #print(n, '%.2g' %opt_num,'%.2g s' %(time.time() - start_time))
    print(n, '%.2g' %opt_num)
    
    Vl = X[:3*num].reshape(-1,3,order='F')
    Vr = X[3*num:6*num].reshape(-1,3,order='F')
    V = np.vstack((Vl,Vr))
    arr = np.arange(num)
    v1,v2,v3,v4 = arr[:-1],arr[:-1]+num,arr[1:]+num,arr[1:]
    flist = np.c_[v1,v2,v3,v4]
    sm = Mesh()
    sm.make_mesh(V, flist)    
    return sm,V,flist

def initial_unfold(V,coo,anchor,is_midaxis,w_straight):
    """ X = [Vl ;  Vr]
        v4  --- v3
         |       |
         |       |
        v1 ---- v2    
    """
    num = int(len(V)/2)
    vl,vr = V[:num],V[num:]
    v1,v4 = vl[:-1],vl[1:]
    v2,v3 = vr[:-1],vr[1:]
    lh = np.linalg.norm(vl-vr,axis=1) # horizontal
    ll = np.linalg.norm(v1-v4,axis=1) # left
    lr = np.linalg.norm(v2-v3,axis=1) # right
    l13 = np.linalg.norm(v1-v3,axis=1) 
    l24 = np.linalg.norm(v2-v4,axis=1)     
    
    d1,d2 = [0],[0]
    k1,k2 = 0,0
    for i in range(num-1):
        k1 += ll[i]
        k2 += lr[i]
        d1.append(k1)
        d2.append(k2)
    Vl,Vr = np.array(d1), np.array(d2)

    dist_axis = lh[0]
    zero = np.zeros(num)
    mid = (Vl[-1]+Vr[-1])/4
    if is_midaxis:
        if coo==0:
            "put in y-axis: y-z plane"
            fixp = np.array([0,anchor,0])
            origin = np.array([0,anchor-dist_axis/2,-mid])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero,zero+dist_axis,Vr]
        elif coo==1:
            "put in x-axis: x-z plane"
            fixp = np.array([anchor, 0, 0])
            origin = np.array([anchor-dist_axis/2, 0, -mid])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero+dist_axis,zero,Vr]
        elif coo==2:
            "put in x-axis: x-y plane"
            fixp = np.array([anchor, 0, 0])
            origin = np.array([anchor-dist_axis/2, -mid, 0])
            Pl = origin + np.c_[zero,Vl,zero]
            Pr = origin + np.c_[zero+dist_axis,Vr,zero]
    else:    
        if coo==0:
            "put in y-axis: y-z plane"
            origin = np.array([0,anchor,0])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero,zero+dist_axis,Vr]
        elif coo==1:
            "put in x-axis: x-z plane"
            origin = np.array([anchor, 0, 0])
            Pl = origin + np.c_[zero,zero,Vl]
            Pr = origin + np.c_[zero+dist_axis,zero,Vr]
        elif coo==2:
            "put in x-axis: x-y plane"
            origin = np.array([anchor, -mid, 0]) ##NOTE: ORIGIN=[an,0,0] or [an,-mid,0]
            Pl = origin + np.c_[zero,Vl,zero]
            Pr = origin + np.c_[zero+dist_axis,Vr,zero]
        fixp = np.r_[Pl[0],Pr[0]]
        
    X = np.r_[Pl.flatten('F'),Pr.flatten('F')]    
    if w_straight:
        "right-const.coordinate"
        if coo==0:
            lconst,rconst = np.mean(Pl[:,1]),np.mean(Pr[:,1])
        elif coo==1:
            lconst,rconst = np.mean(Pl[:,0]),np.mean(Pr[:,0])
        elif coo==2:
            lconst,rconst = np.mean(Pl[:,0]),np.mean(Pr[:,0])
        X = np.r_[X,lconst,rconst]
    var = len(X)
    return X,var,lh,ll,lr,l13,l24,fixp

def con_isometry(X,var,num,lh,ll,lr,l13,l24,coo,anchor,fixp,is_midaxis,w_straight):
    arr = np.arange(num)
    c_vl = np.r_[arr, num+arr, 2*num+arr]
    c_vr = c_vl + 3*num
    
    c_v1 = np.r_[arr[:-1], num+arr[:-1], 2*num+arr[:-1]]
    c_v4 = np.r_[arr[1:], num+arr[1:], 2*num+arr[1:]]
    c_v2, c_v3 = c_v1 + 3*num, c_v4 + 3*num

    def _con_in_plane():
        "coo=0,1,2 ~ yz, xz, xy -- plane"
        row = np.arange(2*num)
        col = coo*num+np.r_[arr,3*num+arr]
        data = np.ones(2*num)
        H = sparse.coo_matrix((data,(row,col)), shape=(2*num, var))
        r = np.zeros(2*num)
        return H, r  
    
    def _con_edge_length():
        def _edge(c1,c2,l1):
            "(P1-P2)^2 = l1**2"
            num = len(l1)
            arr = np.arange(num)
            col = np.r_[c1,c2]
            row = np.tile(arr,6)
            data = 2*np.r_[X[c1]-X[c2],X[c2]-X[c1]]
            r = np.linalg.norm((X[c1]-X[c2]).reshape(-1,3, order='F'),axis=1)**2+l1**2
            H = sparse.coo_matrix((data,(row,col)), shape=(num, var))
            return H,r   
        H1,r1 = _edge(c_vl,c_vr,lh)
        H2,r2 = _edge(c_v1,c_v4,ll)
        H3,r3 = _edge(c_v2,c_v3,lr)
        H4,r4 = _edge(c_v1,c_v3,l13)
        H5,r5 = _edge(c_v2,c_v4,l24)
        H = sparse.vstack((H1*10,H2*10,H3*10,H4,H5)) ##NOTE: NEED TO SET OF WEIGHTS
        r = np.r_[r1*10,r2*10,r3*10,r4,r5]
        return H ,r
    
    def _con_fix():
        "fix the axis-alignment two points"
        row = np.arange(6)
        col = np.r_[c_vl[np.array([0,num,2*num])], c_vr[np.array([0,num,2*num])]]
        data = np.ones(6)
        r = fixp
        H = sparse.coo_matrix((data,(row,col)), shape=(6, var))
        return H * 10, r * 10
    
    def _con_fix2():
        """fix the midaxis-alignment of central point of each strip
        mid = (vl0+vl1+vr0+vr1)/4;
        p=(vl0+vr0)/2 
            if coo=0,yz-plane,y-axis, p[1] == mid[1]
            if coo=1,xz-plane,x-axis, p[0] == mid[0]
            if coo=2,xy-plane,x-axis, p[0] == mid[0]
        """
        def _fix(c_a,c_b,c_c,c_d):
            "va+vb+vc+vd = 4*fixp"
            row = np.tile(np.arange(3),4)
            col = np.r_[c_a,c_b,c_c,c_d]
            data = np.ones(12)
            r = fixp * 4
            H = sparse.coo_matrix((data,(row,col)), shape=(3, var))
            return H * 10, r * 10
        
        def _glide(c_i,c_j,x):
            "([vl0+vr0])/2=x <==> [vl0+vr0][i] = 2x"
            row = np.array([0,0])
            col = np.array([c_i,c_j])
            data = np.array([1,1])
            r = np.array([2*x])
            H = sparse.coo_matrix((data,(row,col)), shape=(1, var))
            return H,r
        if coo==0:
            i = 1
        elif coo==1 or coo==2:
            i = 0
        arr = np.array([0,num,2*num])
        c_i, c_j, x = c_vl[arr][i], c_vr[arr][i], fixp[i]
        H1,r1 = _glide(c_i,c_j,x)
        c_a = c_vl[np.array([0,num,2*num])]
        c_b = c_vr[np.array([0,num,2*num])]   
        c_c = c_vl[np.array([num-1,2*num-1,3*num-1])]
        c_d = c_vr[np.array([num-1,2*num-1,3*num-1])]
        H2,r2 = _fix(c_a,c_b,c_c,c_d)
        H = sparse.vstack((H1,H2))
        r = np.r_[r1,r2]
        return H,r
    
    def _con_strong_straight():
        "1.vertical-coordiante of left+right is constant"
        "2. global lenght of left/right is kept"
        def _global_edgelength(c_vl,ll,c_const):
            " "
            Ll = np.sum(ll)
            # if is_midaxis:
            #     null =  np.zeros([0])
            #     H = sparse.coo_matrix((null,(null,null)), shape=(0,var))
            #     r = np.array([])
            # else:
            if coo==0:
                "put in y-axis: y-z plane"
                pass
            elif coo==1:
                "put in x-axis: x-z plane"
                pass
            elif coo==2:
                "put in x-axis: x-y plane, [x,y,0]"
                "difference(endcoordiante)=global_length" ###NO USE NOW!
                # row = np.zeros(2)
                # col = np.array([c_vl[2*num],c_vl[num]]) ##y[-1]-y[0]
                # data = np.array([1,-1],dtype=int)
                # Hg = sparse.coo_matrix((data,(row,col)), shape=(1, var))
                # rg = np.array([Ll]) 
                "vertical_coordinate == anchor"
                row = np.tile(arr,2)
                col = np.r_[c_vl[:num],np.ones(num)*c_const] #x
                data = np.r_[np.ones(num),-np.ones(num)]
                rc = np.zeros(num)
                Hc = sparse.coo_matrix((data,(row,col)), shape=(num, var))
                #H = sparse.vstack((Hg,Hc))
                #r = np.r_[rg,rc]
            return Hc,rc
        Hs1,rs1 = _global_edgelength(c_vl,ll,var-2)
        Hs2,rs2 = _global_edgelength(c_vr,lr,var-1)
        H = sparse.vstack((Hs1,Hs2))
        r = np.r_[rs1,rs2]
        return H,r

    H1,r1 = _con_in_plane()
    H2,r2 = _con_edge_length()
    "NOTE:BELOW COMMENT FOR EQUAL STRIPS, NEED TO CHECK FOR OTHERS"
    # if is_midaxis:
    #     H3,r3 = _con_fix2()
    # else:
    #     H3,r3 = _con_fix()
    # H = sparse.vstack((H1,H2,H3))
    # r = np.r_[r1,r2,r3]
    H = sparse.vstack((H1,H2))
    r = np.r_[r1,r2]
    
    if w_straight:
        Hs,rs = _con_strong_straight()
        H = sparse.vstack((H,Hs*w_straight))
        r = np.r_[r,rs*w_straight]
    
    opt = np.sum(np.square((H*X)-r))
    return H,r,opt

def matrix_fair(num,var,efair=0.005):
    "for vl and vr polyline"
    arr = np.arange(num)
    c_vl = np.r_[arr, num+arr, 2*num+arr]
    c_vr = c_vl + 3*num
    #num = int(len(c_vl)/3)
    c_l1 = np.r_[c_vl[:num-2],c_vl[num:2*num-2],c_vl[2*num:3*num-2]]
    c_r1 = np.r_[c_vr[:num-2],c_vr[num:2*num-2],c_vr[2*num:3*num-2]]
    c_l2 = np.r_[c_vl[1:num-1],c_vl[num+1:2*num-1],c_vl[2*num+1:3*num-1]]
    c_r2 = np.r_[c_vr[1:num-1],c_vr[num+1:2*num-1],c_vr[2*num+1:3*num-1]]
    c_l3 = np.r_[c_vl[2:num],c_vl[num+2:2*num],c_vl[2*num+2:3*num]]
    c_r3 = np.r_[c_vr[2:num],c_vr[num+2:2*num],c_vr[2*num+2:3*num]]
    c_1 = np.r_[c_l1,c_r1]
    c_2 = np.r_[c_l2,c_r2]
    c_3 = np.r_[c_l3,c_r3]
    num = len(c_1)
    row = np.tile(np.arange(num),3)
    col = np.r_[c_1,c_2,c_3]
    data = np.r_[np.ones(num),-2*np.ones(num),np.ones(num)]
    K = sparse.coo_matrix((data,(row,col)), shape=(num, var))
    return efair * K



# Below on use, only refer:
# -------------------------------------------------------------------------
#          Unroll functions from GO-project: meshFunctions.py
# -------------------------------------------------------------------------

def unrollGeodesicStrip(V, coo=2, anchor=0, sign=-1):
    """forked from meshFunction.py / 
        strip is constructed by two array of points [Varr1; Varr2]
        meshV: [col_i; col_i+1]
        coo: plane x=0, y=0, z=0, (coo=0,1,2)
        
        v1 --- v22
         |       |
         |       |
        v0 ---- v21
    """    
    num = int(len(V)//2)
    L10 = np.linalg.norm(V[1:num,:]-V[:num-1,:],axis=1) #|v1-v0|
    L121 = np.linalg.norm(V[1:num,:]-V[num:2*num-1,:],axis=1) #|v1-v21|
    L221 = np.linalg.norm(V[num+1:2*num,:]-V[1:num,:],axis=1) #|v22-v1|
    L2221 = np.linalg.norm(V[num+1:2*num,:]-V[num:2*num-1,:],axis=1) #|v22-v21|

    # put w0, w21 in x-axis
    v0, v21 = V[0], V[num]
    l210 = np.linalg.norm(v21-v0)

    if coo==0:# put in y-axis: y-z plane
        w0 = np.array([0,anchor,0])
        w21 = w0 + np.array([0,l210,0])
        a,b = 1,2
    elif coo==1:# put in x-axis: x-z plane
        w0 = np.array([anchor, 0, 0])
        w21 = w0 + np.array([l210,0,0])
        a,b = 0,2
    elif coo==2:# put in x-axis: x-y plane
        w0 = np.array([anchor, 0, 0])
        w21 = w0 + np.array([l210,0,0])
        a,b=0,1
    Vl, Vr = w0, w21
    def stripsV(i, w0, w21):
        "get the whole strip vertices"
        # solve order: w0, w21, w1, w22; ....
        # w1=[x1,y1,0] w22=[x22,y22,0]
        l121 = L121[i-1]
        l10 = L10[i-1]
        l221 = L221[i-1]
        l2221 = L2221[i-1]
        # kxl, kxr, ky are very important to impact result
        kxl, kxr = w0[a], w21[a]
        kyl = w0[b] + l10 # may be not plus
        kyr = w21[b] + l2221
        def vv1(x): # x are the variables of the equation
            return np.array([(x[0]-w21[a])**2+(x[1]-w21[b])**2-l121**2,
                             (x[0]-w0[a])**2+(x[1]-w0[b])**2-l10**2])
        w1_solve = fsolve(vv1,[kxl,kyl])
        w1 = np.insert(w1_solve,coo,0)

        def vv22(x):
            return np.array([(x[0]-w21[a])**2+(x[1]-w21[b])**2-l2221**2,
                             (x[0]-w1[a])**2+(x[1]-w1[b])**2-l221**2])
        w22_solve = fsolve(vv22,[kxr,kyr])
        w22 = np.insert(w22_solve,coo,0)
        return w1, w22

    for i in np.arange(num-1)+1: #[1,2,..20]
        w0, w21 = stripsV(i,w0,w21)
        Vl = np.vstack((Vl, w0))
        Vr = np.vstack((Vr, w21))

    vlist = sign * np.vstack((Vl, Vr))
    arr = np.arange(num)
    v1,v2,v3,v4 = arr[:-1],arr[:-1]+num,arr[1:]+num,arr[1:]
    flist = np.c_[v1,v2,v3,v4]
    sm = Mesh()
    sm.make_mesh(vlist, flist)
    return sm,vlist,flist


def unrollStripMesh(self, col_index, coo):
    "for a strip"
    "meshV: [col_i; col_i+1]"
    sMeshFL = self.faceStripReindex().tolist()
    vSR = self.vertexStripIndex(col_index)
    V = self.vertices[vSR]
    sMeshVL = self.unrollGeodesicStrip(V, coo).tolist()
    unrollMesh = Mesh()
    unrollMesh.make_mesh(sMeshVL, sMeshFL)
    return unrollMesh

def unfoldAllGeodesicStrips(self, anchor=0, step=1, coo=2):
    "above unroll a strip, this unrolls all strips"
    col_indices = np.arange(self.colnum+1)
    vertexArray = np.array([0,0,0],dtype=int)
    faceArray=np.array([0,0,0,0],dtype=int)
    fIndex = self.faceStripReindex()
    for i in col_indices:#[0,1,2,...19]
        vSR = self.vertexStripIndex(i)
        V = self.vertices[vSR]
        vlist = self.unrollGeodesicStrip(V, coo, anchor)# len(vIndex)=42*20,rownum=colnum=20

        v0, v21 = V[0], V[len(V)//2]
        l210 = np.linalg.norm(v21-v0)
        anchor += step * l210
        vertexArray = np.vstack((vertexArray, vlist))
        faceArray = np.vstack((faceArray, fIndex+i*(1+np.max(fIndex))))

    sVlist = vertexArray[1:,:].tolist()
    sFlist = faceArray[1:,:].tolist()
    unrollMesh = Mesh()
    unrollMesh.make_mesh(sVlist, sFlist)
    return unrollMesh

def unfoldWholeMeshAIAP(self,Vv,eFair,ee,coo):
    "use file from unfoldingMesh"
    Y = self.opt_unfoldMesh2Plane(Vv,eFair,ee,coo)
    vertices = (Y[:3*self.V].reshape(-1,3)).tolist()
    faces = self.faces_list()
    wmesh = Mesh()
    wmesh.make_mesh(vertices,faces)
    return wmesh

def unfoldParallelogramMeshAIAP(self,ee):
    "use file unfoldingParallelogram (like unfoldWholeMeshAIAP)"
    V = self.vertices
    vM = self.vertexIndexMatrix()
    overlap=False
    Y = self.opt_unfoldParallelogramMesh2Plane(V,vM,overlap,ee,coo=2)
    self.unfoldY = Y
    vertices = (Y[:3*self.V].reshape(-1,3)).tolist()
    faces = self.faces_list()
    pmesh = Mesh()
    pmesh.make_mesh(vertices,faces)
    return pmesh

def unfoldParallelogram_Mesh(self):
    "use file unfoldingParallelogram (like unfoldWholeMeshAIAP)"
    if self.unfoldY is not None:
        Y = self.unfoldY
        vertices = (Y[3*self.V:].reshape(-1,3)).tolist()
        num = len(vertices)//4
        fM = (np.arange(4*num).reshape(-1,num)).T
        flist = fM.tolist()
        pmesh = Mesh()
        pmesh.make_mesh(vertices,flist)
        return pmesh
    
def unfoldOverlapParallelogramMesh(self,ee):# second optimize\showing
    "project all parallelogram to z=0, minimize the overlap points"
    V = self.vertices
    vM = self.vertexIndexMatrix()
    overlap = True
    Y = self.opt_unfoldParallelogramMesh2Plane(V,vM,overlap,ee,coo=2)
    n = 3*self.V
    P1 = Y[n:n+3*self.fnum].reshape(-1,3)
    P2 = Y[3*self.fnum+n:n+6*self.fnum].reshape(-1,3)
    P3 = Y[6*self.fnum+n:n+9*self.fnum].reshape(-1,3)
    P4 = Y[9*self.fnum+n:n+12*self.fnum].reshape(-1,3)
    return P1,P2,P3,P4        
        