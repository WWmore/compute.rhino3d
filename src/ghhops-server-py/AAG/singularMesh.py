# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:46:47 2021

@author: WANGH0M
"""

import numpy as np
from curvature import frenet_frame
from bezierC2Continuity import BezierSpline
from smooth import fair_vertices_or_vectors

def quadmesh_with_1singularity(mesh,diagply=True,diagquad=False):
    "gridshell_sig.obj"
    _,_,lj = mesh.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
    v6 = np.where(lj>4)[0] # should be 6/8
    iply = get_all_polylines_from_a_vertex(mesh,v6) # should be list of (6,arri)

    plylist = []
    vl = vr = np.array([],dtype=int)
    vlft=vctr=vrgh = np.array([],dtype=int)
    "get the index of diagply emanating from the only 1singularity"
    if diagply: 
        for i in range(len(iply)):
            "for each L-shape boundary of several patches"
            vvd = get_a_boundary_L_strip(mesh,iply[i%lj[v6][0]],iply[(i+1)%lj[v6][0]])
            ##print(vvd[:,0],v6,np.where(vvd[:,0]==v6)[0])
            ###print(i%lj[v6][0],(i+1)%lj[v6][0],iply[i][:2])
            for ivvd in vvd:
                iv = get_diagonal_polyline_from_2points(mesh,ivvd)
                if len(iv)!=0:
                   plylist.append(iv) 
                   if len(iv)==2:
                       vl = np.r_[vl,iv[0]]
                       vr = np.r_[vr,iv[1]]
                   else:
                      vl = np.r_[vl,iv[:-1]]
                      vr = np.r_[vr,iv[1:]]
                      vlft = np.r_[vlft,iv[:-2]]
                      vctr = np.r_[vctr,iv[1:-1]]
                      vrgh = np.r_[vrgh,iv[2:]]    
        return plylist, [vl,vr], [vlft, vctr, vrgh]
    
    if diagquad: 
        v1=v2=v3=v4 = np.array([],dtype=int)
        for i in range(len(iply)):
            "for each L-shape boundary of several patches"
            vvd = get_a_boundary_L_strip(mesh,iply[i%lj[v6][0]],iply[(i+1)%lj[v6][0]])
            for ivvd in vvd:
                i1,i2,i3,i4 = get_diagonal_quadface_from_2points(mesh,ivvd)
                if len(i1)!=0:
                   v1 = np.r_[v1,i1]
                   v2 = np.r_[v2,i2]
                   v3 = np.r_[v3,i3]
                   v4 = np.r_[v4,i4]
        return v1,v2,v3,v4


def quadmesh_2familypolyline_with_1singularity(mesh,is_one_or_another,except_updown=False):
    "quadmesh_with_1singularity divides 2family_polylines_of_ctrlmesh"
    _,_,lj = mesh.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
    v6 = np.where(lj>4)[0] # should be 6/8
    iply = get_all_polylines_from_a_vertex(mesh,v6) # should be list of (6,arri)
    
    vLb,num_row,num_col = get_full_L_strip_boundary_and_patchmatrix(mesh,
                                                                  iply[1%lj[v6][0]],
                                                                  iply[(1+1)%lj[v6][0]],
                                                                  v6,True)
    M01=M02 = np.zeros((1,2*num_col-1),dtype=int)
    for i in range(len(iply)):
        if i%2==1:
            Mi = get_full_L_strip_boundary_and_patchmatrix(mesh,
                                                        iply[i%lj[v6][0]],
                                                        iply[(i+1)%lj[v6][0]],
                                                        v6)
            Mj = get_full_L_strip_boundary_and_patchmatrix(mesh,
                                                        iply[(i+1)%lj[v6][0]],
                                                        iply[(i+2)%lj[v6][0]],
                                                        v6)
            M1 = np.hstack((Mi[::-1,::-1],(Mj.T)[:,::-1][:,1:]))
            if except_updown:
                "except up+down row, include left+right column"
                M01 = np.vstack((M01,M1[1:-1,:]))
            else:
                M01 = np.vstack((M01,M1))
        else:
            Mi = get_full_L_strip_boundary_and_patchmatrix(mesh,
                                                        iply[i%lj[v6][0]],
                                                        iply[(i+1)%lj[v6][0]],
                                                        v6)
            Mj = get_full_L_strip_boundary_and_patchmatrix(mesh,
                                                        iply[(i+1)%lj[v6][0]],
                                                        iply[(i+2)%lj[v6][0]],
                                                        v6)
            M2 = np.hstack((Mi[::-1,::-1],(Mj.T)[:,::-1][:,1:]))
            if except_updown:
                "except up+down row, include left+right column"
                M02 = np.vstack((M02,M2[1:-1,:]))
            else:
                M02 = np.vstack((M02,M2))
            
    M01,M02 = M01[1:,:], M02[1:,:]
    # M = M01 if is_one_or_another else M02
    # vl,vr = M[:,:-1].flatten(),M[:,1:].flatten()
    # return vl, vr
    return M01 if is_one_or_another else M02

def get_allpolylines(mesh,diagnet,is_one_or_another):
    if diagnet:
        "singular: only one kind of diagonal polys, emerating from singularV"
        iall,_,_ = quadmesh_with_1singularity(mesh)
    else:
        "only control-net (used for AAG-net)"
        iall = quadmesh_2familypolyline_with_1singularity(mesh,is_one_or_another)
    return iall 

def get_polyline_from_an_edge(mesh,e):
    "from an edge of polyline, orienting to boundary and stop at boundary"
    H = mesh.halfedges
    # = mesh.boundary_curves(corner_split=False)[0]
    iv = [H[e,0]]
    while H[H[H[e,2],4],1] != -1:
        e = H[H[H[e,2],4],2]
        iv.append(H[e,0])
    iv.append(H[H[e,4],0])
    return iv

def get_all_polylines_from_a_vertex(mesh,iv):
    H = mesh.halfedges
    ei = np.where(H[:,0]==iv)[0]
    if True:
        "MUST: order the iply in counter-clock continuous way:"
        ej = ei[0]
        alle = []
        for i in range(len(ei)):
            alle.append(H[H[ej,4],2])
            ej = H[H[ej,4],2]
    else:
        alle = [ei[0]]
        ej = ei[0]
        ei = ei[1:]
        while len(ei)!=0:
            if H[H[ej,4],2] in ei:
                ej = H[H[ej,4],2]
                alle.append(ej)
                ei = np.delete(ei,np.where(ei==ej)[0])

    iply = []
    for e in alle:
        iply.append(get_polyline_from_an_edge(mesh,e))    
    return iply

def get_singulary_polylines(mesh):
    "gridshell_sig.obj,return 6 polylines starting from singularV6"
    _,_,lj = mesh.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
    iv = np.where(lj>4)[0] # should be 6
    plylist = get_all_polylines_from_a_vertex(mesh,iv)
    #print(plylist)
    """ in counter-clock direction around the singulary: 
    [[660, 626, 634, 647, 652, 606, 624, 628, 641, 645, 658], 
     [660, 605, 633, 653, 659, 635, 613, 636, 640, 627, 657], 
     [660, 600, 609, 616, 621, 629, 637, 643, 655, 601, 610], 
     [660, 604, 631, 650, 646, 602, 614, 611, 618, 649, 642], 
     [660, 625, 632, 639, 651, 620, 607, 648, 608, 615, 654], 
     [660, 623, 619, 612, 603, 656, 644, 638, 630, 622, 617]]
    """
    return plylist
    
def get_a_boundary_L_strip(mesh,ib1,ib2):
    """
    ----------
    mesh : input mesh
    ib1 : list or array
        1 list of polyline from vi
    ib2 : TYPE
        next 1 list of polyline from vi
    Returns
    -------
    2-arry:
        L-shape boundary and its diagonal vertices
    """
    H = mesh.halfedges
    i10 = np.where(H[:,0]==ib1[0])[0]
    i11 = np.where(H[H[:,4],0]==ib1[1])[0]
    ie1 = np.intersect1d(i10,i11)[0]
    i20 = np.where(H[:,0]==ib2[0])[0]
    i21 = np.where(H[H[:,4],0]==ib2[1])[0]
    ie2 = np.intersect1d(i20,i21)[0]
    if H[H[ie1,4],1] != H[ie2,1]:
        ib1,ib2 = ib2,ib1
    ib = np.r_[ib2[::-1][1:], ib1[1:-1]]
    idiag = []
    for j in range(len(ib2[::-1])-1):
        i1 = np.where(H[:,0]==ib2[::-1][j])[0]
        i2 = np.where(H[H[:,4],0]==ib2[::-1][j+1])[0]
        ie = np.intersect1d(i1,i2)[0]
        idiag.append(H[H[H[H[ie,4],2],2],0])
    for j in range(len(ib1[1:])-1):
        i1 = np.where(H[:,0]==ib1[1:][j])[0]
        i2 = np.where(H[H[:,4],0]==ib1[1:][j+1])[0]
        ie = np.intersect1d(i1,i2)[0]
        idiag.append(H[H[H[ie,4],3],0])
    return np.c_[ib, idiag]

def get_full_L_strip_boundary_and_patchmatrix(mesh,ib1,ib2,vcorner,full_Lboundary=False):
    H = mesh.halfedges
    v_vd = get_a_boundary_L_strip(mesh,ib1,ib2) ##v_vd=np.c_[v,vd]

    def get_end_vertex(v,vd):
        es1 = np.where(H[:,0]==v)[0]
        es2 = np.where(H[:,0]==vd)[0]
        f1 = H[es1,1]
        f2 = H[es2,1]
        f = np.intersect1d(f1,f2)
        il = es1[np.where(f1==f)[0]][0] # opposite edge1
        if H[H[H[H[il,2],2],4],1]==-1:
            vend = H[H[il,3],0]
        elif H[H[H[il,2],4],1]==-1:
            vend = H[H[il,2],0]
        return vend
    vstart = get_end_vertex(v_vd[0,0],v_vd[0,1])
    vend = get_end_vertex(v_vd[-1,0],v_vd[-1,1])
    
    v_Lboundary = np.r_[vstart,v_vd[:,0],vend]
    num_row = int(np.where(v_Lboundary==vcorner)[0][0])+1 ##ex:len(Lb)=10,where(vc)=5
    num_col = len(v_Lboundary)-num_row+1 ##then, (num_row,num_col)=(6,5)
    
    if full_Lboundary:
        return v_Lboundary,num_row,num_col

    M = np.zeros((num_row,num_col),dtype=int)
    M[:,0],M[-1,1:] = v_Lboundary[:num_row],v_Lboundary[num_row:]
    
    ik = 0
    for ivvd in v_vd:
        iv = get_diagonal_polyline_from_2points(mesh,ivvd)
        if ik<num_row-1:
            M[np.arange(ik+2)[::-1],np.arange(len(iv))] = iv
        else:
            row = np.arange(num_row-len(iv),num_row)[::-1]
            col = np.arange(num_col-len(iv),num_col)
            M[row,col]= iv
        ik += 1
    return M


def get_diagonal_polyline_from_2points(mesh,vi):
    "refer quadrings.py same fun., but different computing way"
    H = mesh.halfedges
    es1 = np.where(H[:,0]==vi[0])[0]
    es2 = np.where(H[:,0]==vi[1])[0]
    f1 = H[es1,1]
    f2 = H[es2,1]
    f = np.intersect1d(f1,f2)
    il = es1[np.where(f1==f)[0]][0] # opposite edge1
    ir = es2[np.where(f2==f)[0]][0] # opposite edge2
    
    vrr = [H[il,0],H[ir,0]]
    while H[H[H[ir,3],4],1] !=-1 and H[H[ir,4],1]!=-1:
        ir = H[H[H[H[ir,4],2],4],3]
        vrr.append(H[ir,0])
    return np.array(vrr,dtype=int)

def get_diagonal_quadface_from_2points(mesh,vi):
    """ similar to above: get_diagonal_polyline_from_2points
    from given 2diagonal vertices [iv1,iv3], get the whole diagonal quads
    [v1,v2,v3,v4]
    """
    H = mesh.halfedges
    es1 = np.where(H[:,0]==vi[0])[0]
    es2 = np.where(H[:,0]==vi[1])[0]
    f1 = H[es1,1]
    f2 = H[es2,1]
    f = np.intersect1d(f1,f2)
    il = es1[np.where(f1==f)[0]][0] # opposite edge1
    ir = es2[np.where(f2==f)[0]][0] # opposite edge2
    
    v1,v2,v3,v4 = [H[il,0]],[H[H[ir,4],0]],[H[ir,0]],[H[H[il,4],0]]

    while H[H[H[ir,3],4],1] !=-1 and H[H[ir,4],1]!=-1:
        ir = H[H[H[H[ir,4],2],4],3]
        il = H[H[ir,2],2]
        v1.append(H[il,0])
        v2.append(H[H[ir,4],0])
        v3.append(H[ir,0])
        v4.append(H[H[il,4],0])

    return np.array(v1,dtype=int),np.array(v2,dtype=int),\
        np.array(v3,dtype=int),np.array(v4,dtype=int)  


def get_even_segment_from_allpolylines(mesh,diagnet,is_one_or_another,except_updown):
    even_list = []
    if diagnet:
        "singular: only one kind of diagonal polys, emerating from singularV"
        iall,_,_ = quadmesh_with_1singularity(mesh)
        for iv in iall:
            # "start with L-shape boundary whose turning==V6, endwithout 1bdry"
            # if len(iv[:-1])>2 and len(iv[:-1])%2==1:
            #     even_list.append(iv[:-1])
            if len(iv)>5 and len(iv)%2==1:
                even_list.append(iv)
    else:
        "only control-net (used for AAG-net), withour 2bdries"
        iall = quadmesh_2familypolyline_with_1singularity(mesh,is_one_or_another,except_updown)
        for iv in iall:
            # if len(iv[1:-1])>2 and len(iv[1:-1])%2==1:
            #     even_list.append(iv[1:-1])
            if len(iv)>5 and len(iv)%2==1:
                even_list.append(iv)
    return even_list

def get_singular_quintic_Bezier_spline_crvs_checker(mesh,normal,
                                                efair=0.01,
                                                is_asym_or_geo=True,
                                                diagpoly=False,
                                                is_one_or_another=False,
                                                is_checker=1,
                                                is_dense=False,num_divide=5,
                                                is_smooth=0.0):
    "For each polyline, 1.[Pi,Ti,Ni,ki] 2.opt to get ctrl-p,polygon,crv"

    V = mesh.vertices
    N = normal ### surf_normal n
    ievenlist = get_even_segment_from_allpolylines(mesh,
                                                   diagpoly,
                                                   is_one_or_another,
                                                   except_updown=True)
    an = np.array([0,0,0])
    ruling = np.array([0,0,0])
    all_kg=all_kn=all_k=all_tau=np.array([])
    arr = np.array([],dtype=int)
    varr = np.array([],dtype=int)
    num = 0
    #if is_Frame:
    P=Pl=Pr = np.array([0,0,0])
    crvPl=crvPr = np.array([0,0,0])
    frm1=frm2=frm3 = np.array([0,0,0])
 
    #seg_q1234,seg_vl, seg_vr = np.array([0,0,0]),[],[]
    ctrlP = []
    kck = is_checker
    
    row_list = []
    dense_row_list = []
    for iv in ievenlist:
        "Except two endpoints on the boudnary"
        
        if diagpoly:
            "only singular-diagonal case"
            iv_sub,iv_sub1 = iv[:-kck],iv[1:-kck+1]
            if kck==1:
                iv_sub,iv_sub1 = iv[:-2],iv[1:-1]
            Pi = V[iv_sub]
            Ti = (V[iv_sub1]-Pi) / np.linalg.norm(V[iv_sub1]-Pi,axis=1)[:,None] 
        else:
            "control"
            #iv_sub,iv_sub1,iv_sub3 = iv[kck:-kck],iv[kck-1:-kck-1],iv[kck+1:-kck+1]
            #if kck==1:
            iv_sub,iv_sub1,iv_sub3 = iv[1:-1],iv[:-2],iv[2:]
            Pi = V[iv_sub]
            frame = frenet_frame(Pi,V[iv_sub1],V[iv_sub3])
            "using surface normal computed by A-net or G-net"
            Ti,Ef2,Ef3 = frame ### Frenet frame (E1,E2,E3)
            
        Ni = N[iv_sub] ### SURF-NORMAL
        "if asym: binormal==Ni; elif geo: binormal == t x Ni"
        #E3i = Ni if is_asym_or_geo else np.cross(Ti,Ni)
        if is_asym_or_geo:
            "asymptotic; orient binormal with surf-normal changed at inflections"
            E3i = Ni
            # i = np.where(np.einsum('ij,ij->i',Ef3,E3i)<0)[0]
            # E3i[i] = -E3i[i]
        else:
            "geodesic"
            E3i = np.cross(Ti,Ni)

        "checker_vertex_partial_of_submesh case"
        Pi = Pi[::kck]
        Ti = Ti[::kck]
        E3i = E3i[::kck]
        iv_ck = iv_sub[::kck]
        #seg_vl.extend(iv_ck[:-1])
        #seg_vr.extend(iv_ck[1:])

        bs = BezierSpline(degree=len(Pi)-1,continuity=3,
                          efair=efair,itera=200,
                          endpoints=Pi,tangents=Ti,normals=E3i)

        p, pl, pr = bs.control_points(is_points=True,is_seg=True)
        P,Pl,Pr = np.vstack((P,p)), np.vstack((Pl,pl)), np.vstack((Pr,pr))
        ctrlP.append(p)
        crvp,crvpl,crvpr = bs.control_points(is_curve=True,is_seg=True)
        crvPl, crvPr= np.vstack((crvPl,crvpl)), np.vstack((crvPr,crvpr))

        row_list.append(len(Pi))
        if not is_dense:
            kg,kn,k,tau,d = bs.get_curvature(is_asym_or_geo)
            an = np.vstack((an,Pi))
            oNi = np.cross(E3i,Ti)
            frm1 = np.vstack((frm1,Ti))  ## Frenet-E1
            frm2 = np.vstack((frm2,oNi)) ## Frenet-E2
            frm3 = np.vstack((frm3,E3i)) ## Frenet-E3

            if False:
                i = np.where(np.einsum('ij,ij->i',E3i,d)<0)[0]
                d[i] = -d[i]

            arr = np.r_[arr, np.arange(len(iv_ck)-1) + num]
            num += len(iv_ck)
            varr = np.r_[varr,iv_ck]

        else:
            kg,kn,k,tau,pts,d,frmi = bs.get_curvature(is_asym_or_geo,
                                                    True,num_divide,
                                                    False,
                                                    False,
                                                    is_smooth)
            dense_row_list.append(len(d))
            an = np.vstack((an,pts))
            frm1 = np.vstack((frm1,frmi[0])) ## Frenet-E1
            frm2 = np.vstack((frm2,frmi[1])) ## Frenet-E2
            frm3 = np.vstack((frm3,frmi[2])) ## Frenet-E3

            if is_smooth:
                Pup = fair_vertices_or_vectors(pts+d,itera=10,efair=is_smooth)
                d = Pup-pts
                d = d / np.linalg.norm(d,axis=1)[:,None]
                
            arr = np.r_[arr, np.arange(len(kg)-1) + num]
            num += len(kg)
            
        ruling = np.vstack((ruling,d))
        all_kg,all_kn = np.r_[all_kg,kg],np.r_[all_kn,kn]
        all_k,all_tau = np.r_[all_k,k],np.r_[all_tau,tau]
        #seg_q1234 = np.vstack((seg_q1234,bs.control_points(is_Q1234=True)))

    P, Pl, Pr, crvPl, crvPr = P[1:],Pl[1:],Pr[1:],crvPl[1:],crvPr[1:]   
    polygon = None #make_polyline_from_endpoints(Pl, Pr)
    crv = None ##make_polyline_from_endpoints(crvPl, crvPr)
    return P,polygon,crv,np.array(ctrlP,dtype=object),\
           [an[1:],frm1[1:],frm2[1:],frm3[1:]],\
           [varr,an[1:],ruling[1:],arr,row_list,dense_row_list],\
           [all_kg,all_kn,all_k,all_tau]
    #return seg_q1234[1:],[seg_vl,seg_vr]