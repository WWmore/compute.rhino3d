#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from __future__ import absolute_import

# from __future__ import print_function

# from __future__ import division

import numpy as np
from scipy import sparse
#------------------------------------------------------------------------------

from meshpy import Mesh

from orient import  orient_rings

from conicSection import sphere_equation

#------------------------------------------------------------------------------

"""
Created on Thu Mar  5 14:30:58 2020

@author: wanghui
"""
#------------------------------------------------------------------------------

"""
Build on geometry.meshpy
Supplementary some mesh functions
meshpy.py --> quadrings.py --> gridshell.py --> gui_basic.py + geolabgui.py
(GeolabGUI)--> opt_auxetic.py
"""

#------------------------------------------------------------------------------


class MMesh(Mesh):

    def __init__(self):
        Mesh.__init__(self)

        self._ringlist = [] # all vertices' ringlist

        self.angle = 90

        self._ver_regular = None # array for vertex of valence 4

        self._num_regular = 0

        self._ver_regular_star = None # _star[:,0] == ver_regular
        self._ver_star_matrix = None # for oriented ver_regular_star
        self._red_regular_vertex = None
        self._ind_ck_regular_vertex = None
        self._ind_ck_tian_regular_vertex = None
        
        self._patch_matrix = None
        self._rot_patch_matrix = None
        #self._num_patch_row, self._num_patch_col = 0, 0
        
        self._orientrn = None # orient regular normal
        
        self._ver_bod_valence3_neib = None
        self._ver_inn_valence3_neib = None
        self._ver_inn_valence5_neib = None
        self._ver_inn_valence6_neib = None
        self._ver_corner_valence4_neib = None
        
        self._vi, self._vj = None, None

        self._inner = None
        self._corner = None
        self._join_vertex = None

        self._num_quadface,self._quadface,self._quadface_order = 0, None, None
        self._num_rrv,self._num_rrf,self._rr_star,self._rr_quadface=0,0,None,None
        self._rr_quadface_order,self._rr_4quad_vers = None,None
        self._rr_star_corner = None
        self._rr_quadface_tristar = None
        self._rr_quadface_4neib = None
        self._ind_rr_star_v4f4,self._ind_rr_quadface_order = None,None
        self._ind_multi_quad_face,self._ind_multi_rr_face = None,None
        self._checker_face,self._checker_vertex = None,None # all
        self._checker_vertex_tian = None # sub-group
        self._checker_vertex_tian_corner = None
        self._quad_check,self._vertex_check,self._vertex_check_star = None,None,None # regular face/vertex
        self._vertex_check_ind,self._vertex_check_corner_ind = None,None
        self._vertex_rr_check_ind,self._vertex_rr_check = None,None
        self._vertex_check_near_boundary_ind = None

        self._ind_multi_close_boundary = None

        self._quad_common_edge = None
        self._matrixI,self._matrixK,self._var = None, None, None
        self.__No,self.__Ns = 0,0
        
        
    # -------------------------------------------------------------------------
    #                        Properties
    # -------------------------------------------------------------------------
    @property
    def ringlist(self):
        if self._ringlist == []:
            self._ringlist = self.vertex_ring_vertices_list()
        return self._ringlist
    @property
    def inner(self):
        if self._inner == []:
            self.nonsingular()
        return self._inner
    @property
    def corner(self):
        if self._corner == [] or self._corner is None:
            self.nonsingular()
        return self._corner
    
    @property
    def join_vertex(self):
        if self._join_vertex is None:
            self.get_close_two_boundary_points_index()
        return self._join_vertex
    @join_vertex.setter
    def join_vertex(self,join_vertex):
        self._join_vertex = join_vertex
    
    @property
    def ver_regular(self):
        if self._ver_regular is None:
            self.nonsingular()
        return self._ver_regular

    @property
    def num_regular(self):
        if self._num_regular==0:
            self.nonsingular()
        return self._num_regular

    @property
    def patch_matrix(self):
        if self._patch_matrix is None:
            self.regular_rectangle_patch()
        return self._patch_matrix
    #when it's patch, self.num_regular=len(self.ind_rr_star_v4f4)=len(self.rr_star)=inner(patch_matrix)
    
    @property
    def rot_patch_matrix(self):
        if self._rot_patch_matrix is None:
            self.regular_rotational_patch()
        return self._rot_patch_matrix

        
    @property
    def orientrn(self):
        self._orientrn =self.new_vertex_normals()[self.rr_star[:,0]]
        return self._orientrn
    
    @property
    def vi(self):
        return self._vi

    @property
    def vj(self):
        return self._vj

    @property
    def ver_regular_star(self):
        if self.angle == 90:
            self._ver_regular_star = self.nonsingular_star_matrix()
        else:
            self._ver_regular_star = np.array(orient_rings(self))
        return self._ver_regular_star
    
    @property
    def ver_star_matrix(self):
        self._ver_star_matrix = np.array(orient_rings(self))
        return self._ver_star_matrix
    
    @property
    def ver_bod_valence3_neib(self):
        self.vertex_valence3_neib()
        return self._ver_bod_valence3_neib  
    @property
    def ver_inn_valence3_neib(self):
        self.vertex_valence3_neib(boundary_vertex=False)
        return self._ver_inn_valence3_neib 
    
    @property
    def ver_inn_valence5_neib(self):
        self.vertex_valence5_neib()        
        return self._ver_inn_valence5_neib 
    
    @property
    def ver_inn_valence6_neib(self):
        self.vertex_valence6_neib()
        return self._ver_inn_valence6_neib
    
    @property
    def ver_corner_valence4_neib(self):
        self.vertex_valence4_neib(corner=True)
        return self._ver_corner_valence4_neib

    @property
    def ver_corner_neib(self):
        self.vertex_corner_neib()
        return self._ver_corner_neib
    
    @property
    def num_quadface(self):
        if self._num_quadface==0:
            self.quadfaces()
        return self._num_quadface
    @property
    def quadface(self):
        if self._quadface is None:
            self.quadfaces()
        return self._quadface
    @property
    def quadface_order(self):
        if self._quadface_order is None:
            self.quadfaces()
        return self._quadface_order
    
    @property
    def num_rrv(self):
        if self._num_rrv==0:
            self.regular_vertex_regular_quad()
        return self._num_rrv
    @property
    def num_rrf(self):
        if self._num_rrf==0:
            self.regular_vertex_regular_quad()
        return self._num_rrf
    @property
    def rr_star(self):
        if self._rr_star is None:
            self.regular_vertex_regular_quad()
        return self._rr_star
    @property
    def rr_quadface(self):
        if self._rr_quadface is None:
            self.regular_vertex_regular_quad()
        return self._rr_quadface
    @property
    def rr_quadface_order(self):
        if self._rr_quadface_order is None:
            self.regular_vertex_regular_quad()
        return self._rr_quadface_order
    @property
    def rr_4quad_vers(self): #[vj1,vj2,vj3,vj4],j=1,2,3,4, at rr_star
        if self._rr_4quad_vers is None:
            self.regular_vertex_regular_quad()
        return self._rr_4quad_vers
    @property
    def ind_rr_quadface_order(self):
        if self._ind_rr_quadface_order is None:
            self.index_of_4quad_face_order_at_regular_vs()
        return self._ind_rr_quadface_order
    @property
    def ind_rr_star_v4f4(self):
        if self._ind_rr_star_v4f4 is None:
            self.index_of_4quad_face_order_at_regular_vs()
        return self._ind_rr_star_v4f4
    @property
    def rr_star_corner(self):
        if self._rr_star_corner is None:
            self.get_rr_vs_4face_corner()
        return self._rr_star_corner    
    @property
    def rr_quadface_tristar(self):
        if self._rr_quadface_tristar is None:
            self.quadface_neighbour_star()
        return self._rr_quadface_tristar
    @property
    def rr_quadface_4neib(self):
        if self._rr_quadface_4neib is None:
            "index in self.rr_quadface_order"
            self.quadface_neighbour_star()
        return self._rr_quadface_4neib   
    @property
    def ind_multi_rr(self):
        if self._ind_multi_quad_face is None:
            self.index_of_multi_quadface_at_regular_vs()
        return self._ind_multi_quad_face
    @property
    def ind_multi_rr_face(self):
        if self._ind_multi_rr_face is None:
            self.index_of_multi_quadface_at_regular_vs(faceindex=True)
        return self._ind_multi_rr_face
    @property
    def quad_check(self):
        if self._quad_check is None:
            self.quadface_checkerboard_order()
        return self._quad_check
    @property
    def vertex_check(self):
        if self._vertex_check is None:
            self.regular_vertex_checkerboard_order()
        return self._vertex_check
    @property
    def vertex_check_star(self):
        if self._vertex_check_star is None:
            self.regular_vertex_checkerboard_order()
        return self._vertex_check_star
    @property
    def vertex_check_ind(self):
        if self._vertex_check_ind is None:
            self.regular_vertex_checkerboard_order()
        return self._vertex_check_ind
    @property
    def vertex_rr_check_ind(self):
        if self._vertex_rr_check_ind is None:
            self.index_of_multi_quadface_at_regular_vs(faceindex=True)
        return self._vertex_rr_check_ind
    @property
    def vertex_rr_check(self):
        if self._vertex_rr_check is None:
            self.index_of_multi_quadface_at_regular_vs(faceindex=True)
        return self._vertex_rr_check
    @property
    def vertex_check_corner_ind(self):
        if self._vertex_check_corner_ind is None:
            self.index_of_checker_vertex_pi_index_at_corner()
        return self._vertex_check_corner_ind
    @property
    def vertex_check_near_boundary_ind(self):
        if self._vertex_check_near_boundary_ind is None:
            self.index_of_checker_vertex_near_boundary()
        return self._vertex_check_near_boundary_ind
    @property
    def checker_face(self):
        if self._checker_face is None:
            self.get_checker_select_faces()
        return self._checker_face
    @property
    def checker_vertex(self):
        if self._checker_vertex is None:
            self.get_checker_select_vertex()
        return self._checker_vertex
    @property
    def ind_red_rr_vertex(self): # no use now. replaced by ind_ck_rr_vertex
        if self._red_regular_vertex is None:
            self.red_regular_regular_vertex(blue=False)
        return self._red_regular_vertex
    @property
    def ind_ck_rr_vertex(self):
        if self._ind_ck_regular_vertex is None:
            self.ind_checker_regular_regular_vertex()
        return self._ind_ck_regular_vertex
    
    @property
    def checker_vertex_tian(self):
        if self._checker_vertex_tian is None:
            self.get_checker_subgroup_vertices(subpartial=True,k=4)
        return self._checker_vertex_tian
    @property
    def checker_vertex_tian_corner(self):
        if self._checker_vertex_tian_corner is None:
            self.get_checker_subgroup_vertices(subpartial=True,k=4)
        return self._checker_vertex_tian_corner
    @property
    def ind_ck_tian_rr_vertex(self):
        if self._ind_ck_tian_regular_vertex is None:
            self.ind_checker_regular_regular_vertex(tian_partial=True)
        return self._ind_ck_tian_regular_vertex
    
    @property
    def quad_common_edge(self):
        if self._quad_common_edge is None:
            self.face_neighbor_face_common_edge()
        return self._quad_common_edge 
    
    @property
    def matrixI(self):
        if self._matrixI is None:
            self.matrix_unit(self._var)
        return self._matrixI
    @property
    def matrixK(self):
        # if self._matrixK is None:
        #     self.matrix_fair(self._var)
        return self._matrixK

    @property
    def ind_multi_close_boundary(self):
        if self._ind_multi_close_boundary is None:
            self.index_of_auxetic_cut_closed_qi()
        return self._ind_multi_close_boundary
    
    
    # -------------------------------------------------------------------------
    #                        Supplementary functions
    # -------------------------------------------------------------------------

    def columnnew(self, arr, num1, num2):
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


    def column(self, array):
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

    def face_neighbour_faces(self):
        "each face [0,0,0,0,...] and its neighbour faces[-1,1,4,-1,...]"
        H = self.halfedges
        i  = self.face_ordered_halfedges()
        f = H[i,1]
        fij = H[H[i,4],1]
        return f,fij
    
    def face_neighbor_face_common_edge(self):
        """ face_i and face_j have commone edge eij
        eij=face_i & face_j is unique, not appear eij=face_j & face_i
        used for r-velocity w=cc + c x mi, mi is midpoint of edge eij, 
        which is shared by two checkerboards.
        """
        "get unique e satisfy above, without consider the neib-face valence"
        H = self.halfedges
        e = np.intersect1d(np.where(H[:,1]!=-1)[0], np.where(H[H[:,4],1]!=-1)[0])
        unique = []
        for i in e:
            if H[i,4] not in unique:
                unique.append(i)
        ue = np.array(unique)        
        fl,fr = H[ue,1],H[H[ue,4],1]          
        "furthre select all neib-face of valence 4, e.g. being quad"
        forder = self.rr_quadface_order
        quad_e = []
        if1=if2 = np.array([],dtype=int)
        for i in range(len(ue)):
            if fl[i] in forder and fr[i] in forder:
                quad_e.append(ue[i])
                if1 = np.r_[if1, np.where(forder==fl[i])[0]]
                if2 = np.r_[if2, np.where(forder==fr[i])[0]]
        self._quad_common_edge = [np.array(quad_e), if1,if2]
        ##return np.array(quad_e), if1,if2
        
    def edge_from_connected_vertices(self,iv1,iv2,index=True):
        H = self.halfedges
        v1,v2 = self.edge_vertices()
        ie = []
        for i in range(len(iv1)):
            i1,i2 = iv1[i], iv2[i]
            e = np.intersect1d(np.where(H[:,0]==i1)[0], np.where(H[H[:,4],0]==i2)[0])[0]
            ie.append(H[e,5])
        if index:
            return np.array(ie,dtype=int)

    def regular_rectangle_patch(self):
        "patch_matrix: only for m x n patch; num_row = m; num_col = n; for DGPC"
        H = self.halfedges
        b = np.where(H[:,1]==-1)[0]
        c = np.where(H[H[H[H[b,4],3],4],1]==-1)[0]
        corner = H[H[b[c],2],0]
        e0 = b[c[0]] # choose the first corner-edge
        def row(e): # vertices' index --- up--down
            row = []
            while H[e,0] not in corner:
                row.append(e)
                e = H[e,3]
            row.append(e)
            return row

        def column(e): # vertices' index --- left--right
            col = H[e,0]
            ecol = [e]
            e = H[H[H[e,4],2],2]
            while H[e,4] not in b:
                col = np.r_[col,H[e,0]]
                ecol.append(e)
                e = H[H[H[e,4],2],2]
            col = np.r_[col,H[e,0]]
            ecol.append(e)
            return col,ecol
        
        left = row(e0)
        v1,e1 = column(e0)
        v0 = H[H[e1,4],0]
        for e in left:  
            vi,_ = column(e)
            v0 = np.c_[v0,vi]
            
        v,v1,v2,v3,v4 = self.ver_star_matrix[0,:]
        if np.where(v0==v)[1] != np.where(v0==v1)[1]:
            self._patch_matrix = v0.T
        else:
            "v1,v,v3 from up to down; v2,v,v4 from left to right"
            self._patch_matrix = v0 
            
    def regular_rotational_patch(self):
        "rot_patch_matrix: only for m x n patch after cutting along a boundary edge[0]"
        H = self.halfedges
        eb1 = np.where(H[:,1]==-1)[0][0]
        erow1 = [eb1] # index for edges
        eb = eb1
        while H[H[eb,2],0] != H[eb1,0]:
            eb = H[eb,2]
            erow1.append(eb)
        erow1.append(H[eb,2]) # last should be equal to the first
        vm = np.array([],dtype=int)
        for e in erow1:
            eci = [e] # index for edge
            while H[H[e,4],1]!=-1:
                e = H[H[H[e,4],2],2]
                eci.append(e)
            vm = np.r_[vm, H[eci,0]]
        vM = (vm.reshape(len(erow1),len(eci))).T
        self._rot_patch_matrix = vM 

    def nonsingular(self):
        "nonsingular(=regular) vertices v in increased order"
        self._vi,self._vj,lj = self.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
        order = np.where(lj==4)[0]
        self._ver_regular = order
        self._num_regular = len(order)
        self._corner = np.where(lj==2)[0]
        self._inner = np.setdiff1d(np.arange(self.V),self.boundary_vertices())

    def nonsingular_star_matrix(self):
        order = self.ver_regular
        ring = [[] for i in range(len(order))]
        for i in range(len(order)):
            v = order[i]
            ring[i]= self.ringlist[v]
        star = np.c_[order.T,ring]
        return star

    def quadfaces(self):
        "for quad diagonals"
        "quadface, num_quadface, quadface_order"
        f, v1, v2 = self.face_edge_vertices_iterators(order=True)
        f4,vi = [],[]
        for i in range(self.F):
            ind = np.where(f==i)[0]
            if len(ind)==4:
                f4.extend([i,i,i,i])
                vi.extend(v1[ind])
                #vj.extend(v2[ind])
        self._num_quadface = len(f4) // 4
        #v1,v2,v3,v4 = vi[::4],vi[1::4],vi[2::4],vi[3::4]
        self._quadface = np.array(vi,dtype=int)
        self._quadface_order = np.unique(f4)

    def regular_vertex_regular_quad(self,delete_multi=True):
        """ oriented self.quadfaces()
        self.num_rrv : same with num_regular
        self.rr_star : same with starM
        self.num_rrf : same with num_quadface
        self.rr_quadface : same but different order with quadface
        self.rr_quadface_order: same but different order with quadface_order
        f4: including boundary added face [0,1,-1,2]...
        """
        H = self.halfedges
        ##starM = np.array(orient_rings(self))
        starM = self.ver_star_matrix
        num = len(starM)
        f4 = []
        for i in range(num):
            "multiple oriented quad faces"
            v,v1,v2,v3,v4 = starM[i,:]
            ei = np.where(H[:,0]==v)[0]
            ej = H[H[H[H[ei,2],2],2],2]
            e1 = ei[np.where(H[H[ei,4],0]==v1)[0]]
            e2 = ei[np.where(H[H[ei,4],0]==v2)[0]]
            e3 = ei[np.where(H[H[ei,4],0]==v3)[0]]
            e4 = ei[np.where(H[H[ei,4],0]==v4)[0]]
            if any(list(ej-ei)): # whose neighbor include not quad face
                if H[e1,1]==-1 and H[H[e2,4],1]==-1:
                    f4.append([v2,v,v1,-1])
                    f4.append([H[H[H[e2,2],2],0][0],v3,v,v2])
                    f4.append([v3,H[H[H[e3,2],2],0][0],v4,v])
                    f4.append([v,v4,H[H[H[e4,2],2],0][0],v1])
                elif H[e2,1]==-1 and H[H[e3,4],1]==-1:
                    f4.append([v2,v,v1,H[H[H[e1,2],2],0][0]])
                    f4.append([-1,v3,v,v2])
                    f4.append([v3,H[H[H[e3,2],2],0][0],v4,v])
                    f4.append([v,v4,H[H[H[e4,2],2],0][0],v1])
                elif H[e3,1]==-1 and H[H[e4,4],1]==-1:
                    f4.append([v2,v,v1,H[H[H[e1,2],2],0][0]])
                    f4.append([H[H[H[e2,2],2],0][0],v3,v,v2])
                    f4.append([v3,-1,v4,v])
                    f4.append([v,v4,H[H[H[e4,2],2],0][0],v1])
                elif H[e4,1]==-1 and H[H[e1,4],1]==-1:
                    f4.append([v2,v,v1,H[H[H[e1,2],2],0][0]])
                    f4.append([H[H[H[e2,2],2],0][0],v3,v,v2])
                    f4.append([v3,H[H[H[e3,2],2],0][0],v4,v])
                    f4.append([v,v4,-1,v1])
            else:
                if H[H[H[H[e1,2],2],2],0]==v2:
                    "one quad face [v2,v,v1,x]"
                    f4.append([v2,v,v1,H[H[H[e1,2],2],0][0]])
                if H[H[H[H[e2,2],2],2],0]==v3:
                    "one quad face [x,v3,v,v2]"
                    f4.append([H[H[H[e2,2],2],0][0],v3,v,v2])
                if H[H[H[H[e3,2],2],2],0]==v4:
                    "one quad face [v3,x,v4,v]"
                    f4.append([v3,H[H[H[e3,2],2],0][0],v4,v])
                if H[H[H[H[e4,2],2],2],0]==v1:
                    "one quad face [v,v4,x,v1]"
                    f4.append([v,v4,H[H[H[e4,2],2],0][0],v1])


        farr = np.unique(f4,axis=0)
        a,b = np.where(farr==-1)
        farr = np.delete(farr,a,axis=0)
        forder=np.array([],dtype=int)
        for f in farr:
            e1=np.where(H[:,0]==f[0])[0]
            e2=np.where(H[H[:,4],0]==f[1])[0]
            e = np.intersect1d(e1,e2)
            forder = np.r_[forder, H[e,1]]

        f4list = np.array(f4)
        if delete_multi: # delete multiple-left faces
            #forder, ind = np.unique(forder,return_index=True) # changed order
            ind=[]
            multi=[]
            for i in range(len(forder)):
                f = forder[i]
                if f not in forder[ind]:
                    ind.append(i)
                else:
                    j = np.where(forder[ind]==f)[0][0]
                    k = forder[ind][j]
                    l = np.setdiff1d(np.where(forder==k)[0], np.array([i]))[0]
                    multi.append(list(farr[l]))
            forder = forder[ind]
            farr = farr[ind]
            for f in multi:
                index=np.array([],dtype=int)
                e1,e2,e3,e4 = f
                a,b,c = [e4,e1,e2,e3],[e3,e4,e1,e2],[e2,e3,e4,e1]
                if a in f4:
                    ind,_ = np.where(f4list==a)
                    index = np.r_[index,ind]
                if b in f4:
                    ind,_ = np.where(f4list==b)
                    index = np.r_[index,ind]
                if c in f4:
                    ind,_ = np.where(f4list==c)
                    index = np.r_[index,ind]
                f4list[index]=np.array(f)

        self._num_rrf = len(farr)
        self._rr_quadface = farr
        self._rr_quadface_order = forder
        #print(len(farr),len(f4list),num,len(index))
        self._num_rrv = num # same with num_regular
        #order = np.setdiff1d(np.arange(num),index//4)
        self._rr_star = starM # same with starM
        self._rr_4quad_vers = f4 #rr_4quad_vers
        #return f4, farr, starM

    def get_rr_quadface_boundaryquad_index(self,vb_quad=True):
        v1,v2,v3,v4  = self.rr_quadface.T
        forder = self.rr_quadface_order
        fb = self.boundary_faces()
        _,out,_ = np.intersect1d(forder,fb, return_indices=True)
        inn = np.setdiff1d(np.arange(len(v1)), out)
        if vb_quad:
            "also including quad'vertex belong to boundary"
            boundary = self.boundary_vertices()
            vb = []
            for i in inn:
                if v1[i] in boundary or v2[i] in boundary:
                    vb.append(i)
                if v3[i] in boundary or v4[i] in boundary:
                    vb.append(i)
            inn = np.setdiff1d(inn,np.array(vb))
            out = np.r_[out,np.array(vb,dtype=int)]
        return inn, out

    def get_rr_vs_4face_corner(self):
        """self.rr_star_corner (should be same with get_vs_diagonal_v)
        vertex star' 4faces' 4 corner vertex [a,b,c,d]
           a   1    d
           2   v    4
           b   3    c
        """
        H = self.halfedges
        star = self.rr_star
        star = star[self.ind_rr_star_v4f4]
        v,v1,v2,v3,v4 = star.T
        va,vb,vc,vd = [],[],[],[]
        for i in range(len(v)):
            e1=np.intersect1d(np.where(H[:,0]==v[i])[0],np.where(H[H[:,4],0]==v1[i])[0])
            va.append(H[H[H[e1,2],2],0])
            e2=np.intersect1d(np.where(H[:,0]==v[i])[0],np.where(H[H[:,4],0]==v2[i])[0])
            vb.append(H[H[H[e2,2],2],0])            
            e3=np.intersect1d(np.where(H[:,0]==v[i])[0],np.where(H[H[:,4],0]==v3[i])[0])
            vc.append(H[H[H[e3,2],2],0])
            e4=np.intersect1d(np.where(H[:,0]==v[i])[0],np.where(H[H[:,4],0]==v4[i])[0])
            vd.append(H[H[H[e4,2],2],0])      
        self._rr_star_corner = np.c_[v,va,vb,vc,vd].T    
            
    def get_rr_vs_4face_centers(self): # no use!
        V = self.vertices
        f4 = self.rr_4quad_vers
        f41,f42,f43,f44 = f4[::4],f4[1::4],f4[2::4],f4[3::4]
        def _face_center(f4i):
            fic = (V[f4i[::1]]+V[f4i[1::4]]+V[f4i[2::4]]+V[f4i[3::4]]) / 4.0
            return fic
        f1c = _face_center(f41)
        f2c = _face_center(f42)
        f3c = _face_center(f43)
        f4c = _face_center(f44)
        return f1c,f2c,f3c,f4c
    
    def get_rr_vs_bounary(self):
        "seperate the inner/boundary rr-vertex"
        fb = self.boundary_faces()
        vsb = np.array([],dtype=int)
        for f in fb:
            vsb = np.r_[vsb,np.array(self.faces_list()[f])]
        vsb = np.unique(vsb)
        star = self.rr_star
        rrv = star[self.ind_rr_star_v4f4][:,0]   
        inn = np.setdiff1d(rrv,vsb)
        rrb = np.intersect1d(rrv,vsb)
        idi,idb = [],[]
        for i in inn:
            idi.append(np.where(rrv==i)[0][0])
        for j in rrb:
            idb.append(np.where(rrv==j)[0][0])
        return np.array(idi), np.array(idb)
    
    def index_of_4quad_face_order_at_regular_vs(self):
        """ most strong regular case: regular vertex & regular quads
        star = self.rr_star
        star = star[self.ind_rr_star_v4f4]
        if1,if2,if3,if4 = self.ind_rr_quadface_order.T
        """
        H = self.halfedges
        star = self.rr_star
        forder = self.rr_quadface_order
        flist = []
        ind = []
        for i in range(len(star)):
            v,v1,v2,v3,v4 = self.rr_star[i,:]
            e=np.where(H[:,0]==v)[0]
            e1=np.where(H[H[:,4],0]==v1)[0]
            e2=np.where(H[H[:,4],0]==v2)[0]
            e3=np.where(H[H[:,4],0]==v3)[0]
            e4=np.where(H[H[:,4],0]==v4)[0]
            i1 = np.intersect1d(e,e1)
            i2 = np.intersect1d(e,e2)
            i3 = np.intersect1d(e,e3)
            i4 = np.intersect1d(e,e4)
            f1,f2,f3,f4 = H[i1,1],H[i2,1],H[i3,1],H[i4,1]
            if f1!=-1 and f2!=-1 and f3!=-1 and f4!=-1:
                "for v whose faces are not 4, de-select it"
                if1 = np.where(forder==f1)[0]
                if2 = np.where(forder==f2)[0]
                if3 = np.where(forder==f3)[0]
                if4 = np.where(forder==f4)[0]
                if len(if1)!=0 and len(if2)!=0 and len(if3)!=0 and len(if4)!=0:
                    "for face who does belong to forder"
                    ind.append(i)
                    flist.append([if1[0],if2[0],if3[0],if4[0]])
        self._ind_rr_star_v4f4 = np.array(ind) # ind_rr_star_v4f4
        self._ind_rr_quadface_order = np.array(flist) 

    def index_of_multi_quadface_at_regular_vs(self,faceindex=True):
        """ind_multi_rr=ind_multi_quad_face
        get Q1234 corresponding quad face
        Q1:= F1P2 = F4P4
        Q2:= F1P1 = F2P3
        Q3:= F2P2 = F3P4
        Q4:= F3P3 = F4P1
        if Fi=-1, directly use its corresponding neighbour face' index
        eg: iq1 = [[id_f1, 2, id_f4, 4],...]
        """
        #f4,farr,_ = self.regular_vertex_regular_quad()
        farr = self.rr_quadface
        f4 = self.rr_4quad_vers
        fu = farr.tolist()
        f41,f42,f43,f44 = f4[::4],f4[1::4],f4[2::4],f4[3::4]
        num = self.num_rrv #==num_regular
        regularv = self.rr_star[:,0]
        if1,if2,if3,if4 = [],[],[],[]
        arr = np.arange(4*self.num_rrf).reshape(4,-1)
        i1,i11,i2,i22,i3,i33,i4,i44 = [],[],[],[],[],[],[],[]

        def _find_quad_index(arr,fu,f41,f44,i1,i11,n):
            def _find_index(arr,fu,f41,n): # n=1 % 4
                e1,e2,e3,e4 = f41
                a,b,c = [e4,e1,e2,e3],[e3,e4,e1,e2],[e2,e3,e4,e1]
                if f41 in fu:
                    i1 = arr[(n)%4,fu.index(f41)]# n=1
                elif a in fu:
                    i1 = arr[(n+3)%4,fu.index(a)]# n=0
                elif b in fu:
                    i1 = arr[(n+2)%4,fu.index(b)]
                elif c in fu:
                    i1 = arr[(n+1)%4,fu.index(c)]
                return i1
            if -1 not in f41 and -1 not in f44:
                a = _find_index(arr,fu,f41,n)
                aa = _find_index(arr,fu,f44,n+2)
                i1.append(a)
                i11.append(aa)
            elif -1 in f41:
                aa = _find_index(arr,fu,f44,n+2)
                i1.append(aa)
                i11.append(aa)
            elif -1 in f44:
                a = _find_index(arr,fu,f41,n)
                i1.append(a)
                i11.append(a)

        def _find_edge_index(arr,fu,f41,f42,f43,f44):
            def _find_index(arr,fu,f41,n):
                e1,e2,e3,e4 = f41
                a,b,c = [e4,e1,e2,e3],[e3,e4,e1,e2],[e2,e3,e4,e1]
                if f41 in fu:
                    i1 = arr[(n)%4,fu.index(f41)]   # e2
                    i2 = arr[(n+1)%4,fu.index(f41)] # e3
                    i3 = arr[(n+2)%4,fu.index(f41)] # e4
                    i4 = arr[(n+3)%4,fu.index(f41)] # e1
                elif a in fu:
                    i1 = arr[(n+3)%4,fu.index(a)]
                    i2 = arr[(n+4)%4,fu.index(a)]
                    i3 = arr[(n+5)%4,fu.index(a)]
                    i4 = arr[(n+6)%4,fu.index(a)]
                elif b in fu:
                    i1 = arr[(n+2)%4,fu.index(b)]
                    i2 = arr[(n+3)%4,fu.index(b)]
                    i3 = arr[(n+4)%4,fu.index(b)]
                    i4 = arr[(n+5)%4,fu.index(b)]
                elif c in fu:
                    i1 = arr[(n+1)%4,fu.index(c)]
                    i2 = arr[(n+2)%4,fu.index(c)]
                    i3 = arr[(n+3)%4,fu.index(c)]
                    i4 = arr[(n+4)%4,fu.index(c)]
                return [i1,i2,i3,i4]
            f1e = _find_index(arr,fu,f41,0) # face1: [e1,e2,e3,e4]
            f2e = _find_index(arr,fu,f42,0)
            f3e = _find_index(arr,fu,f43,0)
            f4e = _find_index(arr,fu,f44,0)
            return f1e,f2e,f3e,f4e

        blue,red = self.vertex_check
        a,b=[],[]
        vb,vr = [],[]
        c = 0
        for i in range(num):
            # Q1:= F1P2 = F4P4
            #_find_quad_index(arr,fu,f41[i],f44[i],i1,i11,1)
            _find_quad_index(arr,fu,f41[i],f44[i],i1,i11,1)

            # Q2:= F2P3 = F1P1
            #_find_quad_index(arr,fu,f41[i],f42[i],i2,i22,0)
            _find_quad_index(arr,fu,f42[i],f41[i],i2,i22,2)

            # Q3:= F3P4 = F2P2
            #_find_quad_index(arr,fu,f42[i],f43[i],i3,i33,1)
            _find_quad_index(arr,fu,f43[i],f42[i],i3,i33,3)

            # Q4:= F4P1 = F3P3
            #_find_quad_index(arr,fu,f43[i],f44[i],i4,i44,2)
            _find_quad_index(arr,fu,f44[i],f43[i],i4,i44,0)

            if faceindex:
                if -1 not in f41[i] and -1 not in f42[i] and -1 not in f43[i] and -1 not in f44[i]:
                    f1e,f2e,f3e,f4e = _find_edge_index(arr,fu,f41[i],f42[i],f43[i],f44[i])
                    if1.append(f1e)
                    if2.append(f2e)
                    if3.append(f3e)
                    if4.append(f4e)
                    if regularv[i] in blue:
                        a.append(c)
                        vb.append(regularv[i])
                    elif regularv[i] in red:
                        b.append(c)
                        vr.append(regularv[i])
                    c += 1

        if faceindex:
            if1,if2,if3,if4 = np.array(if1),np.array(if2),np.array(if3),np.array(if4)
            self._ind_multi_rr_face = [if1,if2,if3,if4] #ind_multi_rr_face
            self._vertex_rr_check_ind=[np.array(a),np.array(b)] #vertex_rr_check_ind
            self._vertex_rr_check = [np.array(vb),np.array(vr)]
        # i1,i2,i3,i4 = np.array(iq1),np.array(iq2),np.array(iq3),np.array(iq4)
        # self._ind_multi_quad_face = [i1,i2,i3,i4]
        i1,i2,i3,i4 = np.array(i1),np.array(i2),np.array(i3),np.array(i4)
        i11,i22,i33,i44 = np.array(i11),np.array(i22),np.array(i33),np.array(i44)
        self._ind_multi_quad_face = [i1,i2,i3,i4,i11,i22,i33,i44]

    def index_of_checker_vertex_pi_index_at_corner(self):
        """ vertex_check_corner_ind
        from oriented quad face [v1,v2,v3,v4],
        for corner quad face, get the index of checker Pi
        blue : V1==P4; V2=P2; V3=P2; V4=P4
        red  : V1==P1; V2=P1; V3=P3; V4=P3
        """
        corner = self.corner
        vblue,vred = self.checker_vertex
        vi = self.rr_quadface
        v1,v2,v3,v4 = vi.T
        num = len(v1)
        arr = np.arange(4*num).reshape(4,-1)
        indb,indr=[],[]
        for i in range(num):
            v1,v2,v3,v4 = vi[i]
            if v1 in corner and v1 in vblue:
                indb.append([v1,arr[3,i]])   #V1==P4
            if v2 in corner and v2 in vblue:
                indb.append([v2,arr[1,i]])   #V2=P2
            if v3 in corner and v3 in vblue:
                indb.append([v3,arr[1,i]])   #V3=P2
            if v4 in corner and v4 in vblue:
                indb.append([v4,arr[3,i]])   #V4=P4

            if v1 in corner and v1 in vred:
                indr.append([v1,arr[0,i]])   #V1==P1
            if v2 in corner and v2 in vred:
                indr.append([v2,arr[0,i]])   #V2=P1
            if v3 in corner and v3 in vred:
                indr.append([v3,arr[2,i]])   #V3=P3
            if v4 in corner and v4 in vred:
                indr.append([v4,arr[2,i]])   #V4=P3
        indb,indr = np.array(indb),np.array(indr)
        self._vertex_check_corner_ind = [indb,indr]


    def index_of_auxetic_cut_closed_qi(self):
        "ind_multi_close_boundary: from vertex index id1,id2 (as vertex star) to get q1,2,3,4"
        id1,id2 = self.join_vertex
        vblue,vred = self.checker_vertex
        H = self.halfedges
        i1,i11,i2,i22,i3,i33,i4,i44 = [],[],[],[],[],[],[],[]
        idbb,idbr = [],[]

        def _update(vs,vb1,vb2,ind,idbb,idbr):
            vs.append(vb1)
            if vb1 in vblue and vb2 in vblue:
                idbb.append(ind)
            elif vb1 in vred and vb2 in vred:
                idbr.append(ind)
            ind += 1
            return vs,ind,idbb,idbr

        def _index(arr,v1,v2,v3,v4,q2,q0,q1):
            v = [v1,v2,v3,v4]
            for i in range(4):
                a = np.where(v[i%4]==q2)[0]
                b = np.where(v[(i+1)%4]==q0)[0]
                c = np.where(v[(i+2)%4]==q1)[0]
                d = np.intersect1d(np.intersect1d(a,b),c)
                if len(d)!=0:
                    x,y = i,d[0]
                    ii = arr[(x+1)%4,y]
                    ijj = arr[x%4,y]
                    return x,ii,ijj

        vi = self.rr_quadface
        v1,v2,v3,v4 = vi.T
        arr = np.arange(4*len(v1)).reshape(4,-1)
        num = len(id1)
        ind = 0
        vs = []
        for i in range(num):
            vb1,vb2 = id1[i],id2[i]
            e1 = np.intersect1d(np.where(H[:,0]==vb1)[0],np.where(H[:,1]==-1)[0])[0]
            e11 = np.intersect1d(np.where(H[:,0]==vb2)[0],np.where(H[:,1]==-1)[0])[0]
            vb1q1,vb2q1 = H[H[e1,3],0], H[H[e11,4],0]
            vb1q3,vb2q3 = H[H[e1,4],0], H[H[e11,3],0]
            vb1q2,vb2q4 = H[H[H[H[e1,4],2],4],0], H[H[H[H[e11,4],2],4],0]

            if len(np.where(H[:,0]==vb1)[0])==3 or len(np.where(H[:,0]==vb2)[0])==3:
                #vs.append(vb1)

                if len(np.where(H[:,0]==vb1)[0])==3 and len(np.where(H[:,0]==vb2)[0])==3:
                    "check both vb1,vb2 of valence 3"
                    x,x1,y1 = _index(arr,v1,v2,v3,v4,vb1q2,vb1,vb1q1) # suppose f1
                    z2,x2,y2 = _index(arr,v1,v2,v3,v4,vb1q3,vb1,vb1q2) # suppose f2
                    z3,x3,y3 = _index(arr,v1,v2,v3,v4,vb2q4,vb2,vb2q3) # suppose f3
                    z4,x4,y4 = _index(arr,v1,v2,v3,v4,vb2q1,vb2,vb2q4) # suppose f4

                    if x==0:
                        a1,b1,a2,b2,a3,b3,a4,b4 = x1,y1,x2,y2,x3,y3,x4,y4
                    elif x==1:
                        a1,b1,a2,b2,a3,b3,a4,b4 = x4,y4,x1,y1,x2,y2,x3,y3
                    elif x==2:
                        a1,b1,a2,b2,a3,b3,a4,b4 = x3,y3,x4,y4,x1,y1,x2,y2
                    elif x==3:
                        a1,b1,a2,b2,a3,b3,a4,b4 = x2,y2,x3,y3,x4,y4,x1,y1
                    i1.append(a1)
                    i22.append(b1)
                    i2.append(a2)
                    i33.append(b2)
                    i3.append(a3)
                    i44.append(b3)
                    i4.append(a4)
                    i11.append(b4)
                    print('3,3:',x,z2,z3,z4)
                    vs,ind,idbb,idbr = _update(vs,vb1,vb2,ind,idbb,idbr)

                ### for torus_2genus.obj, should comment the (2,3), (3,2) case below:
                elif len(np.where(H[:,0]==vb1)[0])==2 and len(np.where(H[:,0]==vb2)[0])==3:
                    if vb1q2==vb1q1:
                        "x=0: i11,i4; x=1: i22,i1; x=2: i33,i2; x=3: i44,i3"
                        x,x2,y2 = _index(arr,v1,v2,v3,v4,vb1q3,vb1,vb1q2) # suppose f2
                        z3,x3,y3 = _index(arr,v1,v2,v3,v4,vb2q4,vb2,vb2q3) # suppose f3
                        z4,x4,y4 = _index(arr,v1,v2,v3,v4,vb2q1,vb2,vb2q4) # suppose f4

                        if x==0:
                            a1,b1,a2,b2,a3,b3,a4,b4 = x2,y2,x3,y3,x4,y4,y4,x2
                        elif x==1:
                            a1,b1,a2,b2,a3,b3,a4,b4 = y4,x2,x2,y2,x3,y3,x4,y4
                        elif x==2:
                            a1,b1,a2,b2,a3,b3,a4,b4 = x4,y4,y4,x2,x2,y2,x3,y3
                        elif x==3:
                            a1,b1,a2,b2,a3,b3,a4,b4 = x3,y3,x4,y4,y4,x2,x2,y2
                        i1.append(a1)
                        i22.append(b1)
                        i2.append(a2)
                        i33.append(b2)
                        i3.append(a3)
                        i44.append(b3)
                        i4.append(a4)
                        i11.append(b4)
                        print('2,3:',x,z3,z4)
                        vs,ind,idbb,idbr = _update(vs,vb1,vb2,ind,idbb,idbr)
                    else: # seems no use.
                        print(vb1,vb1q2,vb1q3)
                        print('need to check!')

                elif len(np.where(H[:,0]==vb1)[0])==3 and len(np.where(H[:,0]==vb2)[0])==2:
                    if vb2q4==vb2q3:
                        "x=0: i11,i4; x=1: i22,i1; x=2: i33,i2; x=3: i44,i3"
                        x,x1,y1 = _index(arr,v1,v2,v3,v4,vb1q2,vb1,vb1q1) # suppose f1
                        z2,x2,y2 = _index(arr,v1,v2,v3,v4,vb1q3,vb1,vb1q2) # suppose f2
                        z4,x4,y4 = _index(arr,v1,v2,v3,v4,vb2q1,vb2,vb2q4) # suppose f4
                        if x==0:
                            a1,b1,a2,b2,a3,b3,a4,b4 = x1,y1,x2,y2,y2,x4,x4,y4
                        elif x==1:
                            a1,b1,a2,b2,a3,b3,a4,b4 = x4,y4,x1,y1,x2,y2,y2,x4
                        elif x==2:
                            a1,b1,a2,b2,a3,b3,a4,b4 = y2,x4,x4,y4,x1,y1,x2,y2
                        elif x==3:
                            a1,b1,a2,b2,a3,b3,a4,b4 = x2,y2,y2,x4,x4,y4,x1,y1
                        i1.append(a1)
                        i22.append(b1)
                        i2.append(a2)
                        i33.append(b2)
                        i3.append(a3)
                        i44.append(b3)
                        i4.append(a4)
                        i11.append(b4)
                        print('3,2:',x,z2,z4)
                        vs,ind,idbb,idbr = _update(vs,vb1,vb2,ind,idbb,idbr)
                    else: # seems no use.
                        print(vb2,vb2q4,vb2q3)
                        print('need to check!')

                ## if vb1 in vblue and vb2 in vblue:
                ##     idbb.append(ind)
                ## elif vb1 in vred and vb2 in vred:
                ##     idbr.append(ind)
                ## ind += 1

        vs = np.array(vs)
        i1,i2,i3,i4 = np.array(i1),np.array(i2),np.array(i3),np.array(i4)
        i11,i22,i33,i44 = np.array(i11),np.array(i22),np.array(i33),np.array(i44)
        idbb,idbr = np.array(idbb), np.array(idbr)
        #print(num,ind,len(vs),len(idbb),len(idbr))
        self._ind_multi_close_boundary = [vs,[i1,i2,i3,i4,i11,i22,i33,i44],[idbb,idbr]]


    def make_quad_mesh_pieces(self,P1,P2,P3,P4):
        vlist = np.vstack((P1,P2,P3,P4))
        num = len(P1)
        arr = np.arange(num)
        flist = np.c_[arr,arr+num,arr+2*num,arr+3*num].tolist()
        ck = Mesh()
        ck.make_mesh(vlist,flist)
        return ck
    
    def make_strip_quad_mesh(self,Pi,Qi):
        "given two boundary vertices, get quad mesh"
        P1, P2 = Pi[:-1],Pi[1:]
        P4, P3 = Qi[:-1],Qi[1:]
        sm = self.make_quad_mesh_pieces(P1,P2,P3,P4)
        return sm
        
    def make_quad_mesh_from_indices(self,V,v1,v2,v3,v4):
        vrr = np.unique(np.r_[v1,v2,v3,v4])
        numf = len(v1)
        vlist = V[vrr]
        farr = np.array([],dtype=int)
        for i in range(numf):
            i1 = np.where(vrr==v1[i])[0]
            i2 = np.where(vrr==v2[i])[0]
            i3 = np.where(vrr==v3[i])[0]
            i4 = np.where(vrr==v4[i])[0]
            farr = np.r_[farr,i1,i2,i3,i4]
        flist = farr.reshape(-1,4).tolist()
        ck = Mesh()
        ck.make_mesh(vlist,flist)
        return ck

    def make_patch_mesh_from_matrix(self,P,patch_matrix):
        "all patch_matrix_quad"
        v1 = patch_matrix[:-1,:-1].flatten() ##left-up
        v2 = patch_matrix[1:,:-1].flatten() ## left-down
        v3 = patch_matrix[1:,1:].flatten() ##right-down
        v4 = patch_matrix[:-1,1:].flatten() ## righg-up
        return self.make_quad_mesh_pieces(P[v1],P[v2],P[v3],P[v4])
    
    def make_diag_patch_mesh_from_matrix(self,P,patch_matrix,type1=True):
        "patch_mesh_diagonal mesh: diamond shape"
        a,b = patch_matrix.shape 
        if type1:
            if a%2==0:
                a -=1
            if b%2==0:
                b -=1    
            M = patch_matrix[:a,:b]
        else:
            if a%2==0 and b%2==0: 
                M = patch_matrix[1:,1:]
        v1 = M[:-2,1:-1][::2,::2].flatten() # top
        v2 = M[1:-1,:-2][::2,::2].flatten() # mid-left
        v3 = M[2:,1:-1][::2,::2].flatten() # down
        v4 = M[1:-1,2:][::2,::2].flatten() # mid-right
        return self.make_quad_mesh_pieces(P[v1],P[v2],P[v3],P[v4])
    
    def make_denser_diag_patch_mesh_from_matrix(self,P,patch_matrix):
        "denser patch_mesh_diagonal mesh: diamond shape using barycenters"
        a,b = patch_matrix.shape
        v1 = patch_matrix[:-1,:-1].flatten() ##left-up
        v2 = patch_matrix[1:,:-1].flatten() ## left-down
        v3 = patch_matrix[1:,1:].flatten() ##right-down
        v4 = patch_matrix[:-1,1:].flatten() ## righg-up
        bary = (P[v1]+P[v2]+P[v3]+P[v4])/4.0
        allV = np.vstack((P, bary))
        Mbary = len(P) + np.arange(len(bary)).reshape(a-1,b-1)
        bl = Mbary[:,:-1].flatten()
        br = Mbary[:,1:].flatten()
        bu = Mbary[:-1,:].flatten()
        bd = Mbary[1:,:].flatten()
        w1 = patch_matrix[:-1,1:-1].flatten() ##up
        w2 = patch_matrix[1:-1,:-1].flatten() ##mid-left
        w3 = patch_matrix[1:,1:-1].flatten() ##down
        w4 = patch_matrix[1:-1,1:].flatten() ##mid-right
        p1 = np.r_[w1,bu]
        p2 = np.r_[bl,w2]
        p3 = np.r_[w3,bd]
        p4 = np.r_[br,w4]
        return self.make_quad_mesh_pieces(allV[p1],allV[p2],allV[p3],allV[p4])
    
    def orient(self,S0,A,B,C,D,Nv4):
        ind1 = np.where(np.einsum('ij,ij->i',self.orientrn,Nv4) < 0)[0]
        if len(ind1)!=0:
            Nv4[ind1] = -Nv4[ind1]
            "new_c = v[ind1]+r[ind1]*n[ind1], --> (b,c,d)[ind1] = -2*a*(v+rn)[ind1]"
            # x,y,z = S0.T
            # bb = -4*A*x-B
            # cc = -4*A*y-C
            # dd = -4*A*z-D
            # B[ind1] = bb[ind1]
            # C[ind1] = cc[ind1]
            # D[ind1] = dd[ind1]
        x_orient = np.sqrt(np.abs(np.einsum('ij,ij->i',Nv4,self.orientrn)))        
        return B,C,D,Nv4,x_orient

    def frame_atregularvertices(self,diag=False): # no use
        "assign (t,n,b) at all vertices, but only inner usefule"
        V = self.vertices
        if diag:
            v,v1,v2,v3,v4 = self.rr_star_corner
        else:
            v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        from frenet_frame import FrenetFrame
        frame1 = FrenetFrame(V[v],V[v1],V[v3])
        frame2 = FrenetFrame(V[v],V[v2],V[v4])
        return frame1, frame2
    
    def new_vertex_normals(self):
        "make vertex_normals orient non-convex direction"
        v0,v1,v2,v3,v4 = (self.rr_star).T
        Ns = self.vertex_normals()[v0]
        V = self.vertices
        uE1 = (V[v1]-V[v0]) / np.linalg.norm(V[v1]-V[v0],axis=1)[:,None]
        uE2 = (V[v2]-V[v0]) / np.linalg.norm(V[v2]-V[v0],axis=1)[:,None]
        uE3 = (V[v3]-V[v0]) / np.linalg.norm(V[v3]-V[v0],axis=1)[:,None]
        uE4 = (V[v4]-V[v0]) / np.linalg.norm(V[v4]-V[v0],axis=1)[:,None]
        N1, N2 = uE1 + uE3, uE2 + uE4
        id1 = np.where(np.einsum('ij,ij->i', Ns, N1) < 0)[0] # non-convex direction
        id2 = np.where(np.einsum('ij,ij->i', Ns, N2) < 0)[0] # non-convex direction
        number = max(len(id1),len(id2))
        if number > len(v0)-len(id1)-len(id2):
            return self.vertex_normals()
        else:
            return -self.vertex_normals()

    def vertex_valence3_neib(self,boundary_vertex=True,corner=False):
        v,vj,lj = self.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
        ##o2 = np.where(lj==2)[0]
        o3 = np.where(lj==3)[0]
        ##o4 = np.where(lj==4)[0]
        boundary = self.boundary_vertices()
        ##self.bvalen2 = np.intersect1d(o2, boundary)

        if boundary_vertex:
            bvalen3 = np.intersect1d(o3, boundary)
            if len(bvalen3)==0:
                pass
            else:
                H = self.halfedges
                vl,vr = [],[]
                for v in bvalen3:
                    "in order"
                    ie = np.where(H[:,0]==v)[0]
                    i = np.intersect1d(np.where(H[ie,1]!=-1)[0],np.where(H[H[ie,4],1]!=-1)[0])
                    vl.append(H[H[ie[i],3],0][0])
                    vr.append(H[H[H[H[ie[i],4],2],4],0][0])                    
                self._ver_bod_valence3_neib = [bvalen3,np.array(vl),np.array(vr)]
        else:
            invalen3 = np.setdiff1d(o3,boundary)
            if len(invalen3)==0:
                pass
            neib = []
            for v in invalen3:
                neib.append(self.ringlist[v])
            self._ver_inn_valence3_neib = [invalen3, np.array(neib)]
        
    def vertex_corner_valence3_neib(self):
        v,vl,vr = self.ver_bod_valence3_neib
        c = self.corner
        ic = []
        for i in range(len(v)):
           if vl[i] in c or vr[i] in c:
               ic.append(i)
        ic = np.array(ic)
        return v[ic],vl[ic],vr[ic]   
        
    def vertex_valence5_neib(self):
        v,vj,lj = self.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
        o5 = np.where(lj==5)[0]
        boundary = self.boundary_vertices()
        inv5 = np.setdiff1d(o5, boundary)   
        if len(inv5)==0:
            pass
        else:
            neib = []
            for v in inv5:
                neib.append(self.ringlist[v])
            self._ver_inn_valence5_neib = [inv5, np.array(neib)]
        
    def vertex_valence6_neib(self):
        v,vj,lj = self.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
        o6 = np.where(lj==6)[0]
        boundary = self.boundary_vertices()
        inv6 = np.setdiff1d(o6, boundary)   
        if len(inv6)==0:
            pass
        else:
            neib = []
            for v in inv6:
                neib.append(self.ringlist[v])
            self._ver_inn_valence6_neib = [inv6, np.array(neib)]     
            
    def vertex_valence4_neib(self,corner=True):
        "v, neib=[v1,v2,v3,v4,va,vb,vc,vd]"
        _,_,lj = self.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
        o4 = np.where(lj==4)[0]
        boundary = self.boundary_vertices()
        inv4 = np.setdiff1d(o4, boundary)   
        if len(inv4)==0:
            pass
        else:
            H = self.halfedges
            vc4 = []
            neib = np.array([],dtype=int)
            for v in inv4:
                if len(np.intersect1d(self.ringlist[v],boundary))==2:
                    vc4.append(v)
                    ie = np.where(H[:,0]==v)[0]
                    abcd = H[H[H[ie,2],2],0]
                    neib = np.r_[neib,abcd,np.array(self.ringlist[v])]
            neib = neib.reshape(-1,8)
            self._ver_corner_valence4_neib = [np.array(vc4), neib]      
        
    def vertex_corner_neib(self):
        H = self.halfedges
        corner = self.corner
        el = []
        for i in range(len(corner)):
            c = corner[i]
            e = np.intersect1d(np.where(H[:,0]==c)[0],np.where(H[:,1]==-1)[0])[0]
            el.append(e)
        va,vb = H[H[el,2],0],H[H[H[el,2],2],0]
        v1,v2 = H[H[el,3],0],H[H[H[el,3],3],0]
        vl,vc,vr = np.r_[corner,corner],np.r_[va,v1],np.r_[vb,v2]
        self._ver_corner_neib = [vl,vc,vr]
        
    def index_of_checker_vertex_near_boundary(self):
        "idb,idr = self.vertex_check_near_boundary_ind "
        vb,vr = self.vertex_rr_check
        boundary = self.boundary_vertices()
        regularv = self.rr_star[:,0]
        indd1,indd2 = [],[]
        for i in range(len(vb)):
            v = vb[i]
            j = np.where(regularv==v)[0][0]
            _,v1,_,v3,_ = self.rr_star[j,:]
            if v1 in boundary or v3 in boundary:
                indd1.append(i)
        for i in range(len(vr)):
            v = vr[i]
            j = np.where(regularv==v)[0][0]
            _,_,v2,_,v4 = self.rr_star[j,:]
            if v2 in boundary or v4 in boundary:
                indd2.append(i)
        idb,idr = self.vertex_rr_check_ind
        if len(indd1)==0: # note: like torus_top.obj may have all blue boundary
            idb = None
        elif len(indd2)==0:
            idr = None
        else:
            idb,idr = idb[np.array(indd1)], idr[np.array(indd2)]
        self._vertex_check_near_boundary_ind = [idb,idr]

    def make_triangle_mesh_pieces(self,P1,P2,P3):
        vlist = np.vstack((P1,P2,P3))
        num = len(P1)
        arr = np.arange(num)
        flist = np.c_[arr,arr+num,arr+2*num].tolist()
        ck = Mesh()
        ck.make_mesh(vlist,flist)
        return ck

    def get_triangulated_quad_mesh(self,loop=0):
        tv = self.vertices
        tf, f = self.face_triangles()
        ###A,B,C = V[Tri[:,0]],V[Tri[:,1]],V[Tri[:,2]]
        for i in range(loop):
            tv,tf = igl.loop(tv,tf)
        tri = Mesh()
        tri.make_mesh(tv,tf)
        return tri
    
    def get_geodesic_disk(self,i,verlist,facelist,exact=False,heat=False,loop=False,upsample=False,frequency=True):
        if loop:
            verlist,facelist = igl.loop(verlist,facelist,number_of_subdivs=1)
        elif upsample:
            verlist,facelist = igl.upsample(verlist,facelist,number_of_subdivs=1)
        ##tri = Mesh()
        ##tri.make_mesh(verlist,facelist)
        num = verlist.shape[0]
        vs, vt = np.array([i],dtype=int), np.arange(num,dtype=int)
        if exact:
            d = igl.exact_geodesic(verlist,facelist,vs,vt)
        elif heat:
            d = igl.heat_geodesic(verlist,facelist,t=0.1,gamma=vs)
        ##print('len=',len(d))
        ##print('geodesic distance: max,mean =',np.max(d),np.mean(d))
        if frequency:
            "note: the bigger strip_size, the denser color_frequency"
            strip_size = 0.2
            d = np.abs(np.sin((d*strip_size*np.pi)))
        return verlist,np.array(facelist), d    
    
    def get_geodesic_disk_circle(self,i,verlist=None,facelist=None,allv=True,num=5,exact=False,heat=False,loop=False,upsample=False):
        "for global geodesic disk circle polyline"
        if allv:
            V = self.vertices
            Tri, f = self.face_triangles()
            verlist,facelist = V, Tri
        trv,trf,geo_dist = self.get_geodesic_disk(i,verlist,facelist,exact,heat,loop,upsample)
        isov,isoe = igl.isolines(trv,trf,geo_dist,n=num)
        vv = np.vstack((isov[isoe[:,0]],isov[isoe[:,1]]))
        data = self.diagonal_cell_array(isoe[:,0])
        poly = Polyline(vv)
        poly.cell_array=data
        #poly.refine()
        #vv = poly.vertices   
        return poly      
    
    def get_global_geodesic_disk(self,i=0,exact=False,heat=False,loop=False,upsample=False,frequency=True):
        "for selected vertex i, get all/global vertices' geodesic distance to i"
        V = self.vertices
        Tri, f = self.face_triangles()
        verlist,facelist = V, Tri
        trv,trf,d = self.get_geodesic_disk(i,verlist,facelist,exact,heat,loop,upsample,frequency)
        tri = Mesh()
        tri.make_mesh(trv,trf)        
        return tri,d
        
    def get_local_geodesic_disk(self,i,ring,forder,exact=False,heat=False,loop=False,upsample=False,frequency=True):
        tm_v,tm_f,i,ind = self.get_local_triangulated_quad_mesh(i,ring,forder)
        trv,trf,d = self.get_geodesic_disk(i,tm_v,tm_f,exact,heat,loop,upsample,frequency)
        # if switch: # false for colored faces, true for colored vertices
        #     """
        #     if True: d in coordinate with trv_vorder[ind]=ring
        #     else: in in coordinate with trv_vertex
        #     """
        #     d = d[ind]
        return trv,trf,d, d[ind]
    
    def get_local_triangulated_quad_mesh(self,vs,ring,forder,trimesh=False):
        "given i-->i_ring-->ring_face_order"
        V = self.vertices
        Tri, f = self.face_triangles()
        ind = np.array([],dtype=int)
        for i in forder:
            ind = np.r_[ind,np.where(f==i)[0]]
        vorder = np.unique(Tri[ind].flatten())
        vs = np.where(vorder==vs)[0]
        ##vt = np.arange(len(vorder))
        ind_ring = np.array([],dtype=int)
        for v in ring:
            "trv: vorder[ind_ring] = ring"
            ind_ring = np.r_[ind_ring,np.where(vorder==v)[0]]        
        tm_v = V[vorder]
        i,j,k = [],[],[]
        for tr in Tri[ind]:
            i.append(np.where(vorder==tr[0])[0][0])
            j.append(np.where(vorder==tr[1])[0][0])
            k.append(np.where(vorder==tr[2])[0][0])
        tm_f = np.c_[i,j,k]
        if trimesh:
            tm = Mesh()
            tm.make_mesh(tm_v,tm_f)         
            return tm, tm_v, tm_f, vs, ind_ring
        else:
            return tm_v, tm_f, vs, ind_ring
        
    def get_gobal_triangulated_rr_quad_mesh(self,ik):
        "given i-->i-quad-ring-->i-quad-face"
        forder = self.rr_quadface_order
        V = self.vertices
        Tri, f = self.face_triangles()
        ind = np.array([],dtype=int)
        for i in forder:
            "f[ind] = forder"
            ind = np.r_[ind,np.where(f==i)[0]]
        vorder = np.unique(Tri[ind].flatten())   
        vs = np.where(vorder==ik)[0]
        tm_v = V[vorder]
        i,j,k = [],[],[]
        for tr in Tri[ind]:
            i.append(np.where(vorder==tr[0])[0][0])
            j.append(np.where(vorder==tr[1])[0][0])
            k.append(np.where(vorder==tr[2])[0][0])
        tm_f = np.c_[i,j,k]        
        return tm_v, tm_f, vs, vorder,forder 
        
    def _get_nearest_point_from_point_cloud(self, allP, startP, inter_dist):
        "given allpoints and start P \in allP, set deta, return ordered P"
        arr = np.arange(len(allP))
        ind = np.argmin(np.linalg.norm(allP-startP, axis=1))
        nextP = allP[ind]

        iv = np.array([ind],dtype=int)  
        arr = np.delete(arr,ind)
        ind1 = np.argmin(np.linalg.norm(allP[arr]-nextP,axis=1)) 
        
        def _get_ind(arr,ind1,iv,Q0):
            Q1 = allP[arr[ind1]]
            deta = np.linalg.norm(Q1-Q0)
            while deta < inter_dist and len(arr)!=1:
                iv = np.r_[iv,arr[ind1]]
                Q1 = allP[arr[ind1]]
                arr = np.delete(arr,ind1)
                ind1 = np.argmin(np.linalg.norm(allP[arr]-Q1,axis=1)) 
                Q2 = allP[arr[ind1]]
                deta = np.linalg.norm(Q2-Q1)
            return arr,iv
                    
        arr,ivv = _get_ind(arr,ind1,iv,nextP)
        if len(arr)!=0:
            ind2 = np.argmin(np.linalg.norm(allP[arr]-nextP,axis=1)) 
            arr,iv2 = _get_ind(arr,ind2,iv,nextP)
            ###print(iv,ivv,iv2)
            if len(iv2)==1:
                #ivv = np.r_[iv2,ivv]
                pass
            else:
                ivv = np.r_[iv2[::-1],ivv]       
        return ivv
        
    
    def _get_sub_geodesic_disk_polyline(self,trv,trf,geo_dist,radius):
        """with given sub-ring triangulated vertices & faces & geo_dist
        0. trv whose geo_dist=r, append[ind_v]
        1. vertex whose geo_dist > radius
        2. corresponding triangular faceslist
        3. check each trf_list: [v1,v2,v3] whose geo_dist [d1,d2,d3]
            if d1,d2,d3>r:
                pass
            if d2,d3<r,d1>r:
                P3 = c*V1 + (1-c)*V2; c=(r-d2)/(d1-d2)
                P4 = e*V1 + (1-e)*V3; e=(r-d3)/(d1-d3)
            if d3<r,d1,d2>r:
                P1 = a*V1 + (1-a)*V3; a=(r-d3)/(d1-d3)
                P2 = b*V2 + (1-b)*V3; b=(r-d3)/(d2-d3)
        4. use nearest_point to get ordered polyline_points
        """
        arr = np.arange(len(trv))
        if any(geo_dist==radius):
            Ver = trv[geo_dist==radius].flatten()
        else:
            Ver = np.array([])
        outer = arr[geo_dist>radius]
        for v in outer:
            row = np.where(trf==v)[0]
            for x in row:
                if any(geo_dist[trf[x]]==radius):
                    continue 
                ind = np.where(geo_dist[trf[x]]>radius)[0]
                if len(ind)==3:
                    continue    
                brr = np.r_[ind,np.setdiff1d(np.arange(3),ind)]
                V1,V2,V3 = trv[trf[x,brr]]
                d1,d2,d3 = geo_dist[trf[x,brr]]
                if len(ind)==1:
                     a = (radius-d2)/(d1-d2)
                     b = (radius-d3)/(d1-d3)
                     P1 = a*V1 + (1-a)*V2
                     P2 = b*V1 + (1-b)*V3
                elif len(ind)==2:
                     a = (radius-d3)/(d1-d3)
                     b = (radius-d3)/(d2-d3)
                     P1 = a*V1 + (1-a)*V3
                     P2 = b*V2 + (1-b)*V3
                Ver = np.r_[Ver,P1,P2]
        Ver = Ver.reshape(-1,3)   
        ###print(Ver.shape)
        startP = Ver[0]
        v1,v2,v3,v4 =self.rr_quadface.T
        Vi = self.vertices
        inter_dist = np.max(np.linalg.norm(Vi[np.r_[v1,v2]]-Vi[np.r_[v3,v4]],axis=1))
        ivv = self._get_nearest_point_from_point_cloud(Ver, startP, inter_dist)
  
        if True: # if only one circle line, for smoother curves
            if np.linalg.norm(Ver[ivv[0]]-Ver[ivv[-1]]) < inter_dist:
                poly = Polyline(Ver[ivv],closed=True)  
            else:
                poly = Polyline(Ver[ivv],closed=False)  
            N = 5
            poly.refine(N) # round the corner
            return [poly,Ver[0]]
        else: # sparse curves
            if np.linalg.norm(Ver[ivv[0]]-Ver[ivv[-1]]) < inter_dist:
                Vl,Vr = Ver[ivv], Ver[np.r_[ivv[1:],ivv[0]]]
                data = self.diagonal_cell_array(ivv)
            else:
                Vl,Vr = Ver[ivv[:-1]], Ver[ivv[1:]]
                data = self.diagonal_cell_array(ivv[:-1])
                poly = Polyline(np.vstack((Vl,Vr)))
                poly.cell_array=data
                return [poly, Ver[0]]   


    def _get_geodesic_circle_isoline(self,trv,trf,geo_dist,depth,radius):
        "get circle isolines; trv,trf,geo_dist from rectangle ring&faces"
        interval=depth
        max_geo = np.max(geo_dist)
        err = radius * 0.01
        for i in range(depth):
            j = i+1
            for k in range(j):
                a = abs(max_geo*k/j - radius)
                if a<err:
                    interval = j
                    break
        print('-'*30)
        print('interval of isolines',interval)
        print('-'*30)
        isov,isoe = igl.isolines(trv,trf,geo_dist,n=interval)
        vv = np.vstack((isov[isoe[:,0]],isov[isoe[:,1]]))
        data = self.diagonal_cell_array(isoe[:,0])
        poly = Polyline(vv)
        poly.cell_array=data
        return [poly,vv[0]]         

    def get_rectangle_ring_at_a_vertex(self,ik,time,step=3,exact=True,heat=False,loop=False,upsample=False,show=False):
        """ at vertex vi, getting radius = k*mean_edge_length geodesic disk
        ik: vertex index
        k: time of mean_edge_length
        
        return:
            ring vertex indices of vi
            geodesic distance of all vj to vi
            --> weight assigned to each vj : 
                if |vj-vi|<r:  weight = 1
                else: weight = ||vj-vi|-r|/ |vj-vi|
        
        note: still limit in square-box-k-rings
        """
        Vi = self.vertices
        radius = time * self.mean_edge_length()
        H = self.halfedges
        ei = np.where(H[:,0]==ik)[0]
        dis = np.linalg.norm(Vi[H[H[ei,4],0]]-Vi[H[ei,0]],axis=1)
        "full or half (continuous) geodesic circle"
        num = []
        for i in range(len(ei)):
            ej = ei[i]
            n = 1
            while dis[i]<radius and H[H[ej,2],4]!=-1:
                ej = H[H[H[ej,2],4],2]
                dis[i] += np.linalg.norm(Vi[H[H[ej,4],0]]-Vi[H[ej,0]])       
                n += 1
            num.append(n)
        depth = max(num)     
        "get rectangle region of ring_vertices and faces"
        ring,faces = self.vertex_multiple_ring_vertices_faces(ik,depth)
        
        trv,trf,_,geo_dist = self.get_local_geodesic_disk(ik,ring,faces,exact,heat,loop,upsample,frequency=False)

        if show:
            print('-'*30)
            print('time of mean_edge_length',time)
            print('radius=',radius)
            print('ring_depth=', depth)
            #print('local geodesic distance:\n',describe(geo_dist))
            print('-'*30)
            poly = self._get_geodesic_circle_isoline(trv,trf,geo_dist,depth,radius)  
            trv,trf,geo_dist,_ = self.get_local_geodesic_disk(ik,ring,faces,exact,heat,loop,upsample,frequency=True)
            tri = Mesh()
            tri.make_mesh(trv,trf)   
            return tri,geo_dist,poly          

        "set weights for ring-vertices, lamda=1 if inside, else=cos()"
        arr = np.zeros(self.V)#np.zeros(len(ring))
        arr[ring] = 1
        return ring,faces,geo_dist,arr                   

    def get_geodesic_disk_blur_quads(self,ik,ring,geo_dist,radius,nonzero=True):
        """two ways to get smaller circular disk: 
            one by k-ring as upper-limit;
            second using all rr-vertices as ring, slower to show
        "from weight=1(inside geodesic disk), get relevant sub-ring-quads without zero-dist."
        "sub_vertex-->sub_face-->sub_ring"            
        """

        sub_ring = sub_ring_face = sub_face = np.array([],dtype=int)
        sub_vertex= ring[geo_dist<=radius]
        H = self.halfedges
        for v in sub_vertex:
            f = H[np.where(H[:,0]==v)[0],1]
            sub_face = np.r_[sub_face,f]
        sub_face = np.unique(sub_face[np.where(sub_face>=0)])
        allf,allv = self.face_vertices_iterators()
        
        if nonzero:
            for f in sub_face:
                facev = allv[np.where(allf==f)[0]]
                check=True
                for j in facev:
                    a = np.where(ring==j)[0]
                    if geo_dist[a] == 0 and ring[a]!=ik:
                        check=False
                if check:
                    sub_ring = np.r_[sub_ring,facev]
                    sub_ring_face = np.r_[sub_ring_face,f]
            sub_ring_face = np.unique(sub_ring_face)
        else:
            for f in sub_face:
                sub_ring = np.r_[sub_ring,allv[np.where(allf==f)[0]]]
            sub_ring_face = sub_face
            
        sub_ring = np.unique(sub_ring)
        return sub_ring,sub_ring_face
    
    def get_geodesic_disk_inside_quads(self,ring,geo_dist,radius):
        sub_ring = sub_ring_face = sub_face = np.array([],dtype=int)
        #print(geo_dist.shape,ring.shape)
        sub_vertex= ring[geo_dist<=radius]
        H = self.halfedges
        for v in sub_vertex:
            f = H[np.where(H[:,0]==v)[0],1]
            sub_face = np.r_[sub_face,f]
        sub_face = np.unique(sub_face[np.where(sub_face>=0)])
        allf,allv = self.face_vertices_iterators()
        for f in sub_face:
            a = np.setdiff1d(allv[np.where(allf==f)[0]],sub_vertex)
            if len(a)==0:
                sub_ring = np.r_[sub_ring,allv[np.where(allf==f)[0]]]
                sub_ring_face = np.r_[sub_ring_face,f]
        sub_ring = np.unique(sub_ring)
        sub_ring_face = np.unique(sub_ring_face)
        return sub_ring,sub_ring_face

    def get_small_geodesic_disk_at_a_vertex(self,ik,time,step=3,exact=True,heat=False,loop=False,upsample=False,show=False):
        ## no use!
        radius = time * self.mean_edge_length()
        tm_v,tm_f,vs,ring,faces = self.get_gobal_triangulated_rr_quad_mesh(ik)
        _,_,geo_dist = self.get_geodesic_disk(vs,tm_v,tm_f,exact,heat,loop,upsample,frequency=False)

        ring,faces = self.get_geodesic_disk_blur_quads(ik,ring,geo_dist,radius)

        trv,trf,trv_geo_dist,geo_dist = self.get_local_geodesic_disk(ik,ring,faces,exact,heat,loop,upsample,frequency=False)

        if show:
            poly = self._get_sub_geodesic_disk_polyline(trv,trf,trv_geo_dist,radius)
            trv,trf,geo_dist,_ = self.get_local_geodesic_disk(ik,ring,faces,exact,heat,loop,upsample,frequency=True)
            tri = Mesh()
            tri.make_mesh(trv,trf)   
            return tri,geo_dist,poly    

        else:
            "set weights for ring-vertices, lamda=1 if inside, else=cos()"
            arr = np.zeros(self.V)#np.zeros(len(ring))
            for i in range(len(ring)):
                "geo_dist is one-to-one map to ring_vertex_index"
                if geo_dist[i]<=radius:
                    arr[ring[i]] = 1
                else:
                    k = (geo_dist[i]-radius) / (np.max(geo_dist)-radius)
                    arr[ring[i]] = np.cos(np.pi / step * k) # here a=2,3,4,5,6...
            return ring,faces,geo_dist,arr  

        
    def get_geodesic_disk_at_a_vertex(self,ik,time,step,gtype,exact=True,heat=False,loop=False,upsample=False,show=False):
        """
        ----------
        ik : the ik-th vertex
        time : time of mean_edge_length
        step : decreased circle outer weights
        gtype : rectangle,circle_inside,circle_blur,circle_blur_big
        exact : igl.exact_geodesic
        heat : igl.heat_geodesic
        loop : loop_subdivision
        upsample : upsample_subdivision
        show : or not
        
        Returns
        -------
        """
        print('gtype=',gtype)
        if gtype==1:
            "rectangle"
            return self.get_rectangle_ring_at_a_vertex(ik,time,step,exact,heat,loop,upsample,show)

        radius = time * self.mean_edge_length()
        tm_v,tm_f,vs,ring,faces = self.get_gobal_triangulated_rr_quad_mesh(ik)
        _,_,geo_dist = self.get_geodesic_disk(vs,tm_v,tm_f,exact,heat,loop,upsample,frequency=False)
               
        if show:
            ring4,faces4 = self.get_geodesic_disk_blur_quads(ik,ring,geo_dist,radius)
            trv,trf,trv_geo_dist,_ = self.get_local_geodesic_disk(ik,ring4,faces4,exact,heat,loop,upsample,frequency=False)
            poly = self._get_sub_geodesic_disk_polyline(trv,trf,trv_geo_dist,radius)
            
        if gtype==2:
            "circle_inside"
            ring,faces = self.get_geodesic_disk_inside_quads(ring,geo_dist,radius)
        
        elif gtype==3:
            "circle_blur"
            ring,faces = self.get_geodesic_disk_blur_quads(ik,ring,geo_dist,radius)
           
        elif gtype==4:
            "circle_blur_big"
            ring,faces = self.get_geodesic_disk_blur_quads(ik,ring,geo_dist,radius*1.5)
        
        trv,trf,trv_geo_dist,geo_dist = self.get_local_geodesic_disk(ik,ring,faces,exact,heat,loop,upsample,frequency=False)

        if show:
            trv,trf,geo_dist,_ = self.get_local_geodesic_disk(ik,ring,faces,exact,heat,loop,upsample,frequency=True)
            tri = Mesh()
            tri.make_mesh(trv,trf)   
            return tri,geo_dist,poly    

        "set weights for ring-vertices, lamda=1 if inside, else=cos()"
        arr = np.zeros(self.V)#np.zeros(len(ring))
        for i in range(len(ring)):
            "geo_dist is one-to-one map to ring_vertex_index"
            if geo_dist[i]<=radius:
                arr[ring[i]] = 1
            else:
                k = (geo_dist[i]-radius) / (np.max(geo_dist)-radius)
                arr[ring[i]] = np.cos(np.pi / step * k) # here a=2,3,4,5,6...
        return ring,faces,geo_dist,arr  
    
    def get_diagonal_polyline_from_2points(self,vi,is_poly=True):
        if len(vi)==1:
            print('You should select at least two diagonal vertices!')
        else:
            H = self.halfedges
            vl = vr = np.array([],dtype=int)
            es1 = np.where([H[:,0]==vi[0]][0])[0]
            es2 = np.where([H[:,0]==vi[1]][0])[0]
            f1 = H[es1,1]
            f2 = H[es2,1]
            f = np.intersect1d(f1,f2)
            il = es1[np.where(f1==f)[0]]
            ir = es2[np.where(f2==f)[0]]

            vl = np.r_[vl,H[il,0]]
            num = len(np.where(H[:,0]==H[il,0])[0])
            while num==4 and H[H[H[H[il,4],2],4],1]!=-1 and H[H[il,4],1]!=-1:
                il = H[H[H[H[il,4],2],4],3]
                if H[il,0] in vl:
                    break
                vl = np.r_[vl, H[il,0]]
                num = len(np.where(H[:,0]==H[il,0])[0])
            
            vr = np.r_[vr,H[ir,0]]
            num = len(np.where(H[:,0]==H[ir,0])[0])
            while num==4 and H[H[H[H[ir,4],2],4],1]!=-1 and H[H[ir,4],1]!=-1:
                ir =H[H[H[H[ir,4],2],4],3]
                if  H[ir,0] in vr:
                    break
                vr = np.r_[vr, H[ir,0]] 
                num = len(np.where(H[:,0]==H[ir,0])[0])
            "Vl = self.vertices[vl[::-1]]; Vr = self.vertices[vr]"
            iv = np.r_[vl[::-1],vr]
            VV = self.vertices[iv]
            if is_poly:
                poly = self.make_polyline_from_endpoints(VV[:-1,:],VV[1:,:])
                return iv,VV,poly
            return iv,VV
    
    def get_strip_along_polyline(self,vis,diag=True,half=True,tangent_ruling=True):
        if diag:
            if tangent_ruling:
                "Cnet-developable case, c1 ruling= c2 tangent"
                iv,_ = self.get_diagonal_polyline_from_2points(vis,is_poly=False)
                v,va,vb,vc,vd = self.rr_star_corner
                if vis[0] in v:
                    i = np.where(v==vis[0])[0]
                    if vis[1] == va[i]:
                        v1, v3 = vb, vd
                    elif vis[1] == vb[i]:
                        v1, v3 = va, vc
                    elif vis[1] == vc[i]:
                        v1, v3 = vb, vd
                    elif vis[1] == vd[i]:
                        v1, v3 = va, vc
                iv0,ivi,ivj = [],[],[]
                for i in iv:
                    if i in v:
                        j = np.where(v==i)[0][0]
                        iv0.append(i)
                        ivi.append(v1[j])
                        ivj.append(v3[j])
                v0 = np.array(iv0,dtype=int)    
                vi,vj = np.array(ivi,dtype=int),np.array(ivj,dtype=int) 
                V = self.vertices
                li = np.linalg.norm(V[vi]-V[v0],axis=1)
                lj = np.linalg.norm(V[vj]-V[v0],axis=1)
                t1 = (V[vi]-V[v0])*(lj**2)[:,None]-(V[vj]-V[v0])*(li**2)[:,None]
                t1 = t1 / np.linalg.norm(t1,axis=1)[:,None]
                k = (li+lj)/2/2
                if half:
                    "polyline is one boundary of the strip"
                    Vs1,Vs2 = V[v0]+t1*k[:,None],V[v0]
                    print(Vs1)
                else:
                    "polyline is the central rail of the entire strip"
                    Vs1,Vs2 = V[v0]+t1*k[:,None],V[v0]-t1*k[:,None]
                arr = np.arange(len(k)-1)
                stripV = np.vstack((Vs1,Vs2))
                sfv1,sfv2 = arr, arr+1
                sfv4,sfv3 = len(k)+sfv1,len(k)+sfv2
                stripF = np.c_[sfv1,sfv2,sfv3,sfv4].tolist()
                sm = Mesh()
                sm.make_mesh(stripV,stripF)
                return sm
        else:
            "support structure case"

    def get_a_boundary_L_strip(self,direction):
        "AG-net: only rectangular-patch shape"
        H = self.halfedges
        v,va,vb,vc,vd = self.rr_star_corner
        corner = self.corner
        vii = vb if direction else va
        i = np.intersect1d(vii,corner)[0]

        def _get_v13(i):
            "should have at least one"
            e = np.intersect1d(np.where(H[:,1]==-1)[0],np.where(H[:,0]==i)[0])[0]
            e0 = e
            ib1,ib2 = [],[]
            while H[H[e,2],0] not in corner:
                ib1.append(e)
                e = H[e,2]
            ib1.append(e)
            
            #---------
            # e = H[e,2]
            # while H[H[e,2],0] not in corner:
            #     e = H[e,2]
            #     ib2.append(e)  
                
            # v1,v3 = H[ib1,0], H[H[H[ib1,4],3],0]
            # v1 = np.r_[v1,H[H[H[H[ib2,4],3],3],0]]
            # v3 = np.r_[v3,H[H[ib2,4],0]]

            #----------------
            ib1 = ib1[::-1]
            v1,v3 = H[ib1,0], H[H[H[ib1,4],3],0]
            while H[H[e0,3],0] not in corner:
                e0 = H[e0,3]
                ib2.append(e0)
            ib2.append(H[e0,3])
            
            v1 = np.r_[v1,H[H[ib2[1:],4],0]]
            v3 = np.r_[v3,H[H[H[H[ib2[1:],4],3],3],0]]  
            return np.c_[v1,v3]
        "return L-shape boundary quads (Lx1): v and only 1diag v"
        return _get_v13(i)
    
    def get_cylinder_annulus_mesh_diagonal_oriented_vertices(self,direction):
        "AG-net: only work for cylinder-annulus-shape"
        "return loop boundary quads (Lx1): v and only 1diag v"
        H = self.halfedges
        vb,_ = self.get_i_boundary_vertex_indices(0)
        vnext = []
        for v in vb:
            if direction:
                e = np.intersect1d(np.where(H[:,1]==-1)[0],np.where(H[:,0]==v)[0])[0]
                vdiag = H[H[H[e,4],3],0]
            else:
                e = np.intersect1d(np.where(H[:,1]==-1)[0],np.where(H[:,0]==v)[0])[0]
                e = H[e,3]
                vdiag = H[H[H[H[e,4],2],2],0]
            vnext.append(vdiag)
        return np.c_[vb, vnext]
        
    def get_diagonal_vertex_list(self,interval,another_direction=True):
        "AG-net: work for rectangular-patch, not cylinder-annulus"
        v,_,_,_,_ = self.rr_star_corner
        vf = self.get_a_boundary_L_strip(another_direction)
        
        alls_v0, alls_vs_v0 = [],[]
        select_v0,select_vs_v0 = [],[]
        for k in range(len(vf)):
            vis = vf[k]
            iv,_ = self.get_diagonal_polyline_from_2points(vis,is_poly=False)
            iv0,jv0 = [],[]
            for i in iv:
                if i in v:
                    j = np.where(v==i)[0][0]
                    iv0.append(i)
                    jv0.append(j)  
            if len(jv0)>1:#!=0:
                alls_v0.append(iv0)
                alls_vs_v0.append(jv0)
                if k>0 and k<len(vf)-1 and k % interval ==0:
                    select_v0.append(iv0)
                    select_vs_v0.append(jv0)
        return alls_v0,alls_vs_v0,select_v0,select_vs_v0

    ## using this first for choosing mesh polys, with full boundary inds.
    def get_both_isopolyline(self,diagpoly=False,is_one_or_another=False,
                             is_poly=False,only_inner=False,interval=1):
        "AG-net: work for rectangular-patch + cylinder-annulus"
        if only_inner:
            v,_,_,_,_ = self.rr_star_corner
        ipllist = []
        vl = vr = np.array([],dtype=int)
        if diagpoly:
            if len(self.corner)!=0:
                "surface shape: rectangular-patch"
                vf1 = self.get_a_boundary_L_strip(direction=True)
                vf2 = self.get_a_boundary_L_strip(direction=False)
            else:
                "surface shape: cylinder-annulus"
                vf1 = self.get_cylinder_annulus_mesh_diagonal_oriented_vertices(True)
                vf2 = self.get_cylinder_annulus_mesh_diagonal_oriented_vertices(False)
            vf = vf1 if is_one_or_another else vf2

            for i, k in enumerate(vf):
                if i%interval==0:
                    iv,_ = self.get_diagonal_polyline_from_2points(k,is_poly=False)   
                    if len(iv)!=0:
                       ipllist.append(iv)
                       vl = np.r_[vl,iv[:-1]]
                       vr = np.r_[vr,iv[1:]]
            # else:
            #     for k in vf1:
            #         iv,_ = self.get_diagonal_polyline_from_2points(k,is_poly=False)
            #         if len(iv)!=0:
            #             ipllist.append(iv)
            #             vl = np.r_[vl,iv[:-1]]
            #             vr = np.r_[vr,iv[1:]]
            #     for k in vf2:
            #         iv,_ = self.get_diagonal_polyline_from_2points(k,is_poly=False)
            #         if len(iv)!=0:
            #             ipllist.append(iv)
            #             vl = np.r_[vl,iv[:-1]]
            #             vr = np.r_[vr,iv[1:]]            
        else: 
            "more general, one boundary v"
            ### below need to check for different direction, and patch/annulus
            if len(self.corner)!=0:
                "patch"
                vb1,_ = self.get_i_boundary_vertex_indices(0) # i=0,1,2,3
                try:
                    vb2,_ = self.get_i_boundary_vertex_indices(1)# i=0,1,2,3 # to check if 1 or 2#TODO
                except:
                    vb2,_ = self.get_i_boundary_vertex_indices(1)
                vb = vb1 if is_one_or_another else vb2
                allplv = self.get_isoline_between_2bdry(vb)   
                #ipllist = allplv
                
                if only_inner:
                    if allplv[0][0] in self.corner:
                        allplv.pop(0)
                    if allplv[-1][0] in self.corner:
                        allplv.pop()
            else:
                "annulus"
                M = self.rot_patch_matrix if is_one_or_another else self.rot_patch_matrix.T
                allplv = M.tolist() 
                #ipllist = allplv
                ##print(allplv,M.shape)
                
            #for iv in allplv:
            for i, iv in enumerate(allplv):
                if i%interval==0:
                    ipllist.append(iv)
                    if len(iv)!=0:
                        vl = np.r_[vl,iv[:-1]]
                        vr = np.r_[vr,iv[1:]]

            # else:
            #     allplv1 = self.get_isoline_between_2bdry(vb1)
            #     allplv2 = self.get_isoline_between_2bdry(vb2)
            #     if only_inner:
            #         if allplv1[0][0] in self.corner:
            #             allplv1.pop(0)
            #         if allplv1[-1][0] in self.corner:
            #             allplv1.pop()
            #     if only_inner:
            #         if allplv2[0][0] in self.corner:
            #             allplv2.pop(0)
            #         if allplv2[-1][0] in self.corner:
            #             allplv2.pop()
            #     ipllist.extend(allplv1,allplv2)
            #     for iv in allplv1:
            #         if len(iv)!=0:
            #             vl = np.r_[vl,iv[:-1]]
            #             vr = np.r_[vr,iv[1:]]
            #     for iv in allplv2:
            #         if len(iv)!=0:
            #             vl = np.r_[vl,iv[:-1]]
            #             vr = np.r_[vr,iv[1:]]
        if is_poly:
            Vl,Vr = self.vertices[vl], self.vertices[vr]
            return self.make_polyline_from_endpoints(Vl,Vr)
        else:    
            return ipllist
       
    def get_strips_along_polylines(self,interval,diag=True,
                                   another_direction=True,big=True):
        "AG-NET: checkerboard way to get strips along geodesic"
        V = self.vertices
        v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        v,va,vb,vc,vd = self.rr_star_corner
        num_list = []
        if diag:
            "checkerboard strips as envelope of tangent planes along geodesic"
            """
               a    1    d
               2    v0   4
               b    3    c  
            """
            _,_,_,isome = self.get_diagonal_vertex_list(interval,another_direction)
            
            if big:
                if False:    
                    for ii in isome:
                        num_list.append(len(ii)+1)
                    "work well for showing strips, but problem for unroll"
                    import itertools
                    idd = list(itertools.chain(*isome)) # flatten 2-d list into 1d
                    v,v1,v2,v3,v4=v[idd],v1[idd],v2[idd],v3[idd],v4[idd]
                    arr = np.unique(np.r_[v1,v2,v3,v4]) 
                    i1,i2,i3,i4=[],[],[],[]
                    for i in range(len(v)):
                        i1.append(np.where(arr==v1[i])[0][0])
                        i2.append(np.where(arr==v2[i])[0][0])
                        i3.append(np.where(arr==v3[i])[0][0])
                        i4.append(np.where(arr==v4[i])[0][0])
                    Varr = V[arr]
                    flist = np.c_[i1,i2,i3,i4]
                    sm = Mesh()
                    sm.make_mesh(Varr,flist)                 
                else:
                    "compatible for unroll strips"
                    allV, allF = np.zeros(3), np.zeros(4,dtype=int)
                    mmm = 0
                    for ipoly in isome:
                        ipl = np.array(ipoly)
                        num = len(ipl)+1
                        va,vb,vc,vd=v1[ipl],v2[ipl],v3[ipl],v4[ipl]
                        if va[1]==vd[0]:
                            vup = np.r_[va,vd[-1]]
                            vdw = np.r_[vb,vc[-1]]
                        elif vd[1]==vc[0]:
                            vup = np.r_[vd,vc[-1]]
                            vdw = np.r_[va,vb[-1]]
                        elif vc[1]==vb[0]:
                            vup = np.r_[vc,vb[-1]]
                            vdw = np.r_[vd,va[-1]]
                        elif vb[1]==va[0]:
                            vup = np.r_[vb,va[-1]]
                            vdw = np.r_[vc,vd[-1]]
                        Vpl = np.vstack((V[vup],V[vdw]))

                        arr1 = mmm + np.arange(num)
                        arr2 = num + arr1
                        flist = np.c_[arr1[:-1],arr2[:-1],arr2[1:],arr1[1:]]
                        allV, allF = np.vstack((allV,Vpl)), np.vstack((allF,flist))
                        mmm = len(allV)-1
                        num_list.append(num)
                    
                        sm = Mesh()
                        sm.make_mesh(allV[1:],allF[1:])            
                return sm,num_list   
            else:
                "central checkerboard strips: va if direction else vb"
                V0 = V[v]
                if not another_direction:
                    "polyline: a-v0-c"
                    Va,Vb,Vc,Vd = V[va],V[vb],V[vc],V[vd]
                    V1,V2,V3,V4 = V[v1],V[v2],V[v3],V[v4]
                else:
                    "polyline: b-v0-d"
                    Va,Vb,Vc,Vd = V[vb],V[vc],V[vd],V[va]
                    V1,V2,V3,V4 = V[v2],V[v3],V[v4],V[v1]

                allV, allF = np.zeros(3), np.zeros(4,dtype=int)
                mmm = 0
                for ipoly in isome:
                    ipl = np.array(ipoly)
                    num = 2*len(ipl)+2
                    Vup = np.zeros((num,3))
                    Vdw = np.zeros((num,3))
                    Vup[0] = (Va[ipl][0]+V1[ipl][0])/2
                    Vup[:-1][1::2] = (V1[ipl]+V0[ipl])/2
                    Vup[:-1][2::2] = (V4[ipl]+V0[ipl])/2
                    Vup[-1] = (V4[ipl][-1]+Vc[ipl][-1])/2
                    
                    Vdw[0] = (Va[ipl][0]+V2[ipl][0])/2
                    Vdw[:-1][1::2] = (V2[ipl]+V0[ipl])/2
                    Vdw[:-1][2::2] = (V3[ipl]+V0[ipl])/2
                    Vdw[-1] = (V3[ipl][-1]+Vc[ipl][-1])/2
                    Vpl = np.vstack((Vup,Vdw))
                    
                    arr1 = mmm + np.arange(num)
                    arr2 = num + arr1
                    flist = np.c_[arr1[:-1],arr2[:-1],arr2[1:],arr1[1:]]
                    
                    allV, allF = np.vstack((allV,Vpl)), np.vstack((allF,flist))
                    mmm = len(allV)-1
                    num_list.append(num)
                
                sm = Mesh()
                sm.make_mesh(allV[1:],allF[1:])            
            return sm,num_list
        else:
            "patch-shape"
            M = self.patch_matrix
            if another_direction:
                M = M.T   
            num = M.shape[0]
            Ml, Mr = M[:,::interval],M[:,1::interval]
            a = min(Ml.shape[1],Mr.shape[1])
            
            allV, allF = np.zeros(3), np.zeros(4,dtype=int)
            mmm = 0
            for i in range(a):
                Vl,Vr = V[Ml[:,i]], V[Mr[:,i]]
                Vpl = np.vstack((Vl, Vr))
                arr1 = mmm + np.arange(num)
                arr2 = num + arr1
                flist = np.c_[arr1[:-1],arr2[:-1],arr2[1:],arr1[1:]]
                allV, allF = np.vstack((allV,Vpl)), np.vstack((allF,flist))
                mmm = len(allV)-1
                num_list.append(num)
                sm = Mesh()
                sm.make_mesh(allV[1:],allF[1:])            
            return sm,num_list  

    def get_isoline_between_2bdry(self,vb):
        "starting from 1 boundary-vertex, get isoline untile to opposit bdry"
        "work for rectangular-patch + cylinder-annulus"
        H = self.halfedges
        if True:
            "reverse the boundary list"
            vi,vj = vb[0],vb[1]
            ei = np.intersect1d(np.where(H[:,0]==vi)[0], np.where(H[H[:,4],0]==vj)[0])[0]
            if H[ei,1]!=-1:
                vb = vb[::-1]
        allv = []
        for i in range(len(vb)-1):
            vi,vj = vb[i],vb[i+1]
            ei = np.intersect1d(np.where(H[:,0]==vi)[0], np.where(H[H[:,4],0]==vj)[0])[0]
            plv = [vi]
            e = ei
            while H[H[e,4],1]!=-1:
                e = H[H[H[e,4],2],2]
                plv.append(H[e,0])
            allv.append(plv)  
        "last boundary"    
        plvj = [vj]    
        e = H[ei,4]
        while H[e,1]!=-1:
            e = H[H[H[e,2],2],4]
            plvj.append(H[e,0])
        allv.append(plvj)   
        return allv
    
    def get_isoline_vertex_list(self,interval,another_direction=True):
        "along one bdry for a patch-shape; two bdry for a star-shape"
        v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        if another_direction:
            vl,vr = v2,v4
        else:
            vl,vr = v1,v3
        if False:
            "patch-shape"
            M = self.patch_matrix
            if another_direction:
                M = M.T
            vb = M[-1,1:-1]
        else:
            "more general, one boundary v"
            vb1,_ = self.get_i_boundary_vertex_indices(0) # i=0,1,2,3
            vb2,_ = self.get_i_boundary_vertex_indices(1) # i=0,1,2,3
            if len(np.intersect1d(np.r_[vl,vr],vb1))!=0:
                vb = vb1
            elif len(np.intersect1d(np.r_[vl,vr],vb2))!=0:
                vb = vb2
            else:
                "need to check and rewrite"
                "patch-shape"
                M = self.patch_matrix
                
                if another_direction:
                    M = M.T
                vb = M[-1,:]
            #vb = vb[1:-1]##[::interval] # Hui comment, may have problem later.
            if len(np.intersect1d(vr,vb))!=0:
                "make sure vl in vb, vr not in vb"
                vl,vr = vr,vl    
                
        alls_v0, alls_vs_v0 = [],[]
        select_v0,select_vs_v0 = [],[]
        allplv = self.get_isoline_between_2bdry(vb)
        for k in range(len(vb)):
            iv = allplv[k]
            iv0,jv0 = [],[]
            for i in iv:
                if i in v:
                    j = np.where(v==i)[0][0]
                    iv0.append(i)
                    jv0.append(j)  
            if len(jv0)>1:#!=0:
                alls_v0.append(iv0)
                alls_vs_v0.append(jv0)
                if k % interval ==0:
                    select_v0.append(iv0)
                    select_vs_v0.append(jv0)
        return alls_v0,alls_vs_v0,select_v0,select_vs_v0  
            
    def get_normal_strips_along_polylines(self,interval,diag=True,
                                          another_direction=True,s=1):
        "AG-NET: get osculating planes along geodesic passing vertex normals"
        V = self.vertices
        #s = np.mean(self.edge_lengths())
        allN = np.zeros((self.V,3))

        if diag:
            """
            checkerboard strips as envelope of tangent planes along geodesic
            in diagonal direction:
               a    1    d
               2    v0   4
               b    3    c  
            """
            _,_,isome,_ = self.get_diagonal_vertex_list(interval,another_direction)
            v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        else:
            """
            checkerboard strips as envelope of tangent planes along geodesic
            in mesh isoline directions: (1-v0-3 or 2-v0-4)
               a    1    d
               2    v0   4
               b    3    c  
            """   
            _,_,isome,_ = self.get_isoline_vertex_list(interval,another_direction)
            v,v1,v2,v3,v4 = self.rr_star_corner
            
        N = np.cross(V[v1]-V[v],V[v2]-V[v])  
        N = N / np.linalg.norm(N,axis=1)[:,None]
        allN[v] = N
        dot = np.einsum('ij,ij->i',self.vertex_normals()[v],N)
        idd = np.where(dot<0)[0]
        allN[v[idd]] = -N[idd]        
        
        allV, allF = np.zeros(3), np.zeros(4,dtype=int)
        k = 0
        mmm = 0
        num_list = []
        for ipl in isome:
            i = np.array(ipl)
            if len(i)>1:
                irr = i
                #s = np.mean(np.linalg.norm(V[irr[1:]]-V[irr[:-1]],axis=1))
                Vpl = np.vstack((V[irr], V[irr]  +  s*allN[irr])) # need to choose + -
                arr1 = mmm + np.arange(len(irr))
                arr2 = len(irr) + arr1
                
                flist = np.c_[arr1[:-1],arr2[:-1],arr2[1:],arr1[1:]]
                allV, allF = np.vstack((allV,Vpl)), np.vstack((allF,flist))
                num_list.append(len(irr))
            k += 1
            mmm = len(allV)-1 
        sm = Mesh()
        sm.make_mesh(allV[1:],allF[1:])    
        return sm,num_list   



    ### geting strips:--------------------------
    def get_strip_from_rulings(self,an,ruling,row_list,is_smooth):
        "AG-NET: rectifying(tangent) planes along asymptotic(geodesic) crv."
        if is_smooth:
            from smooth import fair_vertices_or_vectors
        allV, allF = np.zeros(3), np.zeros(4,dtype=int)
        mmm = 0
        numv = 0
        for num in row_list:
            srr = mmm + np.arange(num)
            Pup = an[srr]+ruling[srr]
            if is_smooth:
                Pup = fair_vertices_or_vectors(Pup,itera=50,efair=is_smooth)
            P1234 = np.vstack((an[srr], Pup))
            arr1 = numv + np.arange(num)
            arr2 = arr1 + num
            flist = np.c_[arr1[:-1],arr1[1:],arr2[1:],arr2[:-1]]
            allV, allF = np.vstack((allV,P1234)), np.vstack((allF,flist))
            numv = len(allV)-1
            mmm += num
        sm = Mesh()
        ##print(allV,allF)
        sm.make_mesh(allV[1:],allF[1:])    
        return sm

    def get_strip_along_asym_or_geo2(self,v=None,an=None,vN=None,
                                    dist=1,interval=2,
                                    diagpoly=True,another_direction=True):
        "AG-NET: rectifying(tangent) planes along asymptotic(geodesic) crv."
        if diagpoly:
            """
            checkerboard strips as envelope of tangent planes along crv.
            in diagonal direction:
               a    1    d
               2    v0   4
               b    3    c  
            """
            _,_,isome,_ = self.get_diagonal_vertex_list(interval,another_direction)
        else:
            """
            checkerboard strips as envelope of tangent planes along crv.
            in mesh isoline directions: (1-v0-3 or 2-v0-4)
               a    1    d
               2    v0   4
               b    3    c  
            """   
            _,_,isome,_ = self.get_isoline_vertex_list(interval,another_direction)
        
        V,N = self.vertices, self.vertex_normals() * dist
        if v is not None:
            V[v], N[v] = an, vN
        
        allV, allF = np.zeros(3), np.zeros(4,dtype=int)
        mmm = 0
        num_list = []
        for ipl in isome:
            irr = np.array(ipl)
            if len(irr)>1:
                sub = []
                for j in irr:
                    if j in v:
                        sub.append(j)
                if len(sub)!=0:
                    srr = np.array(sub)
                    Vpl = np.vstack((V[srr], V[srr]  + N[srr]))
                    arr1 = mmm + np.arange(len(srr))
                    arr2 = len(srr) + arr1
                    
                    flist = np.c_[arr1[:-1],arr2[:-1],arr2[1:],arr1[1:]]
                    allV, allF = np.vstack((allV,Vpl)), np.vstack((allF,flist))
                    num_list.append(len(srr))
            mmm = len(allV)-1 
        sm = Mesh()
        sm.make_mesh(allV[1:],allF[1:])    
        return sm,num_list   
    
    def get_strip_along_asym_or_geo(self,v=None,an=None,vN=None,
                                    dist=1,interval=1,
                                    diagpoly=True,another_direction=True):
        "AG-NET: rectifying(tangent) planes along asymptotic(geodesic) crv."
        if diagpoly:
            """
            checkerboard strips as envelope of tangent planes along crv.
            in diagonal direction:
               a    1    d
               2    v0   4
               b    3    c  
            """
            _,_,isome,_ = self.get_diagonal_vertex_list(interval,another_direction)
        else:
            """
            checkerboard strips as envelope of tangent planes along crv.
            in mesh isoline directions: (1-v0-3 or 2-v0-4)
               a    1    d
               2    v0   4
               b    3    c  
            """   
            _,_,isome,_ = self.get_isoline_vertex_list(interval,another_direction)
        
        V,N = self.vertices, self.vertex_normals() * dist
        if v is not None:
            V[v], N[v] = an, vN
        
        allV, allF = np.zeros(3), np.zeros(4,dtype=int)
        mmm = 0
        num_list = []
        for ipl in isome:
            i = np.array(ipl)
            if len(i)>1:
                irr = i
                Vpl = np.vstack((V[irr], V[irr]  + N[irr]))
                arr1 = mmm + np.arange(len(irr))
                arr2 = len(irr) + arr1
                
                flist = np.c_[arr1[:-1],arr2[:-1],arr2[1:],arr1[1:]]
                allV, allF = np.vstack((allV,Vpl)), np.vstack((allF,flist))
                num_list.append(len(irr))
            mmm = len(allV)-1 
        sm = Mesh()
        sm.make_mesh(allV[1:],allF[1:])    
        return sm,num_list       
   ###--------------------------------------------------------


     
    def get_a_cross_strip(self,e):
        "AG-NET: boundary two cross strips"
        if e is None:
            "patch matrix shape"
            M = self.patch_matrix
            iv1 = M[-1,:]
            iv2 = M[-2,:]
            iv3 = M[:,0][::-1][2:]
            iv4 = M[:,1][::-1][2:]
            strip_v = np.r_[iv1,iv2,iv3,iv4]
            num1,num2 = len(iv1),len(iv3)
            arr1,arr2 = np.arange(num1), num1+np.arange(num1)
            arr3,arr4 = 2*num1+np.arange(num2),2*num1+num2+np.arange(num2)
            v1 = np.r_[arr1[:-1],arr2[0],arr3[:-1]]
            v2 = np.r_[arr1[1:],arr2[1],arr4[:-1]]
            v3 = np.r_[arr2[1:],arr4[0],arr4[1:]]
            v4 = np.r_[arr2[:-1],arr3[0],arr3[1:]]   
            stripV = self.vertices[strip_v]
            stripF = np.c_[v1,v2,v3,v4].tolist()
        sm = Mesh()
        sm.make_mesh(stripV,stripF)
        ia = np.array([M[-2,1],M[-2,2],M[-3,1],M[-1,0]])
        ib = np.array([M[-2,2],M[-2,1],M[-2,3],M[-1,2]])
        ic = np.array([M[-3,1],M[-4,1],M[-2,1],M[-3,0]])
        return sm,strip_v,[arr1,arr2,arr3,arr4],[ia,ib,ic]
    
    def get_cross_strip_propagation(self,e):
        "AG-NET:starting with L-shape strip: get_a_cross_strip; adding new row points"
        allvn = self.vertex_normals()
        M = self.patch_matrix
        
        def _newP(Virowa,Virowd,Vc,Vcl,Nirowb,Nirowc):
            """
             --
            |  |
           cl--c  P 
            |  |
             --a--b-- -- -- --
            |  |  |  |  |  |  |
             -- --d-- -- -- --
            irowa:= a,b,...-0
            irowd:= d,...-0
            len(V[ic])=len(V[icl])==1
            return: len(P)=len(Virowd), [c,P] as new Virowa
            """
            def _point(Va,Vb,Vc,Vcl,Vd,Nib,Nic):
                Nb = np.cross(Va-Vb,Vd-Vb)
                Nb = Nb / np.linalg.norm(Nb) 
                if np.dot(Nib,Nb)<0: # should be positive
                    Nb = -Nb
                Nc = np.cross(Vcl-Vc,Va-Vc)
                Nc = Nc / np.linalg.norm(Nc) 
                if np.dot(Nic,Nc)<0: # should be positive
                    Nc = -Nc
                Ng = np.cross(Nb,Nc)
                Ng = Ng / np.linalg.norm(Ng)
                if np.dot(Ng,(Vc-Va))<0:
                    Ng = -Ng
                k = np.linalg.norm(Vc-Vb)
                Pi = Va + Ng * k
                return Vb,Vc,Pi
            
            Va = Virowa[0] #len>1
            Pi = np.array([0,0,0])
            for i in range(len(Virowd)):
                Nib,Nic = Nirowb[i], Nirowc[i]
                Vb,Vd = Virowa[i+1],Virowd[i]
                Va,Vcl,Vc = _point(Va,Vb,Vc,Vcl,Vd,Nib,Nic)
                Pi = np.vstack((Pi,Vc))
            P = Pi[1:,:]
            return P
        
        csm,strip_v,arr,_ = self.get_a_cross_strip(e)
        iv1,iv2,iv3,iv4 = arr
        num_row,num_col = len(iv3),len(iv1)-2 #len(iv3)=len(iv4);len(iv1)=len(iv2)
        nmV,nmF = csm.vertices,np.array(csm.faces_list()) # all quad faces
        Virowa,Virowd = nmV[iv2[1:]], nmV[iv1[2:]]
        newP = np.array([0,0,0])
        for i in range(1):#num_row):
            irowb = M[i+1,2:]
            irowc = M[i+2,1:-1]
            Nb,Nc = allvn[irowb], allvn[irowc]
            Vc,Vcl = nmV[iv4[i]],nmV[iv3[i]]
            P = _newP(Virowa,Virowd,Vc,Vcl,Nb,Nc)
            
            newP = np.vstack((newP,P))
            Virowd = Virowa[1:,:]
            Virowa = np.vstack((Vc,P))
            if i==0:
                ia,ib = iv2[1:-1],iv2[2:]
                ie,ic=0,0 # no use, just for below
            else:
                ia,ib = ie,ic
            ic = len(nmV) + np.arange(num_col)
            ie = np.r_[iv4[i], ic[:-1]]  
            nmV = np.vstack((nmV,P))
            nmF = np.vstack((nmF,np.c_[ia,ib,ic,ie]))
            
        sm = Mesh()
        sm.make_mesh(nmV,nmF)
        return sm,newP[1:,:]
 
    def get_a_closed_isostrip(self,e=None,nextloop=False,reverse=False):
        "AG-net: Case: for diagnet=Anet, one isoline is geodesic "
        M = self.rot_patch_matrix
        
        if e is not None:
            H = self.halfedges
            i1,i2 = np.where(H[:,5]==e)[0] # should be two
            v1,v2 = H[i1,0],H[i2,0]
            r1,r2 = np.where(M==v1)[0][0],np.where(M==v2)[0][0]
        elif nextloop:    
            row,col = M.shape[0],M.shape[1]-1
            if reverse:
                r1,r2 = 1,0
                strip_v = M[::-1,:-1].flatten()
            else:
                r1,r2 = row-2,row-1
                strip_v = M[:,:-1].flatten()
            stripV = self.vertices[strip_v]
            m = np.arange(row*col).reshape(row,col)
            v1,v4 = m[:-1,:], np.insert(m[:-1,1:],col-1,m[:-1,0],axis=1)
            v2,v3 = m[1:,:], np.insert(m[1:,1:],col-1,m[1:,0],axis=1)               
            stripF = np.c_[v1.flatten(),v2.flatten(),v3.flatten(),v4.flatten()]   
            sm = Mesh()
            sm.make_mesh(stripV,stripF)
            Bi,Pi = self.vertices[M[r1,:-1]], self.vertices[M[r2,:-1]]
            return sm,Bi,Pi    
        else:
            "rotational patch matrix shape"
            r1,r2 = 0,1
            strip_v = np.r_[M[r1,:-1],M[r2,:-1]] #last column==first column
            stripV = self.vertices[strip_v]
            num = M.shape[1]-1
            m = np.arange(2*num).reshape(2,-1)
            v1,v4 = m[0,:], np.r_[m[0,1:],m[0,0]]
            v2,v3 = m[1,:], np.r_[m[1,1:],m[1,0]]                
            stripF = np.c_[v1,v2,v3,v4].tolist()    
            sm = Mesh()
            sm.make_mesh(stripV,stripF)

            M1 = M[r1:r2+1,:-1] # corresponding to strip_v
            M2 = np.insert(M1,0,M1[:,-1],axis=1)
            M2 = np.insert(M2,0,M1[:,-2],axis=1)
            M2 = np.insert(M2,num+2,M1[:,0],axis=1) # Note inserte -1=last-two column
            M2 = np.insert(M2,num+3,M1[:,1],axis=1)
            io = [M2[1,1:-1],M2[0,:-2],M2[0,2:],M2[0,1:-1]]#[center,left,right,next]
            return sm,strip_v,io        

    def get_1closed_strip_propagation(self,row):
        "AG-NET:starting with 1-loop strip: adding new row points"
        from compas.geometry import Plane,intersection_plane_plane
        # allvn = self.vertex_normals()
        # M = self.rot_patch_matrix
        if False:
            "only 1 boundary strip"
            sm,_,_ = self.get_a_closed_isostrip()
            Vp = sm.vertices
            num = int(sm.V / 2)
            Bi,Pi = Vp[:num], Vp[num:]
            num_row = row+2
        else:
            sm,Bi,Pi = self.get_a_closed_isostrip(nextloop=True,reverse=True) # need to choose nextloop+reverse
            Vp = sm.vertices
            num = Bi.shape[0] # column_num
            num_row = int(sm.V / num) + row

        vlist = Vp
        flist = np.array(sm.faces_list())
        colinear = np.array([])
        xx1 = xx2 = np.array([])
        ex_an=ex_P2=ex_tno=ex_on = np.array([0,0,0])
        for i in range(row):
            arr = np.r_[num-2,num-1,np.arange(num),0,1]
            an,anoo = Pi[arr[1:-1]],Bi[arr[1:-1]]

            "normals of tangent planes"
            tN = np.cross(Bi[arr[2:]]-an,Bi[arr[:-2]]-an)
            tN = tN / np.linalg.norm(tN,axis=1)[:,None]

            # dot = np.einsum('ij,ij->i',allvn[M[row,arr[1:-1]]],tN)
            # idd = np.where(dot<0)[0]
            # tN[idd] = -tN[idd]
            
            an,anl,anr,anoo = an[1:-1,:],an[:-2,:],an[2:,:],anoo[1:-1,:]
            tNo,tNl,tNr = tN[1:-1,:],tN[:-2,:],tN[2:,:]

            "normals of osculating planes"
            oN = np.cross(tNo, anoo-an)
            oN = oN / np.linalg.norm(oN,axis=1)[:,None]        
        
            P1,P2 = np.array([]),np.array([])
            for i in range(len(an)):
                plt1, plt2 = Plane(anl[i],tNl[i]), Plane(anr[i],tNr[i])
                l = intersection_plane_plane(plt1,plt2)
                P1,P2 = np.r_[P1,l[0]],np.r_[P2,l[1]]
            P1,P2 = P1.reshape(-1,3),P2.reshape(-1,3)
            
            "if an in line=intersec(T1,T2),then nexP should be as red as possible."
            ck = np.linalg.norm(np.cross(anoo-P1,P2-P1),axis=1) # if anoo / an
            colinear = np.r_[colinear,ck]
            #print('%.2g' %np.mean(ck),'%.2g' %np.max(ck))
            
            vec = (P2-P1)/np.linalg.norm(P1-P2,axis=1)[:,None]
            lamda = np.linalg.norm(anl-anr,axis=1) * 0.25 
            lamda += np.linalg.norm(an-anoo,axis=1)*0.5
            ind = np.where(np.einsum('ij,ij->i',vec,an-anoo)<0)[0]
            lamda[ind] = -lamda[ind]
            if True:
                P1,P2 = an, an + vec * lamda[:,None]  # set new Pi as P2
            else:
                P1,P2 = an, anoo + vec * lamda[:,None]*2  # set new Pi as P2
            
            # Pi = np.array([])
            # for i in range(len(an)):
            #     a1 = [an[i],oN[i]]
            #     b1 = [anl[i],tNl[i]]
            #     c1 = [anr[i],tNr[i]]
            #     P = intersect_plane_plane_plane(a1,b1,c1)
            #     Pi = np.r_[Pi,P]
            # Pi = Pi.reshape(-1,3) 
            # self.nextP = P2#Pi # suose colinear is good enough, directly choose P2.
    
            Bi,Pi =  P1,P2
            vlist = np.vstack((vlist,Pi))
            ex_an = np.vstack((ex_an,an)) # =P1
            ex_P2 = np.vstack((ex_P2,P2))
            ex_tno = np.vstack((ex_tno,tNo))
            ex_on = np.vstack((ex_on,oN))     
            xx1 = np.r_[xx1,np.linalg.norm(anl-anr,axis=1)* 0.25]
            xx2 = np.r_[xx2,np.linalg.norm(anoo-an,axis=1) * 1.1]

        m = np.arange((num_row)*num).reshape(num_row,-1) 
        v1,v4 = m[:-1,:], np.insert(m[:-1,1:],num-1,m[:-1,0],axis=1)
        v2,v3 = m[1:,:], np.insert(m[1:,1:],num-1,m[1:,0],axis=1)               
        flist = np.c_[v1.flatten(),v2.flatten(),v3.flatten(),v4.flatten()]           
   
        sm = Mesh()
        sm.make_mesh(vlist,flist.tolist())
        return sm,[ex_an[1:],ex_P2[1:],ex_tno[1:],ex_on[1:]],colinear,[xx1,xx2]


    def get_1strip_from_a_patch(self,e=None):
        "AG-NET: boundary 1 strip"
        if e is None:
            "patch matrix shape"
            M = self.patch_matrix.T
            iv1 = M[0,:]
            iv2 = M[1,:]
            strip_v = np.r_[iv1,iv2]
            num = len(iv1)
            arr = np.arange(num)
            v1 = arr[:-1]
            v2 = arr[1:]
            v3 = v2 + num
            v4 = v1 + num
            stripV = self.vertices[strip_v]
            stripF = np.c_[v1,v2,v3,v4].tolist()
        sm = Mesh()
        sm.make_mesh(stripV,stripF)
        io = [iv2[arr[1:-1]],iv1[arr[:-2]],iv1[arr[2:]],iv1[arr[1:-1]]] #[center,left,right,next]
        return sm,strip_v,io,M

    def get_1_strip_propagation(self,row):
        "AG-NET:starting with 1strip: adding new row points"
        from compas.geometry import Plane,intersection_plane_plane
        allvn = self.vertex_normals()
        def _normal(N0,tN):
            if tN.reshape(-1,3).shape[0]==1:
                tN = tN / np.linalg.norm(tN)
                if np.dot(N0,tN)<0:
                    return -tN
                return tN
            else:
                tN = tN / np.linalg.norm(tN,axis=1)[:,None]  
                dot = np.einsum('ij,ij->i',N0,tN)
                idd = np.where(dot<0)[0]
                tN[idd] = -tN[idd]
                return tN            

        sm,_,io,M = self.get_1strip_from_a_patch()
        Vp = sm.vertices
        num = int(sm.V / 2)
        Bi,Pi = Vp[:num], Vp[num:]
        
        vlist = Vp
        flist = np.array(sm.faces_list())
        colinear = np.array([])
        xx1 = xx2 = np.array([])
        ex_an=ex_P2=ex_tno=ex_on = np.array([0,0,0])
        for i in range(row):
            arr = np.arange(num)

            "normals of tangent planes"
            tN = np.cross(Bi[arr[:-2]]-Pi[1:-1],Bi[arr[2:]]-Pi[1:-1])
            tNo = _normal(allvn[M[row,arr[1:-1]]], tN)
 
            tNl0 = np.cross(Bi[0]-Pi[0],Pi[1]-Pi[0])
            tNl0 = _normal(allvn[M[row-1,0]], tNl0)
                
            tNr0 = -np.cross(Bi[-1]-Pi[-1],Pi[-2]-Pi[-1])
            tNr0 = _normal(allvn[M[row-1,-1]], tNr0)
            
            tN = np.vstack((tNl0,tNo,tNr0))
            "normals of osculating planes"
            oN = np.cross(tN, Bi-Pi)
            oN = oN / np.linalg.norm(oN,axis=1)[:,None] 

            an,anl,anr,anoo = Pi[1:-1],Pi[:-2],Pi[2:,:],Bi[1:-1]
            tNo,tNl,tNr = tN[1:-1],tN[:-2],tN[2:]

            P1,P2 = np.array([]),np.array([])
            for i in range(len(an)):
                plt1, plt2 = Plane(anl[i],tNl[i]), Plane(anr[i],tNr[i])
                l = intersection_plane_plane(plt1,plt2)
                P1,P2 = np.r_[P1,l[0]],np.r_[P2,l[1]]
            P1,P2 = P1.reshape(-1,3),P2.reshape(-1,3)
            
            "if an in line=intersec(T1,T2),then nexP should be as red as possible."
            ck = np.linalg.norm(np.cross(anoo-P1,P2-P1),axis=1) # if anoo / an
            colinear = np.r_[colinear,ck]
            #print('%.2g' %np.mean(ck),'%.2g' %np.max(ck))
            
            vec = (P2-P1)/np.linalg.norm(P1-P2,axis=1)[:,None]
            #lamda = np.linalg.norm(anl-anr,axis=1) * 0.5 
            lamda = np.linalg.norm(an-anoo,axis=1)
            ind = np.where(np.einsum('ij,ij->i',vec,an-anoo)<0)[0]
            lamda[ind] = -lamda[ind]
            if True:
                P2 = an + vec * lamda[:,None]  # set new Pi as P2
                P20,P21 = 2*Pi[0]-Bi[0],2*Pi[-1]-Bi[-1]
                P2 = np.vstack((P20,P2,P21))
            else:
                P2 = anoo + vec * lamda[:,None]*2  # set new Pi as P2
                P20,P21 = 2*Pi[0]-Bi[0],2*Pi[-1]-Bi[-1]
                P2 = np.vstack((P20,P2,P21))
            
            # Pi = np.array([])
            # for i in range(len(an)):
            #     a1 = [an[i],oN[i]]
            #     b1 = [anl[i],tNl[i]]
            #     c1 = [anr[i],tNr[i]]
            #     P = intersect_plane_plane_plane(a1,b1,c1)
            #     Pi = np.r_[Pi,P]
            # Pi = Pi.reshape(-1,3) 
            # self.nextP = P2#Pi # suose colinear is good enough, directly choose P2.
            
            Bi = Pi
            Pi = P2
            vlist = np.vstack((vlist,Pi))
            ex_an = np.vstack((ex_an,Bi))
            ex_P2 = np.vstack((ex_P2,P2))
            ex_tno = np.vstack((ex_tno,tN))
            ex_on = np.vstack((ex_on,oN))     
            x1 = np.mean(np.linalg.norm(anl-anr,axis=1))
            x2 = np.mean(np.linalg.norm(anoo-an,axis=1))
            xx1 = np.r_[xx1,np.ones(num)*x1* 0.25]
            xx2 = np.r_[xx2,np.ones(num)*x2* 1.1]

        m = np.arange((row+1)*num).reshape(row+1,-1) 
        v1,v4 = m[:-1,:-1], m[:-1,1:]
        v2,v3 = m[1:,:-1], m[1:,1:]              
        flist = np.c_[v1.flatten(),v2.flatten(),v3.flatten(),v4.flatten()]           
  
        sm = Mesh()
        sm.make_mesh(vlist,flist.tolist())
        return sm,[ex_an[1:],ex_P2[1:],ex_tno[1:],ex_on[1:]],colinear,[xx1,xx2]


    def get_polyline_from_an_edge(self,e,is_halfedge=False,is_poly=True): 
        H = self.halfedges
        vl = vr = np.array([],dtype=int)
        if is_halfedge:
            il,ir = e,H[e,4]
        else:
            ##print(np.where(H[:,5]==e),H[e,0],H[H[e,4],0])
            il,ir = np.where(H[:,5]==e)[0] # should be two
        vl = np.r_[vl,H[il,0]]
        while H[H[H[il,2],4],1]!=-1 and H[H[H[H[H[il,2],4],2],4],1]!=-1:
            il = H[H[H[il,2],4],2]
            if H[il,0] in vl:
                break
            vl = np.r_[vl, H[il,0]]
        vl = np.r_[vl,H[H[il,4],0]]
        while H[H[H[ir,2],4],1]!=-1 and H[H[H[H[H[ir,2],4],2],4],1]!=-1:
            ir = H[H[H[ir,2],4],2]
            if  H[H[ir,4],0] in vr:
                break
            vr = np.r_[vr, H[H[ir,4],0]]     
        "Vl = self.vertices[vl[::-1]]; Vr = self.vertices[vr]"
        iv = np.r_[vl[::-1],vr]
        VV = self.vertices[iv]
        if is_poly:
            poly = self.make_polyline_from_endpoints(VV[:-1,:],VV[1:,:])
            return iv,VV,poly
        return iv,VV
    
    def get_polylines_from_edges(self,es):
        ivs = np.array([],dtype=int)
        ns = []
        for e in es:
            iv,VV = self.get_polyline_from_an_edge(e,is_poly=False)
            ivs = np.r_[ivs, iv]
            ns.append(len(VV)-1)
        ns = np.array(ns)
        polys = self.make_multiple_polylines_from_endpoints(self.vertices[ivs], ns)
        return ivs, self.vertices[ivs], polys
    
    def make_multiple_polylines_from_endpoints(self,VV,ns):
        def _multiple_segments_cell_array(ns):
            "ns is an array of nums of each segment"
            ci = np.array([],dtype=int)
            a = 0
            for n in ns:
                i = a + np.arange(n)
                ci = np.r_[ci,i]
                a += n + 1
            c = np.ones(len(ci))*2
            cj = ci + 1 
            cells = np.vstack((c,ci,cj)).T
            cells = np.ravel(cells)
            return cells          
        data = _multiple_segments_cell_array(ns)
        poly = Polyline(VV)
        poly.cell_array=data
        return poly   
    
    def get_multiple_polyline_cylinder_from_endpoints(self,Vl,Vr,C,r,h,Fa,Fv): # no use
        "stack mesh_cylinder as polylines, from endpoints, including closed one"
        "similar to self.get_sphere_packing()"
        #C,r,h,Fa=20,Fv=20
        import mesh_cylinder
        num = C.shape[0]
        M0 = mesh_cylinder(C[0],r[0],h[0],Fa,Fv)
        for i in range(num-1):
            Si = mesh_cylinder(C[i+1], r[i+1], h[i+1], Fa,Fv)
            half = Si.halfedges
            V,E,F = Si.V, Si.E, Si.F
            M0.vertices = np.vstack((M0.vertices, Si.vertices))
            half[:,0] += (i+1)*V
            half[:,1] += (i+1)*F
            half[:,2] += (i+1)*2*E
            half[:,3] += (i+1)*2*E
            half[:,4] += (i+1)*2*E
            half[:,5] += (i+1)*2*E
            M0.halfedges = np.vstack((M0.halfedges, half))
        M0.topology_update()
        return M0
    
    def get_polylines_fair_index(self,es,vvv,vva,vvb):
        "vvv=[v,v],vva=[v1,v2],vvb=[v3,v4]"
        #v,v1,v2,v3,v4 = self.ver_regular_star.T
        ivs,_,_ = self.get_polylines_from_edges(es)
        idvl = []
        for iv in ivs:
            iia = np.where(vva==iv)[0]
            if len(iia)!=0:
                for i in iia:
                    if vvv[i] in ivs and vvb[i] in ivs:
                        idvl.append(i)
        return np.array(idvl,dtype=int)    
    
    def get_polylines_region_fair_index(self,es,vvv,vva,vvb):
        "vvv=[v,v],vva=[v1,v2],vvb=[v3,v4]"
        ivs,_,_ = self.get_polylines_from_edges(es)
        idvl = []
        for iv in ivs:
            iia = np.where(vva==iv)[0]
            if len(iia)!=0:
                for i in iia:
                    if vvv[i] in ivs and vvb[i] in ivs:
                        idvl.append(i)
        return np.array(idvl,dtype=int)
     

    def get_symmtric_points_about_polyline(self,vs,ivpoly): 
        H = self.halfedges
        Vi = self.vertices
        pair = np.array([],dtype=int)
        for v in vs:
            e = H[np.where(H[:,0]==v)[0],5]
            for i in range(4):
                ivp,_ = self.get_polyline_from_an_edge(e[i],is_poly=False)
                if len(np.intersect1d(ivpoly,ivp))!=0:
                    vp = np.intersect1d(ivpoly,ivp) # maybe more than 1 instersection
                    id1 = np.where(ivp==v)[0]
                    
                    if len(vp)!=1:
                        d1=np.linalg.norm(Vi[vp[0]]-Vi[v])
                        d2=np.linalg.norm(Vi[vp[1]]-Vi[v])
                        vp = vp[0] if d1<d2 else vp[1]
                    
                    id2 = np.where(ivp==vp)[0]
                    #print(v,vp, id1,id2) # note: may have prolem for two intersection points
 
                    vp = ivp[2*id2-id1]
                    pair = np.r_[pair,v,vp]
                    break
        return pair.reshape(-1,2),self.vertices[pair]
    
    ###---------------------------------
    def get_quadface_diagonal(self):
        "for isometry_checkerboard"
        vi = self.quadface
        v1,v2,v3,v4 = vi[::4],vi[1::4],vi[2::4],vi[3::4]
        ld1 = np.linalg.norm(self.vertices[v1]-self.vertices[v3],axis=1)
        ld2 = np.linalg.norm(self.vertices[v2]-self.vertices[v4],axis=1)
        ud1 = (self.vertices[v1]-self.vertices[v3]) / ld1[:,None]
        ud2 = (self.vertices[v2]-self.vertices[v4]) / ld2[:,None]
        return ld1,ld2,ud1,ud2
    
    def get_quad_diagonal(self,V,plot=False):
        "isogonal_ck_based: get diagonal edge_length / unit_vector"
        v1,v2,v3,v4 = self.rr_quadface.T # in odrder
        ld1 = np.linalg.norm(V[v1]-V[v3],axis=1)
        ld2 = np.linalg.norm(V[v2]-V[v4],axis=1)
        ud1 = (V[v1]-V[v3]) / ld1[:,None]
        ud2 = (V[v2]-V[v4]) / ld2[:,None]
        anchor = (V[v1]+V[v2]+V[v3]+V[v4]) / 4
        if plot:
            a,b = ld1/4.0, ld2/4.0
            Vl,Vr = anchor+ud1*a[:,None], anchor-ud1*a[:,None]
            pl1 = self.make_polyline_from_endpoints(Vl,Vr)
            Vl,Vr = anchor+ud2*b[:,None], anchor-ud2*b[:,None] 
            pl2 = self.make_polyline_from_endpoints(Vl,Vr)
            return pl1,pl2          
        return ld1,ld2,ud1,ud2,anchor

    def get_quad_midpoint_cross_vectors(self,halfdiag=True,plot=False):
        "isogonal_face_based: get quadface midpoint edge_length / unit_vector"
        if halfdiag:
            ib,ir = self.vertex_check_ind
            _,v1,v2,v3,v4 = self.rr_star.T
            v1,v2,v3,v4 = v1[ib],v2[ib],v3[ib],v4[ib]
        else:   
            v1,v2,v3,v4 = self.rr_quadface.T # in odrder
        e1 = 0.5*(self.vertices[v2]+self.vertices[v3]-self.vertices[v1]-self.vertices[v4])
        e2 = 0.5*(self.vertices[v3]+self.vertices[v4]-self.vertices[v1]-self.vertices[v2])
        ld1 = np.linalg.norm(e1,axis=1)
        ld2 = np.linalg.norm(e2,axis=1)
        ud1 = e1 / ld1[:,None]
        ud2 = e2 / ld2[:,None]
        anchor = (self.vertices[v1]+self.vertices[v2]+self.vertices[v3]+self.vertices[v4]) / 4
        if plot:
            a,b = ld1/3.0, ld2/3.0
            Vl,Vr = anchor+ud1*a[:,None], anchor-ud1*a[:,None]
            pl1 = self.make_polyline_from_endpoints(Vl,Vr)
            Vl,Vr = anchor+ud2*b[:,None], anchor-ud2*b[:,None] 
            pl2 = self.make_polyline_from_endpoints(Vl,Vr)
            return pl1,pl2            
        return ld1,ld2,ud1,ud2,anchor
    
    def get_checkboard_black(self):
        vi = self.quadface
        v1,v2,v3,v4 = vi[::4],vi[1::4],vi[2::4],vi[3::4]
        V = self.vertices
        P1 = (V[v1] + V[v2]) * 0.5
        P2 = (V[v2] + V[v3]) * 0.5
        P3 = (V[v3] + V[v4]) * 0.5
        P4 = (V[v4] + V[v1]) * 0.5
        ck = self.make_quad_mesh_pieces(P1,P2,P3,P4)
        return ck

    def get_checkboard_white(self):
        V = self.vertices
        v,v1,v2,v3,v4 = self.ver_regular_star.T
        P1 = (V[v1] + V[v]) * 0.5
        P2 = (V[v2] + V[v]) * 0.5
        P3 = (V[v3] + V[v]) * 0.5
        P4 = (V[v4] + V[v]) * 0.5
        ck = self.make_quad_mesh_pieces(P1,P2,P3,P4)
        return ck

    def quadface_neighbour_star(self,boundary=True):
        """rr_quadface_tristar: [f,fi,fj]
        get the star metrix of forde/rr_quadface
            f in forder, its neighbour f1234 also in forder
            get the corresponding order of forder
        """
        H = self.halfedges
        forder = self.rr_quadface_order
        farr = self.rr_quadface
        numf = self.num_rrf
        id00,id01,idl,idr,idu,idd=[],[],[],[],[],[]
        if boundary:
            iddd,iddl,iddr = [],[],[]
        if0,if1,if2,if3,if4 = [],[],[],[],[]
        for i in range(numf):
            first = farr[i]
            e1=np.where(H[:,0]==first[0])[0]
            e2=np.where(H[H[:,4],0]==first[1])[0]
            e = np.intersect1d(e1,e2)
            if H[e,1]==-1:
                 e = H[e,4]
            e1 = H[H[e,3],4] # left
            e2 = H[e,4] # down
            e3 = H[H[e,2],4] #right
            e4 = H[H[H[e,2],2],4] #up
            if H[e1,1]!=-1 and H[e3,1]!=-1 and H[e1,1] in forder and H[e3,1] in forder:
                id00.append(i)
                idl.append(np.where(forder==H[e1,1])[0][0])
                idr.append(np.where(forder==H[e3,1])[0][0])
            if H[e2,1]!=-1 and H[e4,1]!=-1 and H[e2,1] in forder and H[e4,1] in forder:
                id01.append(i)
                idd.append(np.where(forder==H[e2,1])[0][0])
                idu.append(np.where(forder==H[e4,1])[0][0])
                
            if H[e1,1]!=-1 and H[e3,1]!=-1 and H[e1,1] in forder and H[e3,1] in forder:
                if H[e2,1]!=-1 and H[e4,1]!=-1 and H[e2,1] in forder and H[e4,1] in forder:
                    if0.append(i)
                    if1.append(np.where(forder==H[e1,1])[0][0])
                    if3.append(np.where(forder==H[e3,1])[0][0])
                    if2.append(np.where(forder==H[e2,1])[0][0])
                    if4.append(np.where(forder==H[e4,1])[0][0])  
                
            if boundary:
                "edge e has four positon: up,down,left,right"
                fd,fu = H[H[e,4],1], H[H[H[H[e,2],2],4],1]
                fl,fr = H[H[H[e,3],4],1], H[H[H[e,2],4],1]
                if fd==-1 and (fl!=-1 or fr!=-1):
                    try:
                        a = np.where(forder==fl)[0][0]
                        b = np.where(forder==fr)[0][0]
                        iddd.append(i)
                        iddl.append(a)
                        iddr.append(b)
                    except:
                        pass
                if fu==-1 and (fl!=-1 or fr!=-1):
                    try:
                        a = np.where(forder==fl)[0][0]
                        b = np.where(forder==fr)[0][0]
                        iddd.append(i)
                        iddl.append(b)
                        iddr.append(a)
                    except:
                        pass
                if fl==-1 and (fd!=-1 or fu!=-1):
                    try:
                        a = np.where(forder==fd)[0][0]
                        b = np.where(forder==fu)[0][0]
                        iddd.append(i)
                        iddl.append(b)
                        iddr.append(a)
                    except:
                        pass
                if fr==-1 and (fd!=-1 or fu!=-1):
                    try:
                        a = np.where(forder==fd)[0][0]
                        b = np.where(forder==fu)[0][0]
                        iddd.append(i)
                        iddl.append(a)
                        iddr.append(b)
                    except:
                        pass     
        if boundary:
            ###print(iddd,iddl,iddr)
            tristar = np.vstack((np.r_[id00,id01,iddd],np.r_[idl,idu,iddl],np.r_[idr,idd,iddr])).T
        else:
            tristar = np.vstack((np.r_[id00,id01],np.r_[idl,idu],np.r_[idr,idd])).T
        self._rr_quadface_tristar = tristar
        self._rr_quadface_4neib = np.c_[if0,if1,if2,if3,if4]
        
    def quadface_S_star_data(self):
        """
        regular quadfaces whose neighbouring 4-faces are quadfaces
        If the 5-quadfaces are planar and tangent to a common sphere,i.e. S*-net
        anypoint in tangent plane: pi (e.g. mid point of quadplane)
        contact(tangent) point: x
        ==> n*(x-p)=0 <==> n*x = n*p
        relation about radius: 
            ni * (xi-C) = r <==> ni*C - ni*pi = -r, i=0,1,2,3,4
        <==> (nj-n0)*C + dj-d0 = 0, where dj=-nj*pj, j=1,2,3,4
        solve C=(x,y,z) from over-constrained 4-equations
        """
        V = self.vertices
        v1,v2,v3,v4  = self.rr_quadface.T
        forder = self.rr_quadface_order
        if0,if1,if2,if3,if4 = self.rr_quadface_4neib.T
        n = np.cross(V[v3[if0]]-V[v1[if0]],V[v4[if0]]-V[v2[if0]])
        n_up = np.cross(V[v3[if1]]-V[v1[if1]],V[v4[if1]]-V[v2[if1]])
        n_dn = np.cross(V[v3[if3]]-V[v1[if3]],V[v4[if3]]-V[v2[if3]])
        n_lf = np.cross(V[v3[if2]]-V[v1[if2]],V[v4[if2]]-V[v2[if2]])
        n_rt = np.cross(V[v3[if4]]-V[v1[if4]],V[v4[if4]]-V[v2[if4]])
        
        n0 = n / np.linalg.norm(n,axis=1)[:,None]
        n1 = n_up / np.linalg.norm(n_up,axis=1)[:,None]
        n3 = n_dn / np.linalg.norm(n_dn,axis=1)[:,None]
        n2 = n_lf / np.linalg.norm(n_lf,axis=1)[:,None]
        n4 = n_rt / np.linalg.norm(n_rt,axis=1)[:,None]
        
        bary = self.face_barycenters()
        p0 = bary[forder[if0]]
        p1, p3 = bary[forder[if1]], bary[forder[if3]]
        p2, p4 = bary[forder[if2]], bary[forder[if4]]
        
        d0 = -np.einsum('ij,ij->i',n0,p0)
        d1 = -np.einsum('ij,ij->i',n1,p1)
        d2 = -np.einsum('ij,ij->i',n2,p2)
        d3 = -np.einsum('ij,ij->i',n3,p3)
        d4 = -np.einsum('ij,ij->i',n4,p4)
        
        num = len(if0)
        row = np.arange(4*num).repeat(3)
        col = np.tile(np.arange(3*num),4)
        data = np.r_[(n1-n0).flatten(),(n2-n0).flatten(),(n3-n0).flatten(),(n4-n0).flatten()]
        A = sparse.coo_matrix((data,(row,col)), shape=(4*num,3*num))
        B = np.r_[d0-d1,d0-d2,d0-d3,d0-d4]
        #try:
            #from pypardiso import spsolve
        #except:
        from scipy.sparse.linalg import spsolve
        C = spsolve(A.T*A, A.T*B, permc_spec=None, use_umfpack=True).reshape(-1,3)
        #C = sparse.linalg.lsqr(A, B)[0].reshape(-1,3)
        #print(C)
        r0 = np.einsum('ij,ij->i',p0-C,n0)
        r1 = np.einsum('ij,ij->i',p1-C,n1)
        r2 = np.einsum('ij,ij->i',p2-C,n2)
        r3 = np.einsum('ij,ij->i',p3-C,n3)
        r4 = np.einsum('ij,ij->i',p4-C,n4)
        x0 = C + n0*r0[:,None]
        x1 = C + n1*r1[:,None]
        x2 = C + n2*r2[:,None]
        x3 = C + n3*r3[:,None]
        x4 = C + n4*r4[:,None]
        
        if True:
            "orient the normals"
            def _orient(ni,pi,C):
                i = np.where(np.einsum('ij,ij->i',pi-C,ni)<0)[0]
                ni[i] = -ni[i]
                return ni
            n0 = _orient(n0,p0,C)
            n1 = _orient(n1,p1,C)
            n2 = _orient(n2,p2,C)
            n3 = _orient(n3,p3,C)
            n4 = _orient(n4,p4,C)
            d0 = -np.einsum('ij,ij->i',n0,p0)
            d1 = -np.einsum('ij,ij->i',n1,p1)
            d2 = -np.einsum('ij,ij->i',n2,p2)
            d3 = -np.einsum('ij,ij->i',n3,p3)
            d4 = -np.einsum('ij,ij->i',n4,p4)
            
        err1 = (np.einsum('ij,ij->i',n1-n0,C) + d1-d0) / np.linalg.norm(n1-n0,axis=1)
        err2 = (np.einsum('ij,ij->i',n2-n0,C) + d2-d0) / np.linalg.norm(n2-n0,axis=1)
        err3 = (np.einsum('ij,ij->i',n3-n0,C) + d3-d0) / np.linalg.norm(n3-n0,axis=1)
        err4 = (np.einsum('ij,ij->i',n4-n0,C) + d4-d0) / np.linalg.norm(n4-n0,axis=1)
        err = np.sqrt(err1**2+err2**2+err3**2+err4**2)
        err2 = (np.abs(r1-r0)+np.abs(r2-r0)+np.abs(r3-r0)+np.abs(r4-r0))/r0
        err3 = np.sqrt((r1-r0)**2+(r2-r0)**2+(r3-r0)**2+(r4-r0)**2)/np.abs(r0)
        return C, [r0,r1,r2,r3,r4],[x0,x1,x2,x3,x4],[n0,n1,n2,n3,n4],[err,err2,err3]

    def quadface_checkerboard_order(self):
        "self.quad_check : regular_regular"
        fblue,fred = self.checker_face
        forder = self.rr_quadface_order
        blue,green = [],[]
        for i in range(len(forder)):
            f = forder[i]
            if f in fblue:
                blue.append(i)
            else:
                green.append(i)
        self._quad_check = [np.array(blue),np.array(green)]

    def regular_vertex_checkerboard_order(self):
        "self.vertex_check, vertex_check_ind : regular_regular"
        "seperate all regular vertex into blue / red"
        vblue,vred = self.checker_vertex
        star = self.rr_star
        regular = star[:,0]
        idblue,idred=[],[]
        blue,red=[],[]
        bstar,rstar=[],[]
        for i in range(len(regular)):
            v = regular[i]
            if v in vblue:
                idblue.append(i)
                blue.append(v)
                bstar.append(star[i,:])
            else:
                idred.append(i)
                red.append(v)
                rstar.append(star[i,:])
        self._vertex_check=[np.array(blue),np.array(red)]
        self._vertex_check_star=[np.array(bstar),np.array(rstar)] #vertex_check_star
        self._vertex_check_ind=[np.array(idblue),np.array(idred)]
        #print(len(star),len(blue),len(red))

    # -------------------------------------------------------------------------
    #                       all faces / vertices
    # -------------------------------------------------------------------------
    def get_checker_select_vertex(self,first=[0]):
        "self.checker_vertex: all vertex...blue ~ 0 ; red ~ 1; left ~ -1"
        vall,vallij=self.vertex_ring_vertices_iterators(sort=True, order=False)
        #vall,vallij = self._vi,self._vj
        arr = -np.ones(self.V)
        left = np.arange(self.V)
        i0 = first       ## f0
        arr[i0] = 0      ## blue
        ind = np.where(vall==i0)[0]
        ij = vallij[ind] ## fneib
        left = np.setdiff1d(left,np.r_[i0,ij])
        a = 0
        while len(left)!=0:
            a = a % 2
            vneib=[]
            for i in ij:
                if i !=-1:
                    arr[i]=1-a
                    ind = np.where(vall==i)[0]
                    vneib.extend(list(vallij[ind]))
            left = np.setdiff1d(left,np.r_[ij,vneib])
            ij=[]
            for v in vneib:
                if v!=-1 and arr[v]==-1:
                    arr[v]=a
                    ij.append(v)
            a += 1
        blue = np.where(arr==0)[0]
        red = np.where(arr==1)[0]
        self._checker_vertex = [blue,red]

    def get_checker_select_faces(self,first=[0]):
        "self.checker_face...blue ~ 0 ; red ~ 1; left ~ -1"
        fall,fallij = self.face_neighbour_faces()
        arr = -np.ones(self.F)
        left = np.arange(self.F)
        i0 = first       ## f0
        arr[i0] = 0      ## blue
        ind = np.where(fall==i0)[0]
        ij = fallij[ind] ## fneib
        left = np.setdiff1d(left,np.r_[i0,ij])
        a = 0
        while len(left)!=0:
            a = a % 2
            fneib=[]
            for i in ij:
                if i !=-1:
                    arr[i]=1-a
                    ind = np.where(fall==i)[0]
                    fneib.extend(list(fallij[ind]))
            left = np.setdiff1d(left,np.r_[ij,fneib])
            ij=[]
            for f in fneib:
                if f!=-1 and arr[f]==-1:
                    arr[f]=a
                    ij.append(f)
            a += 1
        blue = np.where(arr==0)[0]
        red = np.where(arr==1)[0]
        self._checker_face = [blue,red]
    
    def get_checker_subgroup_vertices(self,tian=False,subpartial=False,k=4):
        "self.checker_vertex_tian"
        blue,red = self.checker_vertex
        iall = self.get_both_isopolyline()
        ib = []

        if tian:
            for iv in iall:
                if iv[0] in blue:
                    ib.extend(iv[::2])
            ib = np.array(ib)
            self._checker_vertex_tian_corner = ib
        elif subpartial:
            "only work for patch-mesh, need to redo with singular case" #TODO
            try:
                M = self.patch_matrix
            except:
                M = self.rot_patch_matrix
            ib = M[::k,:].flatten()
            ib = np.unique(np.r_[ib,M[:,::k].flatten()])
            self._checker_vertex_tian_corner = M[::k,::k].flatten()
            
        iy = np.setdiff1d(np.arange(self.V),ib)
        self._checker_vertex_tian = [ib,iy]

    def get_patch_checker_ctrl_or_diag_polyedges(self,diag=False,k=4):
        "patchmesh: subtract-ctrl/diag poly by dividing k"
        V = self.vertices
        try:
            M = self.patch_matrix
        except:
            M = self.rot_patch_matrix
        inn_p = np.setdiff1d(M[::k,::k].flatten(),self.boundary_vertices())
        if diag:
            def _matrix(M):
                a,b = M.shape
                iv,_ = self.get_diagonal_polyline_from_2points([M[0,0],M[1,1]],is_poly=False)
                pl1,pr1 = iv[:-1],iv[1:]
                for i in range(a):
                    if i!=0 and i!=a-1 and i%k==0:
                        vis = [M[i,0],M[i+1,1]]
                        iv,_= self.get_diagonal_polyline_from_2points(vis,is_poly=False)
                        if len(iv)!=k+1:
                            pl1 = np.r_[pl1,iv[:-1]]
                            pr1 = np.r_[pr1,iv[1:]]
                for j in range(b):
                    if j!=0 and j!=b-1 and j%k==0:
                        vis = [M[0,j],M[1,j+1]]
                        iv,_ = self.get_diagonal_polyline_from_2points(vis,is_poly=False)
                        if len(iv)!=k+1:
                            pl1 = np.r_[pl1,iv[:-1]]
                            pr1 = np.r_[pr1,iv[1:]]
                ply1 = self.make_polyline_from_endpoints(V[pl1],V[pr1])
                p1 = np.setdiff1d(np.r_[pl1,pr1],self.boundary_vertices())
                return V[p1],ply1
            P1,ply1 = _matrix(M)
            P2,ply2 = _matrix(np.rot90(M))
        else:
            "inner k-th_row + k-th_col edges [pl-pr]"
            pl1 = M[::k,:-1][1:-1,:].flatten()
            pr1 = M[::k,1:][1:-1,:].flatten()
            pl2 = M[:-1,::k][:,1:-1].flatten()
            pr2 = M[1:,::k][:,1:-1].flatten()
            ply1_p = np.setdiff1d(np.r_[pl1,pr1],self.boundary_vertices())
            ply2_p = np.setdiff1d(np.r_[pl2,pr2],self.boundary_vertices())
            ply1 = self.make_polyline_from_endpoints(V[pl1],V[pr1])
            ply2 = self.make_polyline_from_endpoints(V[pl2],V[pr2])
            P1,P2 = V[ply1_p],V[ply2_p]
        return V[inn_p],P1,P2,ply1,ply2
                          
        
    def get_patch_checker_subbroup_mesh(self,diag=False,denser=False,
                                        tian=False,subpartial=False,k=4):
        V = self.vertices
        if tian:
            try:
                M = self.patch_matrix[::k,::k]
            except:
                M = self.rot_patch_matrix[::k,::k]
            if diag:
                if denser:
                    m = self.make_denser_diag_patch_mesh_from_matrix(V,M)
                else:
                    m = self.make_diag_patch_mesh_from_matrix(V,M)
            else:
                "only corner vertives"
                m = self.make_patch_mesh_from_matrix(V,M)
        elif subpartial:
            pass
        return m


    def get_vertex_neighbor_mean_edge_length(self,allv=True,rr=False,smallest=True):
        Vi = self.vertices
        if allv:
            v,vj = self._vi,self._vj
            alll = np.linalg.norm(Vi[vj]-Vi[v],axis=1)
            l = np.zeros(self.V)
            for i in range(self.V):
                j = np.where(v==i)[0]
                if smallest:
                    l[i] = np.min(alll[j])
                else:
                    l[i] = np.mean(alll[j])
            return l
        else:
            "regular vertex star:"
            if rr:
                star = self.rr_star
                star = star[self.ind_rr_star_v4f4]
                v,v1,v2,v3,v4 = star.T
            else:
                v,v1,v2,v3,v4 = self.ver_regular_star.T
            V0,V1,V2,V3,V4 = Vi[v],Vi[v1],Vi[v2],Vi[v3],Vi[v4]
            l1 = np.linalg.norm(V1-V0,axis=1)
            l2 = np.linalg.norm(V2-V0,axis=1)
            l3 = np.linalg.norm(V3-V0,axis=1)
            l4 = np.linalg.norm(V4-V0,axis=1)
            if smallest:
                return np.min(np.c_[l1,l2,l3,l4],axis=1)
            else:
                return (l1+l2+l3+l4) / 4
       
    # -------------------------------------------------------------------------
    #                        Diagonal mesh
    # -------------------------------------------------------------------------

    def _red_regular_vertex0(self,blue): # problem for strip_model_2.obj
        order = self.ver_regular
        red = []
        while len(order)!=0:
            v = order[0]
            vring = np.r_[v,self.ringlist[v]]
            order = np.setdiff1d(order,vring)
            i = np.where(self.ver_regular==v)[0][0]
            red.append(i)
        if blue:
            red = np.setdiff1d(np.arange(self.num_regular),red)
        return red

    def red_regular_regular_vertex(self,blue=False):
        vblue,vred = self.checker_vertex # depends on all vertex-checker
        order = self.rr_star[:,0][self.ind_rr_star_v4f4]
        idb,idr = [],[]
        for i in range(len(order)):
            v = order[i]
            if v in vblue:
                idb.append(i)
            elif v in vred:
                idr.append(i)
        if blue:
            self._red_regular_vertex = np.array(idb)
        self._red_regular_vertex = np.array(idr)
    
    def ind_checker_regular_regular_vertex(self,tian_partial=False):
        "if tian: substract quadvertices of checker"
        if tian_partial:
            vblue,vred = self.checker_vertex_tian
        else:
            vblue,vred = self.checker_vertex
        order = self.rr_star[:,0][self.ind_rr_star_v4f4]
        idb,idr = [],[]
        for i in range(len(order)):
            v = order[i]
            if v in vblue:
                idb.append(i)
            elif v in vred:
                idr.append(i)
        if tian_partial:
            self._ind_ck_tian_regular_vertex = [np.array(idb),np.array(idr)]
        else:
            self._ind_ck_regular_vertex = [np.array(idb),np.array(idr)]
    
    def _ind_red_regular_vertex(self,blue):
        vblue,vred = self.checker_vertex # depends on all vertex-checker
        order = self.ver_regular
        idb,idr = [],[]
        for i in range(len(order)):
            v = order[i]
            if v in vblue:
                idb.append(i)
            elif v in vred:
                idr.append(i)
        if blue:
            return np.array(idb)
        return np.array(idr)

    def get_diagonal_mesh(self,sort=True,blue=False,whole=False):
        V = self.vertices
        if whole:
            num = self.V
            bary = self.face_barycenters()
            dV = np.vstack((V,bary))
            H = self.halfedges
            i1 = np.where(H[:,1] >= 0)[0]
            i2 = np.where(H[H[:,4],1] >= 0)[0]
            i = np.intersect1d(i1,i2)
            e = np.array([i[0]])
            for j in i[1:]:
                if H[H[j,4],1] not in H[e,1]:
                    e = np.r_[e,j]
            v1, v3 = H[e,0], H[H[e,4],0]
            v2, v4 = num+H[H[e,4],1], num+H[e,1]
            dallv = np.unique(np.r_[v1,v2,v3,v4])
            vlist = dV[dallv]
            iv1 = [np.argwhere(dallv == item)[0][0] for item in v1]
            iv2 = [np.argwhere(dallv == item)[0][0] for item in v2]
            iv3 = [np.argwhere(dallv == item)[0][0] for item in v3]
            iv4 = [np.argwhere(dallv == item)[0][0] for item in v4]
            flist = (np.array([iv1,iv2,iv3,iv4]).T).tolist()
            dmesh = Mesh()
            dmesh.make_mesh(vlist,flist)
            return dmesh
        else:
            v,v1,v2,v3,v4 = self.ver_regular_star.T
            if sort:
                order = self._ind_red_regular_vertex(blue)
            else:
                order = np.arange(self.num_regular)[::2]
            # V1,V2,V3,V4 = V[v1[order]],V[v2[order]],V[v3[order]],V[v4[order]]
            # dmesh = self.make_quad_mesh_pieces(V1,V2,V3,V4)
            v1,v2,v3,v4 = v1[order],v2[order],v3[order],v4[order]
            dmesh = self.make_quad_mesh_from_indices(V,v1,v2,v3,v4)
            return dmesh

    def get_agnet_surf_normal(self,mesh,is_AAG_or_GGA,diag=False):
        """no matter if it's ctrl-net or diag-net
        surface normals Ni:
            AAG: normal from planar-vertex-star
            GGA: normal from G-net, i.e. intersection of two osculating planes
        ##Represent Darboux Frame and corresponding Frenet Frame
        ##Ti: tangents formed by 3 consecutive vertices
        ##Bi: = Ni x Ti
        """
        V = mesh.vertices
        v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        v,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
        V0,V1,V2,V3,V4,Va,Vb,Vc,Vd = V[v],V[v1],V[v2],V[v3],V[v4],V[va],V[vb],V[vc],V[vd]
        allN = mesh.vertex_normals()
        vnn = allN[v]
        if is_AAG_or_GGA:
            "AAGnet or AAGGnet"
            if diag:
                "diagonal-net is A-net, get A-net normals:"
                an1,an2 = np.cross(Vb-V0,Va-V0),np.cross(Vd-V0,Vc-V0)
            else:
                an1,an2 = np.cross(V2-V0,V1-V0),np.cross(V4-V0,V3-V0)
        else:
            "GGAnet: Ni // e1+e3 // e2+e4"
            if diag:
                e1 = (Va-V0) / np.linalg.norm(Va-V0,axis=1)[:,None]
                e2 = (Vb-V0) / np.linalg.norm(Vb-V0,axis=1)[:,None]
                e3 = (Vc-V0) / np.linalg.norm(Vc-V0,axis=1)[:,None]
                e4 = (Vd-V0) / np.linalg.norm(Vd-V0,axis=1)[:,None]
            else:
                e1 = (V1-V0) / np.linalg.norm(V1-V0,axis=1)[:,None]
                e2 = (V2-V0) / np.linalg.norm(V2-V0,axis=1)[:,None]
                e3 = (V3-V0) / np.linalg.norm(V3-V0,axis=1)[:,None]
                e4 = (V4-V0) / np.linalg.norm(V4-V0,axis=1)[:,None]
            an1, an2 = e1+e3, e2+e4
        an1 = an1 / np.linalg.norm(an1,axis=1)[:,None]
        id1 = np.where(np.einsum('ij,ij->i',vnn,an1)<0)[0]
        an1[id1]=-an1[id1]
        an2 = an2 / np.linalg.norm(an2,axis=1)[:,None]
        id2 = np.where(np.einsum('ij,ij->i',vnn,an2)<0)[0]
        an2[id2]=-an2[id2]
        an = (an1+an2) / np.linalg.norm(an1+an2,axis=1)[:,None]
        allN[v] = an
        return allN

    def get_poly_quintic_Bezier_spline_crvs_checker(self,mesh,normal,
                                                    efair=0.01,
                                                    is_asym_or_geo=True,
                                                    diagpoly=False,
                                                    is_one_or_another=False,
                                                    is_checker=1, ### checker seletion
                                                    interval=1,
                                                    #is_onlyinner=True,
                                                    is_dense=False,num_divide=5,
                                                    is_modify=False,
                                                    is_smooth=0.0):
        "For each polyline, 1.[Pi,Ti,Ni,ki] 2.opt to get ctrl-p,polygon,crv"
        from curvature import frenet_frame
        from bezierC2Continuity import BezierSpline
        V = mesh.vertices
        N = normal ### surf_normal n
        kck = is_checker
        inner = False if kck!=1 else True 
        
        _,_,lj = self.vertex_ring_vertices_iterators(return_lengths=True)
        o4 = np.where(lj==4)[0]
        boundary = self.boundary_vertices()
        inv4 = np.intersect1d(o4, boundary)  
        if len(inv4)==0:
            "regular-patch-shape or rotational-shape"
            iall = self.get_both_isopolyline(diagpoly,is_one_or_another,only_inner=inner,interval=interval)
        elif len(self.corner)!=0:
            "schwarzh_02_diag_unitscale_AAG_AAG"
            ipl1,ipl2 = self.get_2families_polyline_from_1closed_bdry(diag=diagpoly,
                                                         interval=interval,
                                                         inner=False) ## need to choose True or False
            iall = ipl1[1] if is_one_or_another else ipl2[1]
        an = np.array([0,0,0])
        ruling = np.array([0,0,0])
        all_kg=all_kn=all_k=all_tau=np.array([])
        arr = np.array([],dtype=int)
        varr = np.array([],dtype=int)
        num = 0
        P=Pl=Pr = np.array([0,0,0])
        crvPl=crvPr = np.array([0,0,0])
        frm1=frm2=frm3 = np.array([0,0,0])
 
        #seg_q1234,seg_vl, seg_vr = np.array([0,0,0]),[],[]
        ctrlP = []

        if kck !=1:
            if self.rot_patch_matrix is not None:
                num_poly = 1
                if diagpoly:
                    num_poly -= 1 ## -=2 IF PATCH, ELIF ROTATIOANL =-1
            elif self.patch_matrix is not None:
                num_poly,num_allpoly = 0,len(iall)-1
                if diagpoly:
                    ##NOTE: TOSELECT:
                    num_poly -= 2 ## -=2 IF PATCH, ELIF ROTATIOANL =-1
                    num_allpoly += 2
            
        row_list = []
        dense_row_list = []
        for iv in iall:
            "Except two endpoints on the boudnary"
            bool_value=False
            if kck !=1:
                if len(self.corner)!=0:
                    "if only-patch-shape:"
                    bool_value = kck !=0 and 0<num_poly and num_poly<num_allpoly and num_poly%(kck)==0 and len(iv)>=(kck*3+1)
                else:
                    "if rotational-shape"
                    bool_value = kck !=0 and num_poly%(kck)==0 and len(iv)>=(kck*3+1)
            else:
                bool_value = len(iv)>=4

            if bool_value:
                if kck==1:
                    iv_sub,iv_sub1,iv_sub3 = iv[1:-1],iv[:-2],iv[2:]
                else:
                    iv_sub,iv_sub1,iv_sub3 = iv[kck:-kck],iv[kck-1:-kck-1],iv[kck+1:-kck+1]

                Pi = V[iv_sub] # n-8
                frame = frenet_frame(Pi,V[iv_sub1],V[iv_sub3])
                "using surface normal computed by A-net or G-net"
                Ti,Ef2,Ef3 = frame ### Frenet frame (E1,E2,E3)
                Ni = N[iv_sub] ### SURF-NORMAL
                "if asym: binormal==Ni; elif geo: binormal == t x Ni"
                #E3i = Ni if is_asym_or_geo else np.cross(Ti,Ni)
                if is_asym_or_geo:
                    "asymptotic; orient binormal with surf-normal changed at inflections"
                    E3i = Ni
                    #i = np.where(np.einsum('ij,ij->i',Ef3,E3i)<0)[0]
                    #E3i[i] = -E3i[i]
                else:
                    "geodesic"
                    E3i = np.cross(Ti,Ni)

                if kck !=1:
                    "checker_vertex_partial_of_submesh case"
                    Pi = Pi[::kck]
                    Ti = Ti[::kck]
                    E3i = E3i[::kck]
                    iv_ck = iv_sub[::kck]
                    #seg_vl.extend(iv_ck[:-1])
                    #seg_vr.extend(iv_ck[1:])
                else:
                    #iv_ck = iv[1:-1] if is_onlyinner else iv
                    iv_ck = iv[1:-1]
                    #seg_vl.extend(iv[1:-2])
                    #seg_vr.extend(iv[2:-1])

                bs = BezierSpline(degree=5,continuity=3,
                                  efair=efair,itera=200,
                                  #is_onlyinner=is_onlyinner,
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
                                                            is_modify,
                                                            is_smooth)
                    dense_row_list.append(len(d))
                    an = np.vstack((an,pts))
                    frm1 = np.vstack((frm1,frmi[0])) ## Frenet-E1
                    frm2 = np.vstack((frm2,frmi[1])) ## Frenet-E2
                    frm3 = np.vstack((frm3,frmi[2])) ## Frenet-E3
                    
                    # if False:
                    #     i = np.where(np.einsum('ij,ij->i',frmi[2],d)<0)[0]
                    #     d[i] = -d[i]
                    # elif False:
                    #     d = -d
                    
                    # if is_smooth:
                    #     from huilab.huimesh.smooth import fair_vertices_or_vectors
                    #     Pup = fair_vertices_or_vectors(pts+d,itera=10,
                    #                                    efair=is_smooth,
                    #                                    is_fix=True)
                    #     d = Pup-pts
                    #     d = d / np.linalg.norm(d,axis=1)[:,None]
                        
                    arr = np.r_[arr, np.arange(len(kg)-1) + num]
                    num += len(kg)
                    
                ruling = np.vstack((ruling,d))
                all_kg,all_kn = np.r_[all_kg,kg],np.r_[all_kn,kn]
                all_k,all_tau = np.r_[all_k,k],np.r_[all_tau,tau]
                #seg_q1234 = np.vstack((seg_q1234,bs.control_points(is_Q1234=True)))
                
            if kck !=1:        
                num_poly +=1
        
        P, Pl, Pr, crvPl, crvPr = P[1:],Pl[1:],Pr[1:],crvPl[1:],crvPr[1:]   
        polygon = None #self.make_polyline_from_endpoints(Pl, Pr)
        crv = None #self.make_polyline_from_endpoints(crvPl, crvPr)
        return P,polygon,crv,np.array(ctrlP,dtype=object),\
               [an[1:],frm1[1:],frm2[1:],frm3[1:]],\
               [varr,an[1:],ruling[1:],arr,row_list,dense_row_list],\
               [all_kg,all_kn,all_k,all_tau]
        #return seg_q1234[1:],[seg_vl,seg_vr]
 
    def get_Bezier_spline_surf_ctrl_mesh(self,P1,P2,pl1_seg,pl2_seg,diag1_seg,diag2_seg):
        "from computed Bezier-spline ctr-points get Bezier-spline-surf-ctrl-pts"
        from changetopology import sub_mesh_valence4
        V = self.vertices
        H = self.halfedges
        
        def _edge(vl,vr):
            e = []
            for i in range(len(vl)):
                #print(np.where(H[:,0]==vl[i])[0],np.where(H[H[:,4],0]==vr[i])[0])
                j = np.intersect1d(np.where(H[:,0]==vl[i])[0],np.where(H[H[:,4],0]==vr[i])[0])[0]
                e.append(j)
            return np.array(e,dtype=int)

        if pl1_seg is not None and pl2_seg is not None:
            "two families of isolines"
            vl1,vr1 = pl1_seg ###[P,seg_edge]
            vl2,vr2 = pl2_seg
            
            e1,e2 = _edge(vl1,vr1), _edge(vl2,vr2)
            v,_,_,_,_ = self.rr_star_corner
            innv = V[v]
            innflist = sub_mesh_valence4(self,v,get_subflist=True)

            newVer = np.vstack((innv, P1, P2))
            newFace = []
            for frr in innflist:
                "check [v1,v2], [v2,v3], [v3,v4], [v4,v1] corresponding P1|2"
                v1,v2,v3,v4 = frr
                ei1 = np.intersect1d(np.where(H[:,0]==v1)[0],np.where(H[H[:,4],0]==v2)[0])[0]
                ei2 = np.intersect1d(np.where(H[:,0]==v2)[0],np.where(H[H[:,4],0]==v3)[0])[0]
                ei3 = np.intersect1d(np.where(H[:,0]==v3)[0],np.where(H[H[:,4],0]==v4)[0])[0]
                ei4 = np.intersect1d(np.where(H[:,0]==v4)[0],np.where(H[H[:,4],0]==v1)[0])[0]
                
                i1 = np.where(e1==ei1)[0]
                i3 = np.where(e1==ei3)[0]
                i2 = np.where(e2==ei2)[0]
                i4 = np.where(e2==ei4)[0]
                ##print(i1,i2,i3,i4)
                j1 = np.where(e2==ei1)[0]
                j3 = np.where(e2==ei3)[0]
                j2 = np.where(e1==ei2)[0]
                j4 = np.where(e1==ei4)[0]
                ##print(j1,j2,j3,j4)
                if len(i1)!=0 or len(i3)!=0:
                    "[v1,v2],[v3,v4] in polyline1, then [v2,v3],[v4,v1] in pl2"
                    if len(i1)!=0:
                        arr1 = len(v) + np.arange(4) + 4*i1[0]
                        arr3 = len(v) + np.arange(4)[::-1] + 4*np.where(H[e1,4]==ei3)[0][0]
                    elif len(i3)!=0:
                        arr1 = len(v) + np.arange(4)[::-1] + 4*np.where(H[e1,4]==ei1)[0][0]
                        arr3 = len(v) + np.arange(4) + 4*i3[0]
                    if len(i2)!=0:
                        arr2 = len(v) + np.arange(4) + len(P1) + 4*i2[0]
                        arr4 = len(v) + np.arange(4)[::-1] + len(P1) + 4*np.where(H[e2,4]==ei4)[0][0]
                    elif len(i4)!=0:
                        arr2 = len(v) + np.arange(4)[::-1] + len(P1) + 4*np.where(H[e2,4]==ei2)[0][0]
                        arr4 = len(v) + np.arange(4) + len(P1) + 4*i4[0]
                elif len(j1)!=0 or len(j3)!=0:
                    "[v1,v2],[v3,v4] in polyline2, then [v2,v3],[v4,v1] in pl1"
                    if len(j1)!=0:
                        arr1 = len(v) + np.arange(4) + len(P1) + 4*j1[0]
                        arr3 = len(v) + np.arange(4)[::-1] + len(P1) + 4*np.where(H[e2,4]==ei3)[0][0]
                    elif len(j3)!=0:
                        arr1 = len(v) + np.arange(4)[::-1] + len(P1) + 4*np.where(H[e2,4]==ei1)[0][0]
                        arr3 = len(v) + np.arange(4) + len(P1) + 4*j3[0]
                    if len(j2)!=0:
                        arr2 = len(v) + np.arange(4) + 4*j2[0]
                        arr4 = len(v) + np.arange(4)[::-1] + 4*np.where(H[e1,4]==ei4)[0][0]
                    elif len(j4)!=0:
                        arr2 = len(v) + np.arange(4)[::-1] + 4*np.where(H[e1,4]==ei2)[0][0]
                        arr4 = len(v) + np.arange(4) + 4*j4[0]
                w1 = np.where(v==v1)[0][0]
                w2 = np.where(v==v2)[0][0]
                w3 = np.where(v==v3)[0][0]
                w4 = np.where(v==v4)[0][0]
                newFace.append(np.r_[w1, arr1, w2, arr2, w3, arr3, w4, arr4])      
            newFace = np.array(newFace).tolist()        
            
        elif diag1_seg is not None and diag2_seg is not None:
            "two families of diagonal"
            P1, vlr1 = diag1_seg
            P2, vlr2 = diag2_seg   
            pass
        bm = Mesh()
        bm.make_mesh(newVer, newFace)
        return bm
    #--------------------------------------------------------------------------
    #                 Plot polylines: isolines + diagonals
    #-------------------------------------------------------------------------- 
    def get_reciprocal_diagram(self,direction=False,quadratic=True):
        #TODO
        # has problem for continus polyline edges
        "v1,v2 = self.get_1family_oriented_polyline(False,poly2=direction)"
        V = self.vertices
        H = self.halfedges
        def _diagram(v1,v2):
            val = []
            k = 0
            for i in range(len(v1)-1):
                a0 = v1[i]
                a = v1[i+1]
                i0 = np.where(H[:,0]==a0)[0]
                ii = np.where(H[H[:,4],0]==a)[0]
                m = np.intersect1d(i0,ii)
                print(m)
                k += 1
                
                if len(m)==0:
                    x = np.arange(k) / (k+1)
                    if quadratic:
                        val.append(x**2)
                    else:
                        "linear"
                        val.append(x)
                    k=0
                    
            x = np.arange(k) / (k+1)
            if quadratic:
                val.append(x**2)
            else:
                "linear"
                val.append(x)        
            pl = self.make_polyline_from_endpoints(V[v1],V[v2])  
            return pl, np.array(val) 
        if direction:
            v1,v2 = self.get_1family_oriented_polyline(False,True,False)  
            pl,data = _diagram(v1, v2)
        else:
            v1,v2 = self.get_1family_oriented_polyline(False,False,True)  
            pl,data = _diagram(v1, v2)
        return pl,data
    
    def get_quad_mesh_1family_isoline(self,diagnet=False,direction=True,edge=False):
        V = self.vertices
        v1,v2 = self.get_1family_oriented_polyline(diagnet,poly2=direction)
        pl = self.make_polyline_from_endpoints(V[v1],V[v2])
        if edge:
            "edge_data = np.zeros(self.E)"
            H = self.halfedges
            e,ib,eb = [],[],[]
            for i in range(len(v1)):
                a,b = v1[i],v2[i]
                j = np.where(H[:,0]==a)[0]
                k = np.where(H[H[:,4],0]==b)[0]
                m = np.intersect1d(j,k)[0]
                e.append(H[m,5])
                if H[m,1]==-1 or H[H[m,4],1]==-1:
                    ib.append(i)
                    eb.append(H[m,5])
            return pl, np.array(e),[np.array(ib),np.array(eb)]
        else:
            an,vec = V[v1], V[v2]-V[v1]
            return pl,an,vec 

    def get_1family_oriented_polyline(self,diagnet=False,poly2=True,demultiple=True):
        "still have problem for the demultiple and oriendted quad faces,bad for thickness"
        if diagnet:
            v,v1,v2,v3,v4 = self.rr_star_corner# in diagonal direction
        else:
            v,v1,v2,v3,v4 = self.ver_star_matrix.T
        num = len(v)
        if poly2:
            vl,vr = v2,v4
        else:
            vl,vr = v1,v3
        va,vb = vl,v
        for i in range(num):
            if v[i] not in np.r_[vl,v] or vr[i] not in np.r_[vl,v]:
                va = np.r_[va,v[i]]
                vb = np.r_[vb,vr[i]]
            else:
                i1 = np.where(va==v[i])[0]
                i2 = np.where(vb==vr[i])[0]
                if len(np.intersect1d(i1,i2))==0:
                    va = np.r_[va,v[i]]
                    vb = np.r_[vb,vr[i]]
        if demultiple:
            "remove multiple edges:"
            ind_del = []
            ck = []
            for i in range(len(va)):
                i1 = np.where(vb==va[i])[0]
                if len(i1) !=0:
                    for j in i1:
                        if j not in ck and va[j] == vb[i]:
                            ind_del.append(i)
                            print(i,va[i],vb[i],va[j],vb[j])
                            ck.append(i)
            if len(ind_del)!=0:
                ind = np.array(ind_del)   
                va = np.delete(va,ind)
                vb = np.delete(vb,ind)  
        return va,vb
    
    ## these two functions have same effects! above maybe faster.
    def get_1family_oriented_polyline0(self,diagnet=False,poly2=True,demultiple=True):
        "still have problem for the demultiple and oriendted quad faces,bad for thickness"
        if diagnet:
            v,v1,v2,v3,v4 = self.rr_star_corner# in diagonal direction
        else:
            v,v1,v2,v3,v4 = self.ver_star_matrix.T
        num = len(v)
        if poly2:
            vl,vr = v2,v4
        else:
            vl,vr = v1,v3
        ind_mul1, ind_mul2 = [], []
        for i in range(num):
            "check if [v,vr] has multiple indices in [vl,v], if not add here."
            i1 = np.where(vl==v[i])[0]
            if len(i1) !=0:
                if len(np.intersect1d(vr[i], v[i1])) !=0:
                    #j = np.where(v[i1]==vr[i])[0]
                    #k = i1[j]
                    ind_mul1.append(i)

            i2 = np.where(vl==vr[i])[0]
            if len(i2) !=0:
                if len(np.intersect1d(v[i], v[i2])) !=0:
                    ind_mul2.append(i)                    
        imul = np.unique(ind_mul1,ind_mul2)    
        k = np.setdiff1d(np.arange(num),imul)
        va = np.r_[vl,v[k]]
        vb = np.r_[v,vr[k]]
        
        if demultiple:
            "remove multiple edges:"
            ind_del = []
            ck = []
            for i in range(len(va)):
                i1 = np.where(vb==va[i])[0]
                if len(i1) !=0:
                    for j in i1:
                        if j not in ck and va[j] == vb[i]:
                            ind_del.append(i)
                            print(i,va[i],vb[i],va[j],vb[j])
                            ck.append(i)
            if len(ind_del)!=0:
                ind = np.array(ind_del)   
                va = np.delete(va,ind)
                vb = np.delete(vb,ind)  
                
        return va,vb  

    def get_2families_polyline_from_1closed_bdry(self,diag=False,interval=1,
                                                 inner=False,
                                                 is_poly=False):
        "along one bdry for a patch-shape; two bdry for a star-shape"
        if diag:
            v0,v1,v2,v3,v4 = self.rr_star_corner
        else:
            v0,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        v13,v24 = np.unique(np.r_[v1,v3]),np.unique(np.r_[v2,v4])
        
        "more general, one closed-boundary v"
        vb,_ = self.get_a_closed_boundary()
        
        H = self.halfedges
        "boundary are divided/filter into 2class based on v13 or v24 direction"
        vb1,vb2 = [],[]
        vfmly1,vfmly2 = [],[]
        i1l=i1r=i2l=i2r = np.array([],dtype=int)
        for i, v in enumerate(vb):
            if i%interval==0:
                if v in v13:
                    "filter to 1st polyline-bdry-vertices"
                    vb1.append(v)
                    if v in v1:
                        j = np.where(v1==v)[0]
                    elif v in v3:
                        j = np.where(v3==v)[0]
                    vx = v0[j]
                    if diag:
                        vvx = np.r_[v,vx]
                        vpl,_ = self.get_diagonal_polyline_from_2points(vvx,is_poly=False)
                    else:
                        e = np.intersect1d(np.where(H[:,0]==v)[0],np.where(H[H[:,4],0]==vx)[0])[0]
                        vpl,_ = self.get_polyline_from_an_edge(e,is_halfedge=True,
                                                               is_poly=False)
                    if inner:
                        vpl = vpl[1:-1]
                    vfmly1.append(vpl)
                    i1l = np.r_[i1l,vpl[:-1]]
                    i1r = np.r_[i1r,vpl[1:]]
                if v in v24:
                    "filter to 2nd polyline-bdry-vertices"
                    vb2.append(v)
                    if v in v2:
                        j = np.where(v2==v)[0]
                    elif v in v4:
                        j = np.where(v4==v)[0]
                    vx = v0[j]
                    if diag:
                        vvx = np.r_[v,vx]
                        vpl,_ = self.get_diagonal_polyline_from_2points(vvx,is_poly=False)
                    else:
                        e = np.intersect1d(np.where(H[:,0]==v)[0],np.where(H[H[:,4],0]==vx)[0])[0]
                        vpl,_ = self.get_polyline_from_an_edge(e,is_halfedge=True,
                                                               is_poly=False)
                    if inner:
                        vpl = vpl[1:-1]
                    vfmly2.append(vpl)
                    i2l = np.r_[i2l,vpl[:-1]]
                    i2r = np.r_[i2r,vpl[1:]]
        if is_poly:
            V = self.vertices
            pl1 = self.make_polyline_from_endpoints(V[i1l],V[i1r])
            pl2 = self.make_polyline_from_endpoints(V[i2l],V[i2r])
            return pl1,pl2
        return [vb1,vfmly1],[vb2,vfmly2]
        
    def make_polyline_from_endpoints(self,Vl,Vr):
        VV = np.vstack((Vl,Vr))
        v = np.arange(len(Vl))
        data = self.diagonal_cell_array(v)
        poly = Polyline(VV)
        poly.cell_array=data
        return poly   
    
    def diagonal_cell_array(self,v1):
        """represent polyline data
        suppose num=6
        a0 = [2,2,2,2,2,2]
        a1 = [0,1,2,3,4,5]
        a2 = [6,7,8,9,10,11] 
        cells = [2,0,6, 2,1,7,..., 2,5,11]
        """
        num = len(v1)
        c = np.repeat(2,num)
        "i = np.arange(num), j = num + i"
        i,j = np.arange(2*num).reshape(2,-1)
        cells = np.vstack((c,i,j)).T
        cells = np.ravel(cells)
        return cells
        
    ##--below----no use now, replaced by get_quad_mesh_1family_isoline--------
    def get_diagonal_edge(self,direction=True,poly=False):
        V = self.vertices
        if direction: # like cost.angle, in order
            f4 = self.rr_quadface
            v1,v2,v3,v4 = f4.T
        else:
            vi = self.quadface
            v1,v2,v3,v4 = vi[::4],vi[1::4],vi[2::4],vi[3::4]
        if poly:
            poly1 = self.make_polyline_from_endpoints(V[v1],V[v3])
            poly2 = self.make_polyline_from_endpoints(V[v2],V[v4])
            return poly1,poly2
        else:
            an1,D1 = (V[v1]+V[v3])*0.5, V[v3]-V[v1]
            an2,D2 = (V[v2]+V[v4])*0.5, V[v4]-V[v2]
            return an1,D1,an2,D2
    
    def get_diagonal_isoline(self,singularcase=False,direction=True,poly=False,vN=None):
        V = self.vertices
        if singularcase:
            from singularMesh import quadmesh_with_1singularity
            _,vlr,_ = quadmesh_with_1singularity(self)
            pl = self.make_polyline_from_endpoints(V[vlr[0]],V[vlr[1]])
            return pl
        else:
            v,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
            if direction:
                i1,i3 = vb, vd
            else:
                i1,i3 = va, vc
            if poly:
                pl1 = self.make_polyline_from_endpoints(V[i1],V[v])
                pl3 = self.make_polyline_from_endpoints(V[i3],V[v])
                if vN is not None:
                    VN = vN[v]
                    return pl1,pl3,VN,V[v],V[i1],V[i3]
                else:
                    VN = self.vertex_normals()[v]
                    if True:
                        "for diagonal net is A-net:"
                        #v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
                        N = np.cross(V[i1]-V[v],V[i3]-V[v])
                        N = N / np.linalg.norm(N,axis=1)[:,None]
                        dot = np.einsum('ij,ij->i',VN,N)
                        idd = np.where(dot<0)[0]
                        N[idd] = -N[idd]
                    else:
                        # "for controlnet is Gnet"
                        # e1 = (V[i1]-V[v])/ np.linalg.norm(V[i1]-V[v],axis=1)[:,None]
                        # e3 = (V[i3]-V[v])/ np.linalg.norm(V[i3]-V[v],axis=1)[:,None]
                        # N = - (e1 + e3)
                        N = VN                
                    return pl1,pl3,N,V[v],V[i1],V[i3]
                return pl1,pl3      
            else:
                an1,D1 = (V[v]+V[i1])*0.5, V[i1]-V[v]
                an2,D2 = (V[v]+V[i3])*0.5, V[i3]-V[v]
                return an1,D1,an2,D2
        
    def get_isoline_or_diagonal_isoline(self,diagnet=False,direction=False):
        "defalut: diagnal [a-c]"
        V = self.vertices
        v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        v,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
        if not diagnet:
            v1,v2,v3,v4,va,vb,vc,vd = va,vb,vc,vd,v1,v2,v3,v4
        if direction:
            i1,i3 = vb, vd # polyline direction
            i2,i4 = va, vc
        else:
            i1,i3 = va, vc # polyline direction
            i2,i4 = vb, vd
        pl1 = self.make_polyline_from_endpoints(V[i1],V[v])
        pl3 = self.make_polyline_from_endpoints(V[i3],V[v])

        VN = self.vertex_normals()[v]
        N = np.cross(V[v1]-V[v],V[v2]-V[v])
        N = N / np.linalg.norm(N,axis=1)[:,None]
        dot = np.einsum('ij,ij->i',VN,N)
        idd = np.where(dot<0)[0]
        N[idd] = -N[idd]
        return pl1,pl3,N,V[v],V[v1],V[v2],V[v3],V[v4],V[i1],V[i2],V[i3],V[i4]         
    
    def get_quad_mesh_isoline(self,singularcase=False,direction=True,poly=True,vN=None): # still used by menubar_agnet/show_1st_geodesic
        V = self.vertices
        if singularcase:
            from singularMesh import quadmesh_2familypolyline_with_1singularity
            M = quadmesh_2familypolyline_with_1singularity(self,direction)
            vl,vr = M[:,:-1].flatten(),M[:,1:].flatten()
            pl = self.make_polyline_from_endpoints(V[vl],V[vr])
            return pl
        else:
            v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
            if direction:
                i1,i3 = v2, v4
            else:
                i1,i3 = v1, v3
            if poly:
                pl1 = self.make_polyline_from_endpoints(V[i1],V[v])
                pl3 = self.make_polyline_from_endpoints(V[i3],V[v])
    
                if vN is not None:
                    VN = vN[v]
                    return pl1,pl3,VN,V[v],V[i1],V[i3]  
                else:
                    VN = self.vertex_normals()[v]
                    if True:
                        "for diagonal net is A-net:"
                        #v,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
                        N = np.cross(V[i1]-V[v],V[i3]-V[v])
                        N = N / np.linalg.norm(N,axis=1)[:,None]
                        dot = np.einsum('ij,ij->i',VN,N)
                        idd = np.where(dot<0)[0]
                        N[idd] = -N[idd]
                    else:
                        # "for controlnet is Gnet"
                        # e1 = (V[i1]-V[v])/ np.linalg.norm(V[i1]-V[v],axis=1)[:,None]
                        # e3 = (V[i3]-V[v])/ np.linalg.norm(V[i3]-V[v],axis=1)[:,None]
                        # N = - (e1 + e3)
                        N = VN                
                    return pl1,pl3,N,V[v],V[i1],V[i3]            
    
                return pl1,pl3      
            else:
                an1,D1 = (V[v]+V[i1])*0.5, V[i1]-V[v]
                an2,D2 = (V[v]+V[i3])*0.5, V[i3]-V[v]
                return an1,D1,an2,D2
    
    def get_quad_mesh_isoline_ver(self,singularcase=False,direction=True): # no use now
        if singularcase:
            from singularMesh import quadmesh_2familypolyline_with_1singularity
            M = quadmesh_2familypolyline_with_1singularity(self,direction)
            vl,vr = M[:,:-1].flatten(),M[:,1:].flatten()
            return vl,vr
        else:
            v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
            if direction:
                i1,i3 = v2, v4
            else:
                i1,i3 = v1, v3
            return np.r_[i1,v],np.r_[v,i3]
        
    ##--above----no use now, replaced by get_quad_mesh_1family_isoline--------
   
    def get_pieces_quad_edge(self,V):
        v1,v2,v3,v4 = np.arange(len(V)).reshape(-1,4,order='F').T
        poly1 = self.make_polyline_from_endpoints(V[v1],V[v2])
        poly2 = self.make_polyline_from_endpoints(V[v2],V[v3])
        poly3 = self.make_polyline_from_endpoints(V[v3],V[v4])
        poly4 = self.make_polyline_from_endpoints(V[v4],V[v1])
        return poly1,poly2,poly3,poly4

    def get_pieces_triangle_edge(self,V):
        v1,v2,v3 = np.arange(len(V)).reshape(-1,3,order='F').T
        poly1 = self.make_polyline_from_endpoints(V[v1],V[v2])
        poly2 = self.make_polyline_from_endpoints(V[v2],V[v3])
        poly3 = self.make_polyline_from_endpoints(V[v3],V[v1])
        return poly1,poly2,poly3
    
    def get_boundary_polyline(self,mesh,one=False):
        Vi = mesh.vertices
        vb = mesh.boundary_curves()
        if one:
            poly = Polyline(Vi[vb[0]],closed=True) # same as below
            #poly.refine() # round the corner
            return poly
        else:# for multiple
            vstart = vend = np.array([],dtype=int)
            for v in vb:
                vstart = np.r_[vstart,v] #v[:-1]
                vend = np.r_[vend, np.r_[v[1:],v[0]]] #v[1:]
            
            poly = self.make_polyline_from_endpoints(Vi[vstart],Vi[vend])
            return poly

    #---above: plot isolines + diagonals---------------------------------------





    def get_helical_spiral_curve(self,v,p,d,is_cut,sub=10,g=0,origin=False,helical=True,spiral=False,is_poly=False):
        """ v--ind_of_vs; p--pitch; d--t_domain; sub--polyline_num; g--gamma
        if helical:
            x(t) = x0 cost - y0 sint; 
            y(t) = x0 sint + y0 cost; 
            z(t) = z0 + pt
            any vertex V0(x0,y0,z0), r=sqrt(x0^2+y0^2), p is given by input
            set t = [-d,d], d=pi
        if spiral at center = (xc,yc,zc) = (0,0,zc)
            x(t) = e^(gt) * (x0 cost - y0 sint); 
            y(t) = e^(gt) * (x0 sint + y0 cost); 
            z(t) = e^(gt) * (z0 -  zc ) + zc
            special case origin==O
        """
        V = self.vertices
        num = len(v)
        arr = np.arange(int(sub/1))/(int(sub/1)-1)*d
        t = np.unique(np.r_[-arr[::-1],arr])

        if is_cut:
            it1 = np.where(t>=0)[0]
            it2 = np.where(t<0)[0]
            tm = self.get_triangulated_quad_mesh(loop=3) # subdivide 3 times
            ib = igl.boundary_loop(np.array(tm.faces_list()))
            bV = tm.vertices[ib]

        allx=ally=allz = np.array([])
        ns = np.array([],dtype=int)
        nvs, nvsi = np.array([],dtype=int), 0

        for i in range(num):
            x0,y0,z0 = V[v[i]]
            if helical:
                x = x0*np.cos(t)-y0*np.sin(t)
                y = x0*np.sin(t)+y0*np.cos(t)
                z = z0 + p*t
            elif spiral:
                "spiral parameter p = g/|c|==g"
                if origin:
                    "center at origin (0,0,0)"
                    x = (x0*np.cos(t)-y0*np.sin(t)) * np.e**(g*t)
                    y = (x0*np.sin(t)+y0*np.cos(t)) * np.e**(g*t)
                    z = np.e**(g*t) * z0
                # elif g == 0:
                #     print("Center is inifinity, should be helical case")
                #     x = x0*np.cos(t)-y0*np.sin(t)
                #     y = x0*np.sin(t)+y0*np.cos(t)
                #     z = z0 + p*t
                if g!=0:
                    x = (x0*np.cos(t)-y0*np.sin(t)) * np.e**(g*t)
                    y = (x0*np.sin(t)+y0*np.cos(t)) * np.e**(g*t)
                    zc = - p/g
                    z = np.e**(g*t) * (z0-zc) + zc
            if is_cut:
                """cut crv. outbound the surf.; two branches from t=0
                for each inn-reg-vertex, 
                d+(-) = crv.p+(-) to all boundary dist.
                choose the smallest one i+(-), 
                get the index [i-,i+]--> interval crv.p
                """
                Vi = np.c_[x,y,z]
                Vi1, Vi2 = Vi[it1], Vi[it2]
 
                def _min_dist(Pi,pos=True):
                    dmin = []
                    for j in range(len(Pi)):
                        dmin.append(np.min(np.linalg.norm(Pi[j]-bV,axis=1)))
                    idm = dmin.index(min(dmin))
                    if pos:
                        "choose pi from t=0, to idm"
                        arr = np.arange(idm)
                    else:
                        "choose pi from idm to t=0"
                        arr = np.arange(len(Pi))[idm:]
                    return Pi[arr]
                
                Vi1_sub = _min_dist(Vi1)
                Vi2_sub = _min_dist(Vi2,pos=False)
                x = np.r_[Vi2_sub[:,0], Vi1_sub[:,0]]
                y = np.r_[Vi2_sub[:,1], Vi1_sub[:,1]]
                z = np.r_[Vi2_sub[:,2], Vi1_sub[:,2]]
                allx = np.r_[allx,x]
                ally = np.r_[ally,y]
                allz = np.r_[allz,z]
                ns = np.r_[ns, len(x)-1]
                nvsi += len(x)
                nvs = np.r_[nvs, nvsi]
            else:
                allx = np.r_[allx,x]
                ally = np.r_[ally,y]
                allz = np.r_[allz,z]
                ns = np.r_[ns, len(x)-1] # len(t)=len(x)
                nvsi += len(x)
                nvs = np.r_[nvs, nvsi]
        allV = np.c_[allx,ally,allz]        
        if is_poly:
            polys = self.make_multiple_polylines_from_endpoints(allV, ns)
            return polys         
        return allV, nvs 

    # -------------------------------------------------------------------------
    #                        For ploting
    # -------------------------------------------------------------------------
    def index_ordered_tangents(self,num1): # to be used
        "e1,e2,e3,e4" # exist multiple(twice) edge_vector/length
        i1,i2,i3,i4 = [],[],[],[]
        for i in range(self.num_regular):
            v,v1,v2,v3,v4 = self.ver_regular_star[i,:]
            id0 = np.where(self.vi==v)[0]
            id1 = np.where(self.vj==v1)[0]
            id2 = np.where(self.vj==v2)[0]
            id3 = np.where(self.vj==v3)[0]
            id4 = np.where(self.vj==v4)[0]
            i1.append(np.intersect1d(id0,id1)[0])
            i2.append(np.intersect1d(id0,id2)[0])
            i3.append(np.intersect1d(id0,id3)[0])
            i4.append(np.intersect1d(id0,id4)[0])
        arr_e1 = self.columnnew(np.array(i1),num1,self.num_regular)
        arr_e2 = self.columnnew(np.array(i2),num1,self.num_regular)
        arr_e3 = self.columnnew(np.array(i3),num1,self.num_regular)
        arr_e4 = self.columnnew(np.array(i4),num1,self.num_regular)
        return arr_e1,arr_e2,arr_e3,arr_e4

    def get_v4_unit_edge(self,rregular=False):
        V = self.vertices
        if rregular:
            v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        else:
            ## v,v1,v2,v3,v4 = self.ver_regular_star.T
            v,v1,v2,v3,v4 = self.ver_star_matrix.T
        E1 = V[v1]-V[v]
        E2 = V[v2]-V[v]
        E3 = V[v3]-V[v]
        E4 = V[v4]-V[v]
        l1 = np.linalg.norm(E1, axis=1)
        l2 = np.linalg.norm(E2, axis=1)
        l3 = np.linalg.norm(E3, axis=1)
        l4 = np.linalg.norm(E4, axis=1)
        e1 = E1 / l1[:,None]
        e2 = E2 / l2[:,None]
        e3 = E3 / l3[:,None]
        e4 = E4 / l4[:,None]
        return l1,l2,l3,l4,e1,e2,e3,e4

    def get_v4_unit_tangents(self,plot=False,rregular=False):
        "only for valence 4, not depends on X"
        l1,l2,l3,l4,e1,e2,e3,e4 = self.get_v4_unit_edge(rregular)
        v = self.ver_star_matrix[:,0]
        anchor = self.vertices[v]
        t1 = (e1-e3)
        t2 = (e2-e4)
        lt1 = np.linalg.norm(t1,axis=1)
        lt2 = np.linalg.norm(t2,axis=1)
        ut1 = t1 / lt1[:,None]
        ut2 = t2 / lt2[:,None]
        angle = np.arccos(np.einsum('ij,ij->i', ut1, ut2))*180/np.pi
        if plot:
            a,b = (l1+l3)/5.0, (l2+l4)/5.0
            Vl,Vr = anchor+ut1*a[:,None], anchor-ut1*a[:,None]
            pl1 = self.make_polyline_from_endpoints(Vl,Vr)
            Vl,Vr = anchor+ut2*b[:,None], anchor-ut2*b[:,None] 
            pl2 = self.make_polyline_from_endpoints(Vl,Vr)
            return pl1,pl2
        return lt1,lt2,ut1,ut2,anchor,angle

    def get_v4_diag_unit_edge(self):
        V = self.vertices
        v,v1,v2,v3,v4 = self.rr_star_corner
        E1 = V[v1]-V[v]
        E2 = V[v2]-V[v]
        E3 = V[v3]-V[v]
        E4 = V[v4]-V[v]
        l1 = np.linalg.norm(E1, axis=1)
        l2 = np.linalg.norm(E2, axis=1)
        l3 = np.linalg.norm(E3, axis=1)
        l4 = np.linalg.norm(E4, axis=1)
        e1 = E1 / l1[:,None]
        e2 = E2 / l2[:,None]
        e3 = E3 / l3[:,None]
        e4 = E4 / l4[:,None]     
        return l1,l2,l3,l4,e1,e2,e3,e4

    def get_v4_diag_unit_tangents(self,plot=False):
        "only for valence 4, not depends on X"
        l1,l2,l3,l4,e1,e2,e3,e4 = self.get_v4_diag_unit_edge()
        v,_,_,_,_ = self.rr_star_corner
        anchor = self.vertices[v]
        t1 = (e1-e3)
        t2 = (e2-e4)
        lt1 = np.linalg.norm(t1,axis=1)
        lt2 = np.linalg.norm(t2,axis=1)
        ut1 = t1 / lt1[:,None]
        ut2 = t2 / lt2[:,None]
        angle = np.arccos(np.einsum('ij,ij->i', ut1, ut2))*180/np.pi
        if plot:
            a,b = (l1+l3)/6.0, (l2+l4)/6.0
            Vl,Vr = anchor+ut1*a[:,None], anchor-ut1*a[:,None]
            pl1 = self.make_polyline_from_endpoints(Vl,Vr)
            Vl,Vr = anchor+ut2*b[:,None], anchor-ut2*b[:,None] 
            pl2 = self.make_polyline_from_endpoints(Vl,Vr)
            return pl1,pl2           
        return lt1,lt2,ut1,ut2,anchor,angle
    
    def get_net_crossing_angle(self, ut1, ut2):
        "for orthogonal / isogonal case"
        cos1 = np.einsum('ij,ij->i', ut1,ut2)
        A = np.arccos(cos1)*180/np.pi
        print('----- net crossing angles : -------')
        print('max=', '%.2g' % np.max(A))
        print('mean=', '%.2g' % np.mean(A))
        print('min=', '%.2g' % np.min(A))
        print('----------------------------------')
        #return np.max(A),np.min(A),np.mean(A),np.median(A)



    def get_net_osculating_tangents(self,diagnet=False,show=False,printerr=False):
        "if diagnet: vabcd; else v1234"
        V = self.vertices
        if diagnet:
            v,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
            V0,V1,V2,V3,V4 = V[v],V[va],V[vb],V[vc],V[vd]
        else:
            v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
            V0,V1,V2,V3,V4 = V[v],V[v1],V[v2],V[v3],V[v4]
        ll1 = np.linalg.norm(V1-V0,axis=1)**2
        ll2 = np.linalg.norm(V2-V0,axis=1)**2
        ll3 = np.linalg.norm(V3-V0,axis=1)**2
        ll4 = np.linalg.norm(V4-V0,axis=1)**2
        t1 = (V3-V0)*ll1[:,None]-(V1-V0)*ll3[:,None]
        t2 = (V4-V0)*ll2[:,None]-(V2-V0)*ll4[:,None]
        lut1 = np.linalg.norm(t1,axis=1)
        lut2 = np.linalg.norm(t2,axis=1)
        ut1 = t1 / lut1[:,None]
        ut2 = t2 / lut2[:,None]
        if show:
            return V0,ut1,ut2
        if printerr:
            cos = np.einsum('ij,ij->i', ut1,ut2)
            A = np.arccos(cos)*180/np.pi
            return A
        return [ll1,ll2,ll3,ll4],[t1,t2],[lut1,ut1],[lut2,ut2]
    
    def get_osculating_tangent(self,diagnet=False,direction=False,show=False):
        "abcd if diag else 1234; bd/24 if direction else ac/13"
        V = self.vertices
        if diagnet:
            v,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
            V0 = V[v]
            if direction:
                Vl, Vr = V[vb],V[vd]
            else:
                Vl, Vr = V[va],V[vc]
        else:
            v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
            V0 = V[v]
            if direction:
                Vl, Vr = V[v2],V[v4]
            else:
                Vl, Vr = V[v1],V[v3]
        ll1 = np.linalg.norm(Vl-V0,axis=1)**2
        ll3 = np.linalg.norm(Vr-V0,axis=1)**2
        t = (Vr-V0)*ll1[:,None]-(Vl-V0)*ll3[:,None]
        lut = np.linalg.norm(t,axis=1)
        ut = t / lut[:,None]
        if show:
            return V0,ut       
        return ll1,ll3,t, lut,ut   

    def _offset_quadface_data(self,V4,V4N): # has problem with quad_dome.obj
        "generally V4==ver_regular, V4N==vertex_normal[ver_regular]"
        from changetopology import sub_mesh_valence4
        l = np.mean(self.edge_lengths())
        off_V = V4 + V4N*l
        newmesh = sub_mesh_valence4(self,self.ver_regular)
        return off_V, newmesh

    def get_both_support_structures_from_edges(self,N=None,dist=1,diagnet=True,demultiple=True,ss1=True,ss2=False):
        if N is None:
            N = self.vertex_normals()
        V = self.vertices
        N = V + N * np.mean(self.edge_lengths())*dist
        
        if demultiple:
            "still have multiple q 3 to demultip"
            if ss1:
                v1,v2 = self.get_1family_oriented_polyline(diagnet,poly2=False)
            elif ss2:
                v1,v2 = self.get_1family_oriented_polyline(diagnet,poly2=True)
            else:
                v11,v12 = self.get_1family_oriented_polyline(diagnet,poly2=False)
                v21,v22 = self.get_1family_oriented_polyline(diagnet,poly2=True)    
                v1,v2 = np.r_[v11,v21],np.r_[v12,v22]            
        else:
            if diagnet:
                "problem: have multiple edges"
                v,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
                v1,v2 = np.r_[va,vb,v,v],np.r_[v,v,vc,vd]
            else:
                v1,v2 = self.edge_vertices()

        # #print(len(v1),v1)        
        # "build non-multiple quad-vertices"
        # allv = np.unique(np.r_[v1,v2])     
        # ss_V = np.vstack((V[allv],N[allv]))
        # iv1,iv2 = [],[]
        # for i in range(len(v1)):
        #     i1 = np.where(allv==v1[i])[0][0]
        #     i2 = np.where(allv==v2[i])[0][0]
        #     iv1.append(i1)
        #     iv2.append(i2)
        # iv1,iv2 = np.array(iv1),np.array(iv2) 
        # #print(len(iv1),allv[iv1])  
        # #print(len(iv2),allv[iv2])    
        # ss_F = np.c_[iv1,iv2, iv2+len(allv),iv1+len(allv)]
        # #print(len(allv),iv1,iv2,ss_F)
        # ss = Mesh()
        # ss.make_mesh(ss_V,ss_F.tolist())
        # return ss
        "note: at singularity, one edge(normal) shares 3 quads, non-manifold"
        "so above commen only works for rectangel-shape patch, below for more general case"
        "but below has multiple-points ,need use Blender to remove"
        V1,V2,V3,V4 = V[v1],V[v2],N[v2],N[v1]
        ss_V = np.vstack((V1,V2,V3,V4))
        f1 = np.arange(len(v1))
        ss_F = np.c_[f1,f1+len(v1),f1+2*len(v1),f1+3*len(v1)]
        ss = Mesh()
        ss.make_mesh(ss_V,ss_F)
        return ss    
    
    def get_supportstructure(self,V4=None,V4N=None): # has problem!
        "support structure mesh of self vertex-normals (all vertices)"
        "each[v,vi,ni,n] form a quad mesh for SS"
        if V4 == None:
            V4 = self.vertices[self.ver_regular]
            V4N = self.vertex_normals()[self.ver_regular]
        off_V, newmesh = self._offset_quadface_data(V4,V4N)
        ss_V = np.vstack((V4,off_V))
        newv,newvj = newmesh.vertex_ring_vertices_iterators(order=True)
        flist = np.c_[newv,newvj,newvj+len(V4N),newv+len(V4N)] # problem:show_multiple_faces
        ss_F = flist.tolist()
        ss = Mesh()
        ss.make_mesh(ss_V,ss_F)
        return ss


    def get_offset(self,scale,thicken=False):
        """scale: pos or neg, to control the thickness of offset mesh
        hard to make sure thick_mesh have right orientation with both meshes,
        then separate into three meshes: self, offset_mesh, thick_mesh
        """
        V = self.vertices
        N = self.vertex_normals()
        H = self.halfedges
        flist = self.faces_list()
        l = np.mean(self.edge_lengths())
        VN = V + N * l * scale
        om = Mesh()
        if thicken: #Note: lack a quad for loop boundary
            allv = self.boundary_vertices()#self.get_all_boundary_vertices(order=True)
            VN = np.vstack((V[allv],VN[allv]))
            vb = self.boundary_curves(True) # suppose in order 
            fv1=fv2 = np.array([],dtype=int)
            for vv in vb:
                v = []
                for i in vv:
                    j = np.where(allv==i)[0][0]
                    v.append(j)
                e = np.intersect1d(np.where(H[:,0]==vv[-1])[0], np.where(H[:,1]==-1)[0])[0]    
                if H[H[e,2],0] ==vv[0]:
                    j = np.where(allv==vv[0])[0][0]
                    v.append(j)
                v1, v2 = v[:-1],v[1:]
                fv1 = np.r_[fv1,v1]
                fv2 = np.r_[fv2,v2]
            fv3,fv4 = fv2+len(allv),fv1+len(allv)
            flist = np.c_[fv1,fv2,fv3,fv4]
        om.make_mesh(VN, flist)
        return om
    
    def get_quadfac_offset(self,V4=None,V4N=None,dist=1):
        "offset mesh of self vertex-normals (all vertices)"
        "same geometry(facelist) as mesh, just subtitude v to n"
        if V4 == None:
            V4 = self.vertices[self.ver_regular]
            V4N = self.vertex_normals()[self.ver_regular] * dist
        off_V, newmesh = self._offset_quadface_data(V4,V4N * dist)
        off_F = newmesh.faces_list()
        offset = Mesh()
        offset.make_mesh(off_V,off_F)
        return offset

    def get_quadfac_offset_GI(self,V4=None,V4N=None):
        "Gauss Image based on vertex_normals of valence 4"
        if V4 == None:
            V4 = self.vertices[self.ver_regular]
            V4N = self.vertex_normals()[self.ver_regular]
        _, newmesh = self._offset_quadface_data(V4,V4N)
        gi_F = newmesh.faces_list()
        gi = Mesh()
        gi.make_mesh(V4N,gi_F)
        return gi

    def get_vertex_normal_GI(self):
        n = self.vertex_normals()
        f = self.faces_list()
        gi = Mesh()
        gi.make_mesh(n,f)
        return gi

    def get_face_normal_GI(self):
        verlist = self.face_normals()
        gi = self.get_barycenters_mesh(self,verlist)
        return gi

    def get_face_normal_facelist(self):
        H = self.halfedges
        star = self.rr_star
        forder = self.rr_quadface_order
        flist = [[] for i in range(len(star))]
        for i in range(len(star)):
            v,v1,v2,v3,v4 = self.rr_star[i,:]
            e=np.where(H[:,0]==v)[0]
            e1=np.where(H[H[:,4],0]==v1)[0]
            e2=np.where(H[H[:,4],0]==v2)[0]
            e3=np.where(H[H[:,4],0]==v3)[0]
            e4=np.where(H[H[:,4],0]==v4)[0]
            i1 = np.intersect1d(e,e1)
            i2 = np.intersect1d(e,e2)
            i3 = np.intersect1d(e,e3)
            i4 = np.intersect1d(e,e4)
            f1,f2,f3,f4 = H[i1,1],H[i2,1],H[i3,1],H[i4,1]
            if1 = np.where(forder==f1)[0]
            if2 = np.where(forder==f2)[0]
            if3 = np.where(forder==f3)[0]
            if4 = np.where(forder==f4)[0]
            if len(if1)!=0:
                flist[i].append(if1[0])
            if len(if2)!=0:
                flist[i].append(if2[0])
            if len(if3)!=0:
                flist[i].append(if3[0])
            if len(if4)!=0:
                flist[i].append(if4[0])
        return flist

    def get_barycenters_mesh(self,mesha,verlist=None):
        "barycenteras vertices, new faces"
        H = mesha.halfedges
        if verlist is None:
            verlist = mesha.face_barycenters()
        v,vj,lj = mesha.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
        iv = np.where(lj==4)[0]
        flist = np.array([])
        for i in iv:
            ei = np.where(H[:,0]==i)[0]
            if -1 in H[ei,1]:
                continue
            e1 = np.where(H[:,0]==i)[0][0]
            e2 = H[H[e1,3],4]
            e3 = H[H[e2,3],4]
            e4 = H[H[e3,3],4]
            f1,f2,f3,f4 = H[e1,1],H[e2,1],H[e3,1],H[e4,1]
            flist = np.r_[flist,f1,f2,f3,f4]
        flist = flist.reshape(-1,4).tolist()   
        
        new = Mesh()
        new.make_mesh(verlist,flist)
        return new

    def get_center_mesh(self,is_barycenter=True,mesh=None): 
        if mesh is None:
            mesh = self
        if is_barycenter:
            "new vertices from the average of each quad points"
            cm = mesh.get_barycenters_mesh(mesh)
        else:
            "suppose PQ: new vi from the intersection of two diagonal edges"
            "mesh should not be None"
            
            V = mesh.vertices
            fc =  mesh.face_barycenters()
            vlist = []
            flist = mesh.faces_list()
            for i in range(mesh.F):
                f = flist[i]
                if len(f)!=4:
                    vlist.append(fc[i])
                else:
                    """
                    quadABCD,O=A+a*AC=B+b*BD-->AB=a*Ac-b*BD;
                    B-A=[x,y,z]; C-A=[x1,y1,z1]; D-B=[x2,y2,z2];
                    only choose first & second coordinate:
                         a=(xy2-yx2)/(x1y2-x2y1)
                         b=(xy1-yx1)/(x1y2-x2y1)
                    """
                    A,B,C,D = V[f[0]],V[f[1]],V[f[2]],V[f[3]]
                    x,y,z = B-A
                    x1,y1,z1 = C-A
                    x2,y2,z2 = D-B
                    a=(x*y2-y*x2)/(x1*y2-x2*y1)
                    b=(x*y1-y*x1)/(x1*y2-x2*y1)
                    Ci = A + (C-A)*a #=B+b*(D-B)
                    vlist.append(Ci)
            verlist = np.array(vlist)
            cm = self.get_barycenters_mesh(mesh,verlist)
        return cm
            
    
    def get_supportstructures(self, Normals=None):
        """ from given normals anchored in vertices to get two directional SSs
        v(=orderNS) == ver_regular;
        starM == ver_regular_star(angle!=0) = orient_rings()
        """
        V = self.vertices
        ##starM = np.array(orient_rings(self)) # a little bit slow
        starM = self.ver_star_matrix
        v,v1,v2,v3,v4= starM.T
        if Normals == None:
            l = np.mean(self.edge_lengths())
            VN = V + self.vertex_normals()*l*1.5 # # need to check if  + or -

        ind1 = np.setdiff1d(v3,v)
        _,id1,_ = np.intersect1d(v3,ind1,return_indices=True)
        ss1va = np.r_[v1, v[id1]]
        ss1vb = np.r_[v, v3[id1]]
        ss1Va,ss1Vb = V[ss1va], V[ss1vb]
        ss1Na,ss1Nb = VN[ss1va], VN[ss1vb]

        ind2 = np.setdiff1d(v4,v)
        _,id2,_ = np.intersect1d(v4,ind2,return_indices=True)
        ss2va = np.r_[v2, v[id2]]
        ss2vb = np.r_[v, v4[id2]]
        ss2Va,ss2Vb = V[ss2va], V[ss2vb]
        ss2Na,ss2Nb = VN[ss2va], VN[ss2vb]

        ss1varry = np.vstack((ss1Va,ss1Vb, ss1Nb,ss1Na))
        ss2varry = np.vstack((ss2Va,ss2Vb, ss2Nb,ss2Na))

        def flist(num):
            f1 = np.arange(num)
            f2 = f1+num
            f3 = f2+num
            f4 = f3+num
            flist = (np.c_[f1,f2,f3,f4]).tolist()
            return flist

        f1list = flist(len(ss1va))
        f2list = flist(len(ss2va))

        SS1 = Mesh()
        SS2 = Mesh()
        SS1.make_mesh(ss1varry, f1list)
        SS2.make_mesh(ss2varry, f2list)

        return SS1, SS2


    ###----------------------------------------------------- 
    def get_isometry_error_from_checkerboard_diagonal(self,Vi,Wi,length=False,angle=False,area=False,dig_ctrl=False):
        """two isometry deformed mesh vi, wi
        using changes of a diagonal edge length of corresponding checkerboard faces
        to measure isometry error
        """
        iv0,iv1,iv2,iv3 = self.rr_quadface.T  
        el = self.mean_edge_length()
        
        d11 = np.linalg.norm(Vi[iv2]-Vi[iv0],axis=1)
        d12 = np.linalg.norm(Vi[iv3]-Vi[iv1],axis=1)
        d21 = np.linalg.norm(Wi[iv2]-Wi[iv0],axis=1)
        d22 = np.linalg.norm(Wi[iv3]-Wi[iv1],axis=1)  
        
        if length:
            "diagonal length of control face"    
            err = (np.abs(d11-d21)+np.abs(d12-d22)) / el
            print('\n Isometry (length) error = (np.abs(d11-d21)+np.abs(d12-d22)) / el : \n')

        elif angle:
            ag1 = np.einsum('ij,ij->i',Vi[iv2]-Vi[iv0],Vi[iv3]-Vi[iv1])
            ag2 = np.einsum('ij,ij->i',Wi[iv2]-Wi[iv0],Wi[iv3]-Wi[iv1])
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
            a1 = np.linalg.norm(np.cross(Vi[iv1]-Vi[iv0],Vi[iv3]-Vi[iv0]),axis=1)
            a2 = np.linalg.norm(np.cross(Wi[iv1]-Wi[iv0],Wi[iv3]-Wi[iv0]),axis=1)
            err = np.abs(a1-a2) / (el**2) / 2
            print('\n Isometry (area) error : \n')

        elif dig_ctrl:
            "diagonal length of checkerboard quad"
            d11 = np.linalg.norm((Vi[iv1]+Vi[iv0])/2-(Vi[iv2]+Vi[iv3])/2,axis=1)
            d12 = np.linalg.norm((Vi[iv2]+Vi[iv1])/2-(Vi[iv3]+Vi[iv0])/2,axis=1)
            d21 = np.linalg.norm((Wi[iv1]+Wi[iv0])/2-(Wi[iv2]+Wi[iv3])/2,axis=1)        
            d22 = np.linalg.norm((Wi[iv2]+Wi[iv1])/2-(Wi[iv3]+Wi[iv0])/2,axis=1)
            err = (np.abs(d11-d21)+np.abs(d12-d22))/2 / el

        print('   max = ','%.3g' % np.max(err),'\n')
        
        P1,P2,P3,P4 = (Vi[iv0]+Vi[iv1])/2,(Vi[iv1]+Vi[iv2])/2,(Vi[iv2]+Vi[iv3])/2,(Vi[iv3]+Vi[iv0])/2
        PP1,PP2,PP3,PP4 = (Wi[iv0]+Wi[iv1])/2,(Wi[iv1]+Wi[iv2])/2,(Wi[iv2]+Wi[iv3])/2,(Wi[iv3]+Wi[iv0])/2
        ck1 = self.make_quad_mesh_pieces(P1,P2,P3,P4)
        ck2 = self.make_quad_mesh_pieces(PP1,PP2,PP3,PP4)
        return ck1,ck2,err
    
    # -------------------------------------------------------------------------
    #                     Discrete Differential Geometry
    # -------------------------------------------------------------------------
    def get_curvature(self,mesh,order=None):
        "from davide' eigen of shape-operator"
        k1,k2, D1, D2 = mesh.principal_curvatures(True, use_sine=True)
        K = mesh.gaussian_curvature()
        H = mesh.mean_curvature()
        eps = np.finfo(float).eps
        if order is not None:
            K,H = K[order],H[order]
            D1,D2 = D1[order],D2[order]
            k1,k2 = k1[order],k2[order]
        ratio = [np.min(k1/(k2+eps)),np.mean(k1/(k2+eps)),np.max(k1/(k2+eps))]
        return ratio,[np.min(K),np.mean(K),np.max(K)],[np.min(H),np.mean(H),np.max(H)],D1,D2

    def get_curvature_libigl(self,mesh,evalue=False):# from libigl triangulated-mesh: 
        "via quadric fitting (Panozzo, 2010)"
        trv = mesh.vertices
        trf,_ = mesh.face_triangles()
        D1,D2,k1,k2 = igl.principal_curvature(trv,trf)  
        if evalue:
            K,H = k1*k2, (k1+k2)/2
            return trv,[np.min(K),np.mean(K),np.max(K)],[np.min(H),np.mean(H),np.max(H)],D1,D2
        return trv, k2, k1, D2, D1 #rearange in [min,max]

    def get_curvature_ratio(self,mesh):
        _,_,lj = mesh.vertex_ring_vertices_iterators(return_lengths=True)
        order = np.where(lj==4)[0]
        k1,k2, _,_ = mesh.principal_curvatures(True, use_sine=True)
        if order is not None:
            k1,k2 = k1[order],k2[order]
        eps = np.finfo(float).eps
        ratio = [np.min(k1/(k2+eps)),np.mean(k1/(k2+eps)),np.max(k1/(k2+eps))]
        return ratio
    
    def principal_curvatures_quadface(self,Vi,Ni,Gauss=False,Mean=False,evalue=False):
        """ based on each quadface,from E,F,G,e,f,g Solve eigenproblem.
        Xu:= v3-v1; Xv:= v4-v2; 
        E:= (v3-v1)^2
        F:= (v3-v1)*(v4-v2)
        G:= (v4-v2)^2
        e:= -(N3-N1)*(v3-v1)
        f:= -(N3-N1)*(v4-v2)=-(N4-N2)*(v3-v1)
        g:= -(N4-N2)*(v4-v2)
        --> K=(eg-f^2)/(EG-F^2); H=(eG+Eg-2fF)/[2(EG-F^2)]
        --> k1 = H+sqrt(H^2-K); k2 = H-sqrt(H^2-K)
        or eigen(II/I) --> k1,k2, D1,D2
        """
        #Vi = mesh.vertices
        v1,v2,v3,v4  = self.rr_quadface.T
        an = (Vi[v3]+Vi[v1]+Vi[v4]+Vi[v2])/4
        Xu, Xv = Vi[v3]-Vi[v1], Vi[v4]-Vi[v2]
        num = len(v1)
        E,F,G = self.get_first_fundamental_form(Vi)
        e,f,g = self.get_second_fundamental_form(Vi,Ni)
        if Gauss and not evalue:
            "direct compute: K=(e*g-f**2) / (E*G-F**2)"
            K = (e*g-f**2) / (E*G-F**2)
            
            inn,_ = self.get_rr_quadface_boundaryquad_index()
            dmin = round(np.min(K[inn]),5)
            dmean = round(np.mean(K[inn]),5)
            dmax = round(np.max(K[inn]),5)
            print('inn-K[min,mean,max]=',dmin,dmean,dmax)

            dmin = round(np.min(K),5)
            dmean = round(np.mean(K),5)
            dmax = round(np.max(K),5)
            print('K[min,mean,max]=',dmin,dmean,dmax)
            
            return K
        elif Mean and not evalue:
            "H = (e*G+E*g-2*f*F)/(E*G-F**2)/2"
            H = (e*G+E*g-2*f*F)/(E*G-F**2)/2
            
            inn,_ = self.get_rr_quadface_boundaryquad_index()
            dmin = round(np.min(H[inn]),5)
            dmean = round(np.mean(H[inn]),5)
            dmax = round(np.max(H[inn]),5)
            print('K[min,mean,max]=',dmin,dmean,dmax)
            
            return H
        k1,k2 = [],[]
        d1,d2 = np.array([]),np.array([])
        from scipy.linalg import eigh
        for i in range(num):
            I = np.array([[E[i],F[i]],[F[i],G[i]]])
            II = np.array([[e[i],f[i]],[f[i],g[i]]])
            vals, vecs = eigh(II,I) 
            a = np.argmin(vals) # should 0 or 1
            k1.append(vals[a])
            k2.append(vals[a-1])
            vmin,vmax = vecs[:,a], vecs[:,a-1]
            du1,dv1 = vmin
            du2,dv2 = vmax
            d1 = np.r_[d1, du1*Xu[i] + dv1*Xv[i]]
            d2 = np.r_[d2, du2*Xu[i] + dv2*Xv[i]]
        k1,k2 = np.array(k1),np.array(k2)
        d1 = d1.reshape(-1,3)
        d2 = d2.reshape(-1,3)
        if Gauss and evalue: # after check, same as above, err~=e-17
            print(np.max(np.abs(k1*k2-(e*g-f**2) / (E*G-F**2))))
            return k1*k2
        elif Mean and evalue: # after check, same as above, err~=e-16
            print(np.max(np.abs((k1+k2)/2-(e*G+E*g-2*f*F)/(E*G-F**2)/2)))
            return (k1+k2)/2
        if evalue:
            K,H = k1*k2, (k1+k2)/2
            
            inn,_ = self.get_rr_quadface_boundaryquad_index()
            kmin = round(np.min(K[inn]),5)
            kmean = round(np.mean(K[inn]),5)
            kmax = round(np.max(K[inn]),5)
            hmin = round(np.min(H[inn]),5)
            hmean = round(np.mean(H[inn]),5)
            hmax = round(np.max(H[inn]),5)
            print('\n Gauss(inn) K :[min,mean,max] = ','{:.5f} | {:.5f} | {:.5f}'.format(kmin,kmean,kmax))
            print('\n Mean(inn)  H :[min,mean,max] = ','{:.5f} | {:.5f} | {:.5f}'.format(hmin,hmean,hmax))
            
            return an,[np.min(K),np.mean(K),np.max(K)],[np.min(H),np.mean(H),np.max(H)],d1,d2
        return k1,k2,d1,d2
    
    def gaussian_curvature_quadface(self,Vi,Ni):
        K = self.principal_curvatures_quadface(Vi,Ni)
        return K[0]*K[1]

    def mean_curvature_quadface_quadface(self,Vi,Ni):
        K = self.principal_curvatures_quadface(Vi,Ni)
        H =  0.5*(K[0] + K[1])
        return H
 
                
    def get_snet_diag_curvature(self,is_k=False,is_kg=False,is_kn=False):
        """ different from below face_based, here is vertex_based
        the Snet is in diagonal direction. Refer [Snet-paper]-Eq(7,9)
        osculating tangents t1, t2 of three planar points
        t1 = (v1-v)*l3^2-(v3-v)*l1^2 # coordinats
        t2 = (v2-v)*l4^2-(v4-v)*l2^2 # coordinats
        """
        Vi, Ni = self.vertices, -self.vertex_normals()
        # get Snet-diagnet radius & normals:
        s0,s1,s2,s3,s4 = self.get_vs_diagonal_v()
        S0,S1,S2,S3,S4 = Vi[s0],Vi[s1],Vi[s2],Vi[s3],Vi[s4]
        from conicSection import interpolate_sphere
        _,radius,coeff,Nv4 = interpolate_sphere(S0,S1,S2,S3,S4)
        if True:
            "make orientation of the normals"
            A,B,C,D,E = coeff.reshape(5,-1)
            _,_,_,Nv4,_ = self.orient(S0,A,B,C,D,Nv4)
        # get Snet diagnet normals:
        un = Nv4 / np.linalg.norm(Nv4,axis=1)[:,None]      
        v0,va,vb,vc,vd = self.rr_star_corner# in diagonal direction
        Ni[v0] = un
        inn,_ = self.get_rr_vs_bounary()
        v,v1,v2,v3,v4 = v0[inn],va[inn],vb[inn],vc[inn],vd[inn]
        def _vector(Vi):
            "three points-->osculating plane-->tangent"
            V0,V1,V2,V3,V4 = Vi[v],Vi[v1],Vi[v2],Vi[v3],Vi[v4]
            l1 = np.linalg.norm(V1-V0,axis=1)
            l2 = np.linalg.norm(V2-V0,axis=1)
            l3 = np.linalg.norm(V3-V0,axis=1)
            l4 = np.linalg.norm(V4-V0,axis=1)
            t1 = (V1-V0)*(l3**2)[:,None] - (V3-V0)*(l1**2)[:,None]
            t1 = t1 / np.linalg.norm(t1,axis=1)[:,None]
            t2 = (V2-V0)*(l4**2)[:,None] - (V4-V0)*(l2**2)[:,None]
            t2 = t2 / np.linalg.norm(t2,axis=1)[:,None]
            b1 = np.cross(V1-V0,V3-V0)
            b2 = np.cross(V2-V0,V4-V0)
            n1 = np.cross(b1,t1) / np.linalg.norm(np.cross(b1,t1),axis=1)[:,None]
            n2 = np.cross(b2,t2) / np.linalg.norm(np.cross(b2,t2),axis=1)[:,None]
            r1=np.linalg.norm(V1-V0,axis=1)**2/np.einsum('ij,ij->',V1-V0,n1)/2
            r2=np.linalg.norm(V2-V0,axis=1)**2/np.einsum('ij,ij->',V2-V0,n2)/2
            return t1,t2,b1,b2,n1,n2,r1,r2

        t1,t2,b1,b2,n1,n2,r1,r2 = _vector(Vi)
        # tn1,tn2,_,_,_,_,_,_ = _vector(Ni)
        # s1 = np.einsum('ij,ij->i', t1, t2)
        # s2 = np.einsum('ij,ij->i', tn1,tn2)
        if False:
            "simplified representation of k1/k2"
            ratio = np.sqrt((1-s1)*(1+s2)/((1+s1)*(1-s2)))
            ratio = 1/ratio # to show k2/k1
        else:
            "from the numerical result, max,min near mean,this better"
            yit1,yit2 = t1+t2, t1-t2
            # cit1,cit2 = tn1+tn2, tn1-tn2
            # dn = np.cross(cit1/(np.linalg.norm(cit1,axis=0))**2, cit2)
            # dn = np.einsum('ij,ij->i', dn, Ni[v])
            # dv = np.cross(yit1/(np.linalg.norm(yit1,axis=0))**2, yit2)
            # dv = np.einsum('ij,ij->i', dv, Ni[v])
            #eps = np.finfo(float).eps
            #ratio = dv / (dn+eps)  # get k1/k2
            #ratio = dn / (dv+eps)  # to show k2/k1
            if is_k:
                radius = radius[inn]#self.get_snet_radius(Vi[v],Vi[v1],Vi[v2],Vi[v3],Vi[v4])
                kn = 1/radius
                #k1 = dv*(2*kn)/(dv*(1+s1)+dn*(1-s1))
                #k2 = dn*(2*kn)/(dv*(1+s1)+dn*(1-s1))
                #print(np.min(k1),np.max(k1))  
                #print(np.min(k2),np.max(k2))  
                pcd1 = yit1 / np.linalg.norm(yit1,axis=1)[:,None]
                pcd2 = yit2 / np.linalg.norm(yit2,axis=1)[:,None]
                cos2 = np.einsum('ij,ij->i',t1,pcd1)**2
                sin2 = 1-cos2
                k1,k2 = kn/cos2/2, kn/sin2/2
                #print(np.min(k1),np.max(k1))  
                #print(np.min(k2),np.max(k2))  
                ratio = sin2/cos2 #k1/k2
                print(np.min(ratio),np.mean(ratio),np.max(ratio))
                return k1,k2,pcd1,pcd2
        if is_kg:
            kg1 = r1 * np.linalg.norm(np.cross(Ni[v],n1),axis=1)
            kg2 = r2 * np.linalg.norm(np.cross(Ni[v],n2),axis=1)
        if is_kn:
            kn1 = r1 * np.einsum('ij,ij->',Ni[v],n1)
            kn2 = r2 * np.einsum('ij,ij->',Ni[v],n2)
        return kg1,kg2,kn1,kn2   

    def get_focal_from_curvature(self,is_k1=False,is_k2=False,is_sub=True):      
        """
        given principal direction t, principal curvature k1, k2
              estimate the curve normal of each line of curvature
              along each polyline: pcd1, pcd, pcd2
              b1:= pcd x pcd1
              b2:= pcd x pcd2
              b:= (b1+b2 ) /||
              n:= b x t
              ci:= vi + n * 1/k1
        """
        v0,v1,v2,v3,v4 = self.ver_star_matrix.T

        if True:
            "Compute PC from [P]-Snet"
            K = self.principal_curvatures()
            pc1, pc2 = K[0], K[1]
            pcd1, pcd2 = K[2], K[3]
            k1,k2,d1,d2 = self.get_snet_diag_curvature(is_k=True)
            vind,_,_,_,_ = self.rr_star_corner
            inn,_ = self.get_rr_vs_bounary()
            ind = vind[inn]
            pc1[ind],pc2[ind] = k1,k2
            pcd1[ind],pcd2[ind] = d1,d2 
        else:
            #"libigl: the computed curvatures are more accurate"
            _,pc1,pc2,pcd1, pcd2 = self.get_curvature_libigl(self)
            
            k1, k2 = pcd1[v0], pcd2[v0]
            _,_,ut1,ut2,an,_ = self.get_v4_unit_tangents()
            dot1, dot2 = np.einsum('ij,ij->i',ut1,k1),np.einsum('ij,ij->i',ut2,k2)
            i, j = np.where(dot1<0)[0], np.where(dot2<0)[0]
            pcd1[v0[i]], pcd2[v0[j]] = -k1[i], -k2[j]
        
        # print(np.min(pc1),np.max(pc1))  
        # print(np.min(pc2),np.max(pc2))  

        N = self.vertex_normals()
        if is_k1:
            t0,t2,t4 = pcd1[v0],pcd1[v2],pcd1[v4] # NOTE: need to check if pcd1[v2,v4] or pcd1[v1,v3]
            b = np.cross(t2,t0)+np.cross(t0,t4)
            b = b / np.linalg.norm(b,axis=1)[:,None]
            n = np.cross(b, t0)
            N[v0] = n
            ci = self.vertices - N / (pc1+np.finfo(float).eps)[:,None] # NOTE: need to check if +/-
        elif is_k2:
            t0,t1,t3 = pcd2[v0],pcd2[v1],pcd2[v3] # NOTE: need to check if pcd2[v2,v4] or pcd2[v1,v3]
            b = np.cross(t1,t0)+np.cross(t0,t3)
            b = b / np.linalg.norm(b,axis=1)[:,None]
            n = np.cross(b, t0)
            N[v0] = n
            ci = self.vertices - N / (pc2+np.finfo(float).eps)[:,None] # NOTE: need to check if +/-
        # if is_mesh:
        #     fm = Mesh()
        #     fm.make_mesh(ci,self.faces_list())
        #     if is_sub:
        #         "if sub once, the second inner boundary vertices happens bad"
        #         from huilab.huimesh.changetopology import sub_mesh_valence4
        #         fm = sub_mesh_valence4(fm,v0)
        #         "get the another sub-mesh without the second inn-boundary"
        #         ind = fm.inner_vertices()
        #         fm = sub_mesh_valence4(fm,ind)
        #     return fm
        # if is_sub:
        #     ci = ci[v0]
        #     an = self.vertices[v0]
        #     ni = -n
        fm = Mesh()
        fm.make_mesh(ci,self.faces_list())
        "if sub once, the second inner boundary vertices happens bad"
        from changetopology import sub_mesh_valence4
        fm = sub_mesh_valence4(fm,v0)
        "get the another sub-mesh without the second inn-boundary"
        ind = fm.inner_vertices()
        fm = sub_mesh_valence4(fm,ind)
        ci = ci[v0[ind]]
        an = self.vertices[v0[ind]]
        ni = -n[ind]
        t0 = t0[ind]
        return [fm,[an,t0,ni,ci]]  
                

    def get_distortion_from_revolution(self,length,angle,is_T=False): 
        m1v = self.vertices 
        _,rm = self.get_revolution(is_T=is_T)
        patch = self.patch_matrix.T if is_T else self.patch_matrix
        ind = []
        arr = patch.flatten()
        for i in range(self.V):
            ind.append(np.where(arr==i)[0][0])
        m2v = rm.vertices[np.array(ind)]
        ck1,ck2,err = self.get_isometry_error_from_checkerboard_diagonal(m1v,m2v,length,angle)
        return ck2,err
    
    def get_revolution(self,is_T=False): 
        """corresponding rotational mesh of a patch
        v1,v,v3 from up to down; v2,v,v4 from left to right 
             v1
        v2   v   v4
             v3
        """
        patch = self.patch_matrix.T if is_T else self.patch_matrix
        nrow,ncol = patch[1:-1,1:-1].shape
        ver = self.vertices

        vv1 = patch[:-1,1:].flatten()
        vv = patch[1:,1:].flatten()
        vv2 = patch[1:,:-1].flatten()
        VV1,VV,VV2 = ver[vv1],ver[vv],ver[vv2]
        ll = np.linalg.norm(VV1-VV,axis=1)
        e1 = (VV1-VV)/ll[:,None]
        e2 = (VV2-VV)/np.linalg.norm(VV2-VV,axis=1)[:,None]
        theta = np.arccos(np.einsum('ij,ij->i',e1,e2))
        the = np.mean(theta.reshape(-1,ncol+1),axis=1)
        
        Va, Vc = ver[patch[:-1,:].flatten()], ver[patch[1:,:].flatten()]
        lvertical = np.linalg.norm(Va-Vc,axis=1).reshape(-1,ncol+2)
        lv_row = np.mean(lvertical,axis=1) * np.sin(the)

        Vb, Vd = ver[patch[:,:-1].flatten()], ver[patch[:,1:].flatten()]
        lhorizon = np.linalg.norm(Vb-Vd,axis=1).reshape(-1,ncol+1)
        lh_row = np.mean(lhorizon,axis=1)

        "from optimized revoluted mesh to reconstruct AIAP revolution mesh"
        "the horizontal segments should be equal"
        "the horizontal geodesic parallel satisfy"
        "vertices\faces: from the first row to last"
        vM = np.arange(self.V).reshape(-1,ncol+2)
        newM = vM
        r1 = newM[:-1,:-1].flatten()
        r2 = newM[1:,:-1].flatten()
        r3 = newM[1:,1:].flatten()
        r4 = newM[:-1,1:].flatten()
        facelist = (np.vstack((r1,r2,r3,r4)).T).tolist()

        def points(center,r,segNum):
            "center=[0,0,z],segNum is unique"
            x = center[0] + r*np.cos(2*np.pi*np.arange(segNum+1)/float(segNum))
            y = center[1] + r*np.sin(2*np.pi*np.arange(segNum+1)/float(segNum))
            z = center[2]
            P = np.vstack((x, y, z*np.ones(segNum+1))).T
            return P

        radius = lh_row/float(2) / np.sin(np.pi / (ncol+1))
        height = np.sqrt(np.square(lv_row)-np.square(radius[:-1]-radius[1:]))
        
        POINTS = np.array([0,0,0])
        z = 0
        for i in range(nrow+2):
            if i==0:
                h=0
            else:
                h = height[i-1]
            z += h
            center=[0,0,z]
            Pi = points(center,radius[i],ncol+1)
            POINTS = np.vstack((POINTS,Pi))
            
        xmin,xmax = self.bounding_box()[0]
        an = np.array([xmax-xmin,0,0])
        POINTS += an
        
        rmesh = Mesh()
        rmesh.make_mesh(POINTS[1:,:], facelist)
        return POINTS[1:,:],rmesh  


    def get_curvature_disk_polylines(self,is_quad=False,is_tri=False,is_sub=False):
        "use mayavi.modules.isosurface curvature: k1,k2,K,H"
        if is_tri:
            Tri, f = self.face_triangles()
            trv,trf = self.vertices, Tri
            if is_sub:
                trv,trf = igl.upsample(trv,trf,number_of_subdivs=2)
            tri = Mesh()
            tri.make_mesh(trv,trf)
            mesh = tri
            _,_,k1,k2 = igl.principal_curvature(trv,trf)  
        if is_sub:
            self.catmull_clark() # tend to stuck, has problem for 2nd click
        if is_quad:
            mesh = self
            k1,k2, D1, D2 = self.principal_curvatures(True, use_sine=True)
        K, H = k1*k2, (k1+k2)/2
        # print(np.min(k1),np.max(k1))
        # print(np.min(k2),np.max(k2))
        # print(np.min(K),np.max(K))
        # print(np.min(H),np.max(H))
        def get_poly(k,num=8):
            # result not right no use now
            isov,isoe = igl.isolines(trv,trf,k,n=num)
            vv = np.vstack((isov[isoe[:,0]],isov[isoe[:,1]]))
            data = self.diagonal_cell_array(isoe[:,0])
            poly = Polyline(vv)
            poly.cell_array=data 
            return poly
        #polyk1,polyk2,polyK,polyH = get_poly(k1),get_poly(k2),get_poly(K),get_poly(H)
        return mesh,k1,k2,K,H
           
    def get_second_fundamental_form(self,Vi,Ni,quadface_based=True,rr=True): 
        ##Vi = mesh.vertices
        #Ni = mesh.vertex_normals()
        if quadface_based:
            """
               4    3    
               1    2                  
            e = -Nu*Xu = -(N3-N1)*(v3-v1)
            g = -Nv*Xv = -(N4-N2)*(v4-v2)
            f = -Nv*Xu = -(N3-N1)*(v4-v2)=-(N4-N2)*(v3-v1)
            E = Xu*Xu =(v3-v1)^2
            G = Xv*Xv =(v4-v2)^2
            F = Xu*Xv =(v3-v1)*(v4-v2)
            """            
            v1,v2,v3,v4  = self.rr_quadface.T
            V1,V2,V3,V4 = Vi[v1],Vi[v2],Vi[v3],Vi[v4]
            N1,N2,N3,N4 = Ni[v1],Ni[v2],Ni[v3],Ni[v4] 
            e = -np.einsum('ij,ij->i',N3-N1, V3-V1) 
            g = -np.einsum('ij,ij->i',N4-N2, V4-V2) 
            f = -np.einsum('ij,ij->i',N3-N1, V4-V2)/2
            f -= np.einsum('ij,ij->i',N4-N2, V3-V1)/2          
        else:
            """ mesh = self or refermesh
               a    1    d
               2    v0   4
               b    3    c                
            e = -Nu*Xu = N*Xuu = N*(vb+vd-2v)
            d = -Nv*Xv = N*Xvv = N*(va+vc-2v)
            f = -Nv*Xu = -Nu*Xv = N*Xuv 
              = -1/4[(N1-N4)*(vd-v0)+(N2-N3)*(v0-vb)+(N1-N2)*(va-v0)+(N4-N3)*(v0-vc)]
            """             
            if rr:
                star = self.rr_star
                star = star[self.ind_rr_star_v4f4]
                v,v1,v2,v3,v4 = star.T
                _,va,vb,vc,vd = self.rr_star_corner
            V0 = Vi[v]
            N0,N1,N2,N3,N4 = Ni[v],Ni[v1],Ni[v2],Ni[v3],Ni[v4]
            Va,Vb,Vc,Vd = Vi[va],Vi[vb],Vi[vc],Vi[vd]
            e = np.einsum('ij,ij->i',N0,Vb+Vd-2*V0)
            g = np.einsum('ij,ij->i',N0,Va+Vc-2*V0)
            f = -np.einsum('ij,ij->i',N1-N4,Vd-V0)/4
            f -= np.einsum('ij,ij->i',N2-N3,V0-Vb)/4
            f -= np.einsum('ij,ij->i',N1-N2,Va-V0)/4
            f -= np.einsum('ij,ij->i',N4-N3,V0-Vc)/4
        return e,f,g

    def get_first_fundamental_form(self,Vi,quadface_based=True,rr=True): 
        #Vi = mesh.vertices
        if quadface_based:
            """
               4    3    
               1    2                  
            E = Xu*Xu =(v3-v1)^2
            G = Xv*Xv =(v4-v2)^2
            F = Xu*Xv =(v3-v1)*(v4-v2)
            """            
            v1,v2,v3,v4  = self.rr_quadface.T
            V1,V2,V3,V4 = Vi[v1],Vi[v2],Vi[v3],Vi[v4]         
            E = np.linalg.norm(V3-V1,axis=1)**2
            G = np.linalg.norm(V4-V2,axis=1)**2
            F = np.einsum('ij,ij->i',V3-V1, V4-V2)          
        else:
            """ mesh = self or refermesh
               a    1    d
               2    v0   4
               b    3    c                
            E = (vd-vb)**2
            G = (va-vc)**2
            F = (vd-vb)*(va-vc)
            """             
            if rr:
                # star = self.rr_star
                # star = star[self.ind_rr_star_v4f4]
                # v,v1,v2,v3,v4 = star.T
                _,va,vb,vc,vd = self.rr_star_corner
            Va,Vb,Vc,Vd = Vi[va],Vi[vb],Vi[vc],Vi[vd]
            E = np.linalg.norm(Vd-Vb,axis=1)**2
            G = np.linalg.norm(Va-Vc,axis=1)**2
            F = np.einsum('ij,ij->i',Vd-Vb, Va-Vc)
        return E,F,G

#------------------------------------------------------------------------------    
    def get_quadface_face_normals(self,Vi,rr=True):
        ##Vi = mesh.vertices
        ##Ni = mesh.vertex_normals()
        v1,v2,v3,v4  = self.rr_quadface.T
        V1,V2,V3,V4 = Vi[v1],Vi[v2],Vi[v3],Vi[v4]  
        norm = np.linalg.norm(np.cross(V3-V1,V4-V2),axis=1)
        N = np.cross(V3-V1,V4-V2) / norm[:,None]
        #norm2 = np.linalg.norm(Ni[v1]+Ni[v2]+Ni[v3]+Ni[v4],axis=1)
        #a,b = np.sqrt(norm), np.sqrt(norm2)
        return N.flatten('F'),norm,np.sqrt(norm)#a,norm2,b

    def get_mesh_interpolated_sphere(self):
        centers,radius,_,_ = sphere_equation(self.vertices,self.ver_regular,self.ringlist)
        C,r = np.mean(centers,axis=0), np.mean(radius)
        return C,r
    
    def get_vs_interpolated_sphere(self,v):
        C,r,_,_ = sphere_equation(self.vertices,v,self.ringlist)
        all_vi = np.array([],dtype=int)
        for i in v:
            all_vi = np.r_[all_vi,np.array(self.ringlist[i])]
        Vneib = self.vertices[all_vi]
        return C,r,Vneib
        
    def get_sphere_packing(self,C,r,Fa=20,Fv=20):
        import mesh_sphere
        num = C.shape[0]
        M0 = mesh_sphere(C[0],r[0],Fa,Fv)
        for i in range(num-1):
            Si = mesh_sphere(C[i+1], r[i+1],Fa,Fv)
            half = Si.halfedges
            V,E,F = Si.V, Si.E, Si.F
            M0.vertices = np.vstack((M0.vertices, Si.vertices))
            half[:,0] += (i+1)*V
            half[:,1] += (i+1)*F
            half[:,2] += (i+1)*2*E
            half[:,3] += (i+1)*2*E
            half[:,4] += (i+1)*2*E
            half[:,5] += (i+1)*2*E
            M0.halfedges = np.vstack((M0.halfedges, half))
        M0.topology_update()
        return M0

    def get_mobius(self):
        "For A-net first"
        V = self.vertices
        C = np.mean(V, axis=0)
        L,W,H = self.bounding_box()
        diag = np.sqrt((L[1]-L[0])**2+(W[1]-W[0])**2+(H[1]-H[0])**2)
        r = diag * 0.5
        Vmobius = (V-C)/(np.linalg.norm(V-C,axis=1)**2)[:,None] * r**2 + C
        face = self.faces_list()
        mmesh = Mesh()
        mmesh.make_mesh(Vmobius, face)
        return mmesh

    def polylineVertices(self,v1,v2):
        "from given indices of 2 neighbour vertices v1,v2, to get a polyline_vertices"
        H = self.halfedges
        e1 = np.where(H[:,0]==v1)[0]
        for i in e1:
            if H[H[i,4],0]==v2:
                es1 = i
                es2 = H[i,4]
        poly = [v1]
        while H[H[H[es1,3],4],1] !=-1 and v2 not in poly:
            es1 = H[H[H[es1,3],4],3]
            vnew = H[es1,0]
            poly += [vnew]
        if v2 not in poly:
            while H[H[H[es2,3],4],1] !=-1:
                es2 = H[H[H[es2,3],4],3]
                vnew = H[es2,0]
                poly += [vnew]
        poly += [v2]
        return poly

    def polylineVertices2(self,v1,v2):
        "from given indices of 2 neighbour vertices v1,v2, to get a polyline_vertices"
        H = self.halfedges
        e1 = np.where(H[:,0]==v1)[0]
        for i in e1:
            if H[H[H[H[H[i,4],3],4],2],0]==v2:
                es1 = i
                es2 = H[H[H[H[i,3],4],2],4]
        poly = []
        while H[es1,0] not in poly and H[es1,1]!=-1:# and H[H[H[H[es1,4],3],4],1]!=-1:
            poly.append(H[es1,0])
            es1 = H[H[H[H[es1,4],3],4],2]
        while H[es2,0] not in poly:
            poly.append(H[es2,0])
            es2 = H[H[H[H[es2,3],4],2],4]
        return poly


    def get_all_corner_vertices(self):
        v = self.mesh_corners()
        #v = self.corner
        C = self.vertices[v]
        return v,C
    
    def get_i_boundary_vertices(self,e,by_closed=False,by_corner2=False):
        H = self.halfedges
        il,ir = np.where(H[:,5]==e)[0] # should be two
        if H[il,1]==-1:
            e = il
        else:
            e = ir
        if by_closed:
            "no corner vertex of valence 2, which forms closed boundry"
            eb = []
            e1=e
            e2 = H[e,2]
            while H[e1,0] not in H[np.array(eb,dtype=int),0]:
                eb.append(e1)
                e1 = H[e1,3]
            #eb.append(e1)
            while H[e2,0] not in H[np.array(eb,dtype=int),0]:
                eb.append(e2)
                e2 = H[e2,2]
            #eb.append(e2)
            vb = H[np.array(eb,dtype=int),0]
            "above should be no duplicate, then below no use"
            #u, ind = np.unique(vb, return_index=True)
            #vb = u[np.argsort(ind)]
            return vb, self.vertices[vb]
        elif by_corner2: 
            "split by corner vertex of valence 2"
            eb = []
            e1=e
            e2 = H[e,2]
            _,_,lj = self.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
            corner = np.where(lj==2)[0]
            while H[e1,0] not in corner:
                eb.append(e1)
                e1 = H[e1,3]
            eb.append(e1)
            while H[e2,0] not in corner:
                eb.append(e2)
                e2 = H[e2,2]
            eb.append(e2)
            vb = H[np.array(eb,dtype=int),0]
            return vb, self.vertices[vb]
        else:
            "all boundary vertices"
            vb = self.boundary_curves()
            for v in vb:
                if H[e,0] in v:
                    return v, self.vertices[v]
    
    def get_i_boundary_vertex_indices(self,i):
        "work for rectangular-patch + cylinder-annulus"
        #i = self.show_i_boundary
        _,_,lj = self.vertex_ring_vertices_iterators(sort=True,return_lengths=True)
        vcorner = np.where(lj==2)[0]
        H = self.halfedges
        ec = []
        if len(vcorner)!=0:
            for v in vcorner:
                e = np.intersect1d(np.where(H[:,0]==v)[0],np.where(H[:,1]==-1)[0])[0]
                ec.append(e)
            ec = H[np.array(ec,dtype=int),5]
            v,B = self.get_i_boundary_vertices(ec[i],by_corner2=True) 
        else:
            "rotational-case"
            i = int(i%2) # i always equal 0 or 1
            bi = self.boundary_curves(corner_split=True)
            for b in bi:
                v = b[0]
                e = np.intersect1d(np.where(H[:,0]==v)[0],np.where(H[:,1]==-1)[0])[0]
                ec.append(e)
            ec = H[np.array(ec,dtype=int),5]     
            v,B = self.get_i_boundary_vertices(ec[i],by_closed=True)         
        return v,B

    def get_all_boundary_vertices(self,order=False):
        v = self.boundary_vertices()
        if order:
            # v = np.array([],dtype=int)
            v = self.boundary_curves(corner_split=False)[0]
            # for i in vi:
            #     v = np.r_[v,i]
        B = self.vertices[v]
        return v,B

    def get_all_boundary_from_corner_of_v2(self):
        "compared with above get_all_boundary_vertices, it's more in order"
        H = self.halfedges
        c = self.corner
        vb = np.array([],dtype=int)
        v = c[0]
        num = 1
        while num < len(c)+1:
            vb = np.r_[vb,v]
            e1 = np.where(H[:,0]==v)[0]
            e2 = e1[np.where(H[e1,1]==-1)[0]]
            e3 = H[e2,2]
            while H[e3,0] not in c and H[e3,0] not in vb:
                vb = np.r_[vb, H[e3,0]]
                e3 = H[e3,2]
            v = H[e3,0]
            num += 1
        B = self.vertices[vb]
        return vb,B

    def get_close_two_boundary_points_index(self,ratio=0.5,torus=False):
        "join_vertex: after cut, join this two boundaries"
        err = np.min(self.edge_lengths()) * ratio
        v,B = self.get_all_boundary_vertices(order=True)
        #print(len(v),len(np.unique(v)))
        #v,B = self.get_all_boundary_from_corner_of_v2()
        #print(len(v),len(np.unique(v)))
        id1,id2 = [],[]
        arr = np.arange(len(v))
        for i in arr:
            Bi = B[i]
            brr = np.setdiff1d(arr,np.arange(i+1))
            for j in brr:
                Bj = B[j]
                d = np.linalg.norm(Bj-Bi)
                if d < err:
                    id1.append(v[i])
                    id2.append(v[j])
        #print(len(id1),len(id2),len(np.unique(id1)),len(np.unique(id2)),len(np.intersect1d(id1,id2)))
        id1,id2 = np.array(id1),np.array(id2)

        if torus:
            "torus=True only work for torus_2genus.obj"
            corner = self.corner
            k = 0
            a1,a11,b1,b11,a2,a22,b2,b22=[],[],[],[],[],[],[],[]
            for i in np.arange(len(v)-1)+1:
                if v[i-1] in corner:
                    if k==0:
                        a1.append(v[i-1])
                    elif k==1:
                        b1.append(v[i-1])
                    elif k==2:
                        a2.append(v[i-1])
                    elif k==3:
                        b11.append(v[i-1])
                    elif k==4:
                        a22.append(v[i-1])
                    elif k==5:
                        b2.append(v[i-1])
                    elif k==6:
                        a11.append(v[i-1])
                    elif k==7:
                        b22.append(v[i-1])
                if k==0:
                    a1.append(v[i]) # r
                elif k==1:
                    b1.append(v[i]) # r
                elif k==2:
                    a2.append(v[i]) # r
                elif k==3:
                    b11.append(v[i]) # g
                elif k==4:
                    a22.append(v[i]) # g
                elif k==5:
                    b2.append(v[i]) # g
                elif k==6:
                    a11.append(v[i]) # g
                elif k==7:
                    b22.append(v[i]) # r
                if v[i] in corner:
                    k += 1
            b22.append(v[0])
            id1 = np.r_[a1,b1,a2[4:],b2,a2[:5]]
            id2 = np.r_[a11[4:][::-1],b11[::-1],a22[::-1],b22[::-1],a11[:5][::-1]]
        #print(v[0],v[1])
        # print(len(a1),len(b1),len(a2),len(b2))
        # print(len(a11),len(b11),len(a22),len(b22))
        # print(len(id1),len(id2),len(np.unique(id1)),len(np.unique(id2)),len(np.intersect1d(id1,id2)))
        # print(a1,b1,a2,b11,a22,b2,a11,b22)
        self._join_vertex = [id1,id2]

    def get_a_closed_boundary(self,first=True):
        "for non-closed-mesh, return 1-closed-bdry-vertices"
        vblist = self.boundary_curves(corner_split=True)
        if first:
            vcb = vblist[0]
            vblist = np.delete(vblist, 0, 0)
        else:
            vcb = vblist[1]
            vblist = np.delete(vblist, 1, 0)
        vs,vd = vcb[0],vcb[-1]
        while vs!=vd:
            for i in range(len(vblist)):
                vi = vblist[i]
                if vi[0]==vd:
                    vcb = np.r_[vcb,vi[1:]]
                    vd = vi[-1]
                    vblist = np.delete(vblist, i, 0)
                    break
        vcb = np.delete(vcb,-1)
        return vcb, self.vertices[vcb]

    #--------------------------------------------------------------------------
    #                            Unfold quads + joint
    #--------------------------------------------------------------------------
       #  ----p3----            ----p3----
       # |          |          |          |
       # p4   f1    p2=q1==q11=p4   f4    p2
       # |          |    |     |          |
       #  ----p1----    /  \    ----p1----
       #      ||       /    \       ||
       #      q22     |     |       q4
       #      ||      -  *  -       ||
       #      ||      |     |       ||
       #      q2       \   /        q44
       #      ||        \ /         ||
       #  ----p3----     |      ----p3----
       # |          |          |          |
       # p4   f2    p2=q33==q3=p4   f3    p2
       # |          |          |          |
       #  ----p1----            ----p1----
               # blue: c1r=c3r=0, c11r=c33r=1, c22r=c44r=0,c2r=c4r=1
               #  red: c2r=c4r=0, c22r=c44r=1, c1r=c3r=1, c11r=c33r=0

    def _con_cross_product(self,X,c1,c2,coo=2):
        "c1xc2=[x2y3-x3y2,x3y1-x1y3,x1y2-x2y1]"
        "x1y2-x2y1=a**2"
        pass

    def get_part_qi(self,c_p1,c_p2,c_p3,c_p4,idb,index):
        i1,i2,i3,i4,i11,i22,i33,i44 = index
        allx,ally,allz = np.hstack((c_p1.reshape(3,-1),c_p2.reshape(3,-1),c_p3.reshape(3,-1),c_p4.reshape(3,-1)))
        def _qxyz(i1,i11):
            c1 = np.r_[allx[i1],ally[i1],allz[i1]]
            c2 = np.r_[allx[i11],ally[i11],allz[i11]]
            return c1,c2
        cq1,cq11 = _qxyz(i1[idb],i11[idb])
        cq2,cq22 = _qxyz(i2[idb],i22[idb])
        cq3,cq33 = _qxyz(i3[idb],i33[idb])
        cq4,cq44 = _qxyz(i4[idb],i44[idb])
        return cq1,cq11,cq2,cq22,cq3,cq33,cq4,cq44

    def get_all_qi(self,c_p1,c_p2,c_p3,c_p4,idb):
        "idb = np.arange(numv)"
        i1,i2,i3,i4,i11,i22,i33,i44 = self.ind_multi_rr
        allx,ally,allz = np.hstack((c_p1.reshape(3,-1),c_p2.reshape(3,-1),c_p3.reshape(3,-1),c_p4.reshape(3,-1)))
        def _qxyz(i1,i11):
            c1 = np.r_[allx[i1],ally[i1],allz[i1]]
            c2 = np.r_[allx[i11],ally[i11],allz[i11]]
            return c1,c2
        cq1,cq11 = _qxyz(i1[idb],i11[idb])
        cq2,cq22 = _qxyz(i2[idb],i22[idb])
        cq3,cq33 = _qxyz(i3[idb],i33[idb])
        cq4,cq44 = _qxyz(i4[idb],i44[idb])
        return cq1,cq11,cq2,cq22,cq3,cq33,cq4,cq44

    def get_all_ratioi(self,c_a,c_b,c_c,c_d,idb):
        """
        blue: c1r=c3r=0, c11r=c33r=1
        red:  c2r=c4r=0, c22r=c44r=1
        """
        i1,i2,i3,i4,i11,i22,i33,i44 = self.ind_multi_rr
        allr = np.r_[c_a,c_b,c_c,c_d]
        def _qxyz(i1,i11,idb):
            i1,i11 = i1[idb],i11[idb]
            return allr[i11], allr[i1]
        cq11r,cq1r = _qxyz(i1,i11,idb) # f4: 1-d, f1: b
        cq22r,cq2r = _qxyz(i2,i22,idb) # f1: 1-a, f2: c
        cq33r,cq3r = _qxyz(i3,i33,idb) # f2: 1-b, f3: d
        cq44r,cq4r = _qxyz(i4,i44,idb) # f3: 1-c, f4: a
        return cq1r,cq11r,cq2r,cq22r,cq3r,cq33r,cq4r,cq44r

    def get_all_ei(self,c_e1,c_e2,c_e3,c_e4,idb):
        ie1,ie2,ie3,ie4 = self.ind_multi_rr_face
        allx,ally,allz = np.hstack((c_e1.reshape(3,-1),c_e2.reshape(3,-1),c_e3.reshape(3,-1),c_e4.reshape(3,-1)))
        def _qxyz(ie1,idb):
            i1,i2,i3,i4 = ie1.T
            i1,i2,i3,i4 = i1[idb],i2[idb],i3[idb],i4[idb]
            ce1 = np.r_[allx[i1],ally[i1],allz[i1]]
            ce2 = np.r_[allx[i2],ally[i2],allz[i2]]
            ce3 = np.r_[allx[i3],ally[i3],allz[i3]]
            ce4 = np.r_[allx[i4],ally[i4],allz[i4]]
            return [ce1,ce2,ce3,ce4]
        cf1e = _qxyz(ie1,idb)
        cf2e = _qxyz(ie2,idb)
        cf3e = _qxyz(ie3,idb)
        cf4e = _qxyz(ie4,idb)
        return cf1e,cf2e,cf3e,cf4e

    def get_all_pi(self,c_p1,c_p2,c_p3,c_p4): # replaced by get_v4f4_pi below
        "idb = numv"
        idb,idr = self.vertex_rr_check_ind
        arr = np.arange(len(idb)+len(idr))
        ###arr = np.unique(np.r_[idb,idr])
        cf1p,cf2p,cf3p,cf4p = self.get_all_ei(c_p1,c_p2,c_p3,c_p4,arr)
        return cf1p,cf2p,cf3p,cf4p

    def get_v4f4_pi(self,c_p1,c_p2,c_p3,c_p4):
        if1,if2,if3,if4 = self.ind_rr_quadface_order.T
        #forder = self.rr_quadface_order
        #f1,f2,f3,f4 = forder[if1],forder[if2],forder[if3],forder[if4]
        x1,y1,z1 = c_p1.reshape(3,-1)
        x2,y2,z2 = c_p2.reshape(3,-1)
        x3,y3,z3 = c_p3.reshape(3,-1)
        x4,y4,z4 = c_p4.reshape(3,-1)
        def _qxyz(ifi):
            cp1 = np.r_[x1[ifi],y1[ifi],z1[ifi]]
            cp2 = np.r_[x2[ifi],y2[ifi],z2[ifi]]
            cp3 = np.r_[x3[ifi],y3[ifi],z3[ifi]]
            cp4 = np.r_[x4[ifi],y4[ifi],z4[ifi]]
            return [cp1,cp2,cp3,cp4]
        cf1p = _qxyz(if1)
        cf2p = _qxyz(if2)
        cf3p = _qxyz(if3)
        cf4p = _qxyz(if4)
        return cf1p,cf2p,cf3p,cf4p

    def get_all_ci(self,c_center):
        "c_centers ~ forder ~ c_pi"
        "vs: 4 quadface centers -- c1,c2,c3,c4"
        x,y,z = c_center.reshape(3,-1)
        if1,if2,if3,if4 = self.ind_rr_quadface_order.T
        def _c(i1):
            return np.r_[x[i1],y[i1],z[i1]]
        return _c(if1),_c(if2),_c(if3),_c(if4)

    # def get_all_ci(self,c_center): could work for mesh whose v4~f4
    #     "c_centers ~ forder ~ c_pi"
    #     "vs: 4 quadface centers -- c1,c2,c3,c4"
    #     i1,i2,i3,i4,_,_,_,_ = self.ind_multi_rr
    #     x,y,z = c_center.reshape(3,-1)
    #     allx = np.tile(x,4)
    #     ally = np.tile(y,4)
    #     allz = np.tile(z,4)
    #     def _c(i1):
    #         return np.r_[allx[i1],ally[i1],allz[i1]]
    #     return _c(i1),_c(i2),_c(i3),_c(i4)

    def get_all_li(self,c_l1,c_l2,c_l3,c_l4,idb):
        "four edge_edges of each face at vertex star"
        ie1,ie2,ie3,ie4 = self.ind_multi_rr_face
        alll = np.r_[c_l1,c_l2,c_l3,c_l4]
        def _qxyz(ie1,idb):
            i1,i2,i3,i4 = ie1.T
            i1,i2,i3,i4 = i1[idb],i2[idb],i3[idb],i4[idb]
            cl1,cl2,cl3,cl4 = alll[i1],alll[i2],alll[i3],alll[i4]
            return [cl1,cl2,cl3,cl4]
        cf1l = _qxyz(ie1,idb)
        cf2l = _qxyz(ie2,idb)
        cf3l = _qxyz(ie3,idb)
        cf4l = _qxyz(ie4,idb)
        return cf1l,cf2l,cf3l,cf4l

    def get_all_diagonal_li(self,c_l13,c_l24,idb):
        ie1,ie2,ie3,ie4 = self.ind_multi_rr_face
        alldl = np.r_[c_l13,c_l24,c_l13,c_l24]
        def _qxyz(ie1,idb):
            i1,i2,i3,i4 = ie1.T
            i1,i2,i3,i4 = i1[idb],i2[idb],i3[idb],i4[idb]
            cl13,cl24 = alldl[i1],alldl[i2]
            return [cl13,cl24]
        cf1l = _qxyz(ie1,idb)
        cf2l = _qxyz(ie2,idb)
        cf3l = _qxyz(ie3,idb)
        cf4l = _qxyz(ie4,idb)
        return cf1l,cf2l,cf3l,cf4l

    def get_vs_diagonal_v(self,v4f4=True,ck1=False,ck2=False,index=True):
        """
        should be same with v,va,vb,vc,vd = self.rr_star_corner
        """
        H = self.halfedges
        star = self.rr_star
        if v4f4:
            star = self.rr_star[self.ind_rr_star_v4f4]
        if ck1:
            star = self.rr_star[self.ind_ck_rr_vertex[0]]
        elif ck2:
            star = self.rr_star[self.ind_ck_rr_vertex[1]]

        is1,is2,is3,is4 = [],[],[],[]
        for i in range(len(star)):
            v,v1,v2,v3,v4 = star[i,:]
            e=np.where(H[:,0]==v)[0]
            e1=np.where(H[H[:,4],0]==v1)[0]
            e2=np.where(H[H[:,4],0]==v2)[0]
            e3=np.where(H[H[:,4],0]==v3)[0]
            e4=np.where(H[H[:,4],0]==v4)[0]
            i1 = np.intersect1d(e,e1)[0]
            i2 = np.intersect1d(e,e2)[0]
            i3 = np.intersect1d(e,e3)[0]
            i4 = np.intersect1d(e,e4)[0]
            is1.append(H[H[H[i1,2],2],0])
            is2.append(H[H[H[i2,2],2],0])
            is3.append(H[H[H[i3,2],2],0])
            is4.append(H[H[H[i4,2],2],0])
        if index:
            return star[:,0],np.array(is1),np.array(is2),np.array(is3),np.array(is4)
        arr_s0 = self.columnnew(star[:,0],0,self.V)
        arr_s1 = self.columnnew(np.array(is1),0,self.V)
        arr_s2 = self.columnnew(np.array(is2),0,self.V)
        arr_s3 = self.columnnew(np.array(is3),0,self.V)
        arr_s4 = self.columnnew(np.array(is4),0,self.V)
        return arr_s0,arr_s1,arr_s2,arr_s3,arr_s4

    def get_rectangle_patch_index_matrix(self,face=True):
        "only for rectangle-patch m*n matrix"
        H = self.halfedges
        c = self.corner[0] # first corner
        e = np.intersect1d(np.where(H[:,0]==c)[0], np.where(H[:,1]==-1)[0])[0]
        def row(e):
            e_ri = []
            while H[H[e,2],0] not in self.corner:
                e_ri.append(e)
                e = H[e,2]
            return e_ri
        def col(e):
            e_ci = []
            while H[e,0] not in self.corner:
                e_ci.append(e)
                e = H[e,3]
            e_ci.append(e)
            return e_ci
        def row_face(e):
            e = H[H[H[e,4],2],2]
            f_ri = np.array([],dtype=int)
            while H[H[e,4],1]!=-1:
                f_ri = np.r_[f_ri, H[e,1]]
                e = H[H[H[e,4],2],2]
            f_ri = np.r_[f_ri, H[e,1]]
            return f_ri

        e_left = col(H[e,3])
        f_matrix = []
        for e in e_left:
            fi = row_face(e)
            f_matrix.append(fi)
        return np.array(f_matrix)

    def get_symmetric_rectangle_patch_face_order(self,col=False): # need to check col
        "symmetric ways: 1.column (left-right), 2.row (up-down)"
        fmatrix = self.get_rectangle_patch_index_matrix()
        m,n = fmatrix.shape
        forder = self.rr_quadface_order
        new = []
        if col:
            "1.column (left-right)"
            for f in forder:
                i,j = np.where(fmatrix==f)
                i,j = i[0],n-1-j[0]
                new.append(fmatrix[i,j])
            ind1 = fmatrix[5,:]    
        else:
            "2.row (up-down)"
            for f in forder:
                i,j = np.where(fmatrix==f)
                i,j = m-1-i[0],j[0]
                new.append(fmatrix[i,j])
            ind1 = fmatrix[:,5]  
        forder1 = np.array(new)

        i1,i2 = [],[]
        for i in ind1:
            i1.append(np.where(forder==i)[0][0])
            i2.append(np.where(forder1==i)[0][0])
        idd1,idd2 = np.array(i1),np.array(i2)
        return forder1,idd1,idd2
    
    def get_symmetric_rectangle_patch_vertex_order(self,col=True): 
        "symmetric ways: 1.column (left-right), 2.row (up-down)"
        vmatrix = self.patch_matrix[1:-1,1:-1]
        m,n = vmatrix.shape
        matrix = np.arange(m*n).reshape(m,n)
        vorder = self.rr_star[:,0]
        vorder0 = []
        new = []
        if col:
            "1.column (left-right)"
            for v in vorder:
                i,j = np.where(vmatrix==v)
                i,j = i[0],n-1-j[0]
                vorder0.append(vmatrix[i,j])
                new.append(matrix[i,j])
            ind1 = vmatrix[5,:]    
        else:
            "2.row (up-down)"
            for v in vorder:
                i,j = np.where(vmatrix==v)
                i,j = m-1-i[0],j[0]
                vorder0.append(vmatrix[i,j])
                new.append(matrix[i,j])
            ind1 = vmatrix[:,5]  
        vorder0 = np.array(vorder0)
        vorder1 = np.array(new)
        i1,i2 = [],[]
        for i in ind1:
            i1.append(np.where(vorder==i)[0][0])
            i2.append(np.where(vorder0==i)[0][0])
        idd1,idd2 = np.array(i1),np.array(i2)
        return vorder1,idd1,idd2       
#-----------------------------------------------------------------------------
    def con_auxetic_edge_length(self,X,numf,c_p1,c_p2,c_p3,c_p4,L0,var): # could use, but no here.
        "(Pj-Pi)^2 = li^2"
        #numf = quad.F
        l1,l2,l3,l4 = L0[:numf],L0[numf:2*numf],L0[2*numf:3*numf],L0[3*numf:4*numf]
        l13,l24 = L0[4*numf:5*numf],L0[5*numf:]
        # arr = np.arange(3*numf)
        # c_p1,c_p2,c_p3,c_p4 = arr,3*numf+arr,6*numf+arr,9*numf+arr
        idb,idr = self.vertex_rr_check_ind
        num = len(idb)+len(idr)
        cf1l,cf2l,cf3l,cf4l = self.get_all_li(l1,l2,l3,l4,np.arange(num))
        cf1dl,cf2dl,cf3dl,cf4dl = self.get_all_diagonal_li(l13,l24,np.arange(num))
        cf1p,cf2p,cf3p,cf4p = self.get_all_ei(c_p1,c_p2,c_p3,c_p4,np.arange(num))
        arr0 = np.arange(num)
        def _quad_edge_length(cf1l,cf1p,cf1dl):
            def _edge(c1,c2,l1):
                "(P1-P2)^2 = l1**2"
                col = np.r_[c1,c2]
                row = np.tile(arr0,6)
                data = 2*np.r_[X[c1]-X[c2],X[c2]-X[c1]]
                r = np.linalg.norm((X[c1]-X[c2]).reshape(-1,3, order='F'),axis=1)**2+l1**2
                H = sparse.coo_matrix((data,(row,col)), shape=(num, var))
                return H,r
            H1,r1 = _edge(cf1p[0],cf1p[1],cf1l[0]) # l1
            H2,r2 = _edge(cf1p[1],cf1p[2],cf1l[1]) # l2
            H3,r3 = _edge(cf1p[2],cf1p[3],cf1l[2]) # l3
            H4,r4 = _edge(cf1p[3],cf1p[0],cf1l[3]) # l4
            H5,r5 = _edge(cf1p[2],cf1p[0],cf1dl[0]) # l13
            H6,r6 = _edge(cf1p[3],cf1p[1],cf1dl[1]) # l24
            H = sparse.vstack((H1,H2,H3,H4,H5,H6))
            r = np.r_[r1,r2,r3,r4,r5,r6]
            return H,r
        H1,r1 = _quad_edge_length(cf1l,cf1p,cf1dl)
        H2,r2 = _quad_edge_length(cf2l,cf2p,cf2dl)
        H3,r3 = _quad_edge_length(cf3l,cf3p,cf3dl)
        H4,r4 = _quad_edge_length(cf4l,cf4p,cf4dl)
        H = sparse.vstack((H1,H2,H3,H4))
        r = np.r_[r1,r2,r3,r4]
        return H,r

    def con_auxetic_touch(self,c_p1,c_p2,c_p3,c_p4,idb,N):
        """ connected quads by touching points at each regular_vertex
        four quads 1,2,3,4 in clockwise:
        all_c = [c_1,c_2,c_3,c_4]
        Qi = all_c[i1] = all_c[i11]
        """
        cq1,cq11,cq2,cq22,cq3,cq33,cq4,cq44 = self.get_all_qi(c_p1,c_p2,c_p3,c_p4,idb)
        num = len(cq1)
        row = np.tile(np.arange(num),2)
        data = np.r_[np.ones(num), -np.ones(num)]
        col1 = np.r_[cq1,cq11]
        col2 = np.r_[cq2,cq22]
        col3 = np.r_[cq3,cq33]
        col4 = np.r_[cq4,cq44]
        H1 = sparse.coo_matrix((data,(row,col1)), shape=(num, N))
        H2 = sparse.coo_matrix((data,(row,col2)), shape=(num, N))
        H3 = sparse.coo_matrix((data,(row,col3)), shape=(num, N))
        H4 = sparse.coo_matrix((data,(row,col4)), shape=(num, N))
        H = sparse.vstack((H1,H2,H3,H4))
        r = np.zeros(4*num)
        return H * 100, r * 100

    def con_shrink_points(self,numv,c_p1,c_p2,c_p3,c_p4,var,collapse=False,boundary=True,corner=True):
        "blue: Q1==Q3; red: Q2=Q4"
        def shrink_points(cq1,cq11,cq3,cq33,idb):
            "q1=q3"
            col = np.r_[cq1,cq11,cq3,cq33]
            num = len(idb)*2
            row = np.tile(np.arange(3*num),2)
            data = np.r_[np.ones(3*num),-np.ones(3*num)]
            H = sparse.coo_matrix((data,(row,col)), shape=(3*num, var))
            r = np.zeros(3*num)
            return H*100, r*100
        idb,idr = self.vertex_check_ind
        cq1,cq11,cb2,cb22,cq3,cq33,cb4,cb44 = self.get_all_qi(c_p1,c_p2,c_p3,c_p4,idb)
        H1,r1 = shrink_points(cq1,cq11,cq3,cq33,idb)
        cr1,cr11,cq2,cq22,cr3,cr33,cq4,cq44 = self.get_all_qi(c_p1,c_p2,c_p3,c_p4,idr)
        H2,r2 = shrink_points(cq2,cq22,cq4,cq44,idr)
        H = sparse.vstack((H1,H2))
        r = np.r_[r1,r2]

        if boundary:
            idbb,idrr = self.vertex_check_near_boundary_ind
            if idbb is not None:
                cf1p,cf2p,cf3p,cf4p  = self.get_all_ei(c_p1,c_p2,c_p3,c_p4,idbb)
                H1,r1 = shrink_points(cf1p[2],cf2p[0],cf4p[2],cf3p[0],idbb)
                H = sparse.vstack((H,H1))
                r = np.r_[r,r1]
            if idrr is not None:
                cf1p,cf2p,cf3p,cf4p  = self.get_all_ei(c_p1,c_p2,c_p3,c_p4,idrr)
                H2,r2 = shrink_points(cf1p[3],cf3p[1],cf2p[3],cf4p[1],idrr)
                H = sparse.vstack((H,H2))
                r = np.r_[r,r2]

        if collapse:
            "blue: Q1==Q3=Vblue; red: Q2=Q4=Vred"
            vblue,vred = self.vertex_check
            c_vb = np.r_[vblue,numv+vblue,2*numv+vblue]
            c_vr = np.r_[vred,numv+vred,2*numv+vred]
            H1,r1 = shrink_points(cq1,cq11,c_vb,c_vb,idb)
            H2,r2 = shrink_points(cq3,cq33,c_vb,c_vb,idb)
            H3,r3 = shrink_points(cq2,cq22,c_vr,c_vr,idr)
            H4,r4 = shrink_points(cq4,cq44,c_vr,c_vr,idr)
            H = sparse.vstack((H,H1,H2,H3,H4))
            r = np.r_[r,r1,r2,r3,r4]
            if boundary: # including above case + boundary, but no corner
                "blue: Q2=v2,Q4=v4; red: Q1=v1,Q3=v3; vertex_star=[v,v1,v2,v3,v4]"
                vbstar,vrstar=self.vertex_check_star
                v2,v4 = vbstar[:,2],vbstar[:,4]
                v1,v3 = vrstar[:,1],vrstar[:,3]
                c_vb2 = np.r_[v2,numv+v2,2*numv+v2]
                c_vb4 = np.r_[v4,numv+v4,2*numv+v4]
                c_vr1 = np.r_[v1,numv+v1,2*numv+v1]
                c_vr3 = np.r_[v3,numv+v3,2*numv+v3]
                H1,r1 = shrink_points(cb2,cb22,c_vb2,c_vb2,idb)
                H2,r2 = shrink_points(cb4,cb44,c_vb4,c_vb4,idb)
                H3,r3 = shrink_points(cr1,cr11,c_vr1,c_vr1,idr)
                H4,r4 = shrink_points(cr3,cr33,c_vr3,c_vr3,idr)
                H = sparse.vstack((H,H1,H2,H3,H4))
                r = np.r_[r,r1,r2,r3,r4]
            if corner:
                "blue: V[indb[0]=P[indb[1]]; red: V[indr[0]=P[indr[1]]"
                indb,indr = self.vertex_check_corner_ind
                if len(indb)!=0 and len(indr)!=0:
                    vi = np.r_[indb[:,0],indr[:,0]]
                    pi = np.r_[indb[:,1],indr[:,1]]
                elif len(indb)!=0:
                    vi,pi = indb[:,0],indb[:,1]
                elif len(indr)!=0:
                    vi,pi = indr[:,0],indr[:,1]
                num = len(vi)
                allx,ally,allz = np.hstack((c_p1.reshape(3,-1),c_p2.reshape(3,-1),c_p3.reshape(3,-1),c_p4.reshape(3,-1)))
                colp = np.r_[allx[pi],ally[pi],allz[pi]]
                colv = np.r_[vi,numv+vi,2*numv+vi]
                col = np.r_[colv,colp]
                row = np.tile(np.arange(3*num),2)
                data = np.r_[np.ones(3*num),-np.ones(3*num)]
                Hc = sparse.coo_matrix((data,(row,col)), shape=(3*num, var))
                rc = np.zeros(3*num)
                H = sparse.vstack((H,Hc))
                r = np.r_[r,rc]
        return H,r

    def con_cos(self,X,c1,c2,c3,cos,l1,l2,var): # no use!
        numf = len(cos)
        arr0 = np.arange(numf)
        row = np.tile(arr0,9)
        col = np.r_[c1,c2,c3]
        data =np.r_[X[c3]-X[c2],2*X[c2]-X[c1]-X[c3],X[c1]-X[c2]]
        H = sparse.coo_matrix((data,(row,col)), shape=(numf, var))
        r = np.einsum('ij,ij->i',(X[c1]-X[c2]).reshape(-1,3, order='F'),(X[c3]-X[c2]).reshape(-1,3, order='F'))
        r = r-l1*l2*cos
        return H,r

    def get_auxetic_sphere(self,quad,X=None,initial=True):
        if initial: # basic way
            B = quad.bounding_box()
            C = quad.mesh_center()
            rmin = np.linalg.norm(np.array([B[0][0],B[1][0],B[2][0]])-C)*0.5
            rmax = np.linalg.norm(np.array([B[0][1],B[1][1],B[2][1]])-C)*0.5
            r = (rmax+rmin)*0.5
        # elif initial: # interpolate way
        #     #star = np.insert(self.ver_regular_star, 0, values=self.ver_regular, axis=1)
        #     centers,radius,_,_ = sphere_equation(self.vertices,self.ver_regular,self.ringlist)
        #     C,r = np.mean(centers,axis=0), np.mean(radius)
        #     #print(centers,radius)
        else:
            c_c,c_r = self.__Ns+np.arange(3), self.__Ns+3
            C,r = X[c_c],X[c_r]
        return C,r

    def initial_unfold(self,quad,sphere,nonoverpalling,coo=2):
        #"X = [Q1234(x,y,z)]=[Q1x,Q2x,Q3x,Q4x, Q1y,2y,3y,4y,1z,2z,3z,4z]"
        "X=[Q1x,y,z; Q2x,y,z;...]"
        "quad.F * 4 = quad.V"
        qv = quad.vertices
        numf = quad.F
        idrr = np.arange(4*numf).reshape(4,-1)
        P1 = qv[idrr[0,:]].flatten('F')
        P2 = qv[idrr[1,:]].flatten('F')
        P3 = qv[idrr[2,:]].flatten('F')
        P4 = qv[idrr[3,:]].flatten('F')
        X = np.r_[P1,P2,P3,P4]
        if nonoverpalling:
            arr = np.arange(3*numf)
            c_p1,c_p2,c_p3,c_p4 = arr,3*numf+arr,6*numf+arr,9*numf+arr
            numv = self.num_rrv
            idb = np.arange(numv)
            cq1,cq11,cq2,cq22,cq3,cq33,cq4,cq44 = self.get_all_qi(c_p1,c_p2,c_p3,c_p4,idb)
            def _normal(cq1,cq2,cq3,cq4,n0):
                Q1,Q2,Q3,Q4 = X[cq1],X[cq2],X[cq3],X[cq4]
                vec1 = (Q1-Q3).reshape(-1,3,order='F')
                vec2 = (Q2-Q3).reshape(-1,3,order='F')
                vec3 = (Q4-Q3).reshape(-1,3,order='F')
                n1 = np.cross(vec1,vec2)
                n2 = np.cross(vec3,vec1)
                d1 = np.abs(np.einsum('ij,ij->i',n1,n0))
                d2 = np.abs(np.einsum('ij,ij->i',n2,n0))
                eps = np.finfo(float).eps
                a1 = np.sqrt(d1+eps)
                a2 = np.sqrt(d2+eps)
                sqrt_a = np.r_[a1,a2]
                return sqrt_a
            n0 = np.zeros(len(cq1)).reshape(-1,3)
            n0[:,coo] = 1
            aa = _normal(cq1,cq2,cq3,cq4,n0)
            aaa = _normal(cq11,cq22,cq33,cq44,n0)
            self.__No = len(X)
            X = np.r_[X,aa,aaa]
        if sphere:
            C,r = self.get_auxetic_sphere(quad,initial=True)
            #C,r = [0,0,0],1
            self.__Ns = len(X)
            X = np.r_[X,C,r]
        var = len(X)
        return X,var

    def opt_unfold_joint_quad(self,quad,X,L0,var,shrink,plane,sphere,nonoverpalling,coo=2,ortho=False):
        numf,numv = quad.F,quad.V
        arr0 = np.arange(numf)
        arr = np.arange(3*numf)
        c_p1,c_p2,c_p3,c_p4 = arr,3*numf+arr,6*numf+arr,9*numf+arr
        numq = self.num_rrv
        idb = np.arange(numq)
        cq1,cq11,cq2,cq22,cq3,cq33,cq4,cq44 = self.get_all_qi(c_p1,c_p2,c_p3,c_p4,idb)
        def _con_in_plane(var,coo):
            "coo=0,1,2 ~ yz, xz, xy -- plane"
            row = np.arange(numv)
            crr = np.arange(numf)
            c1 = coo*numf+crr
            c2 = 3*numf+c1
            c3 = 3*numf+c2
            c4 = 3*numf+c3
            col = np.r_[c1,c2,c3,c4]
            data = np.ones(numv)
            H = sparse.coo_matrix((data,(row,col)), shape=(numv, var))
            r = np.zeros(numv)
            return H, r
        def _con_on_sphere(var):
            "(qi-c)^2=r^2"
            c_c,c_r = self.__Ns+np.arange(3), self.__Ns+3
            col_c,col_r = np.repeat(c_c,numf),np.tile(c_r,numf)
            row = np.tile(np.tile(arr0,7),4)
            col1 = np.r_[c_p1,col_c,col_r]
            col2 = np.r_[c_p2,col_c,col_r]
            col3 = np.r_[c_p3,col_c,col_r]
            col4 = np.r_[c_p4,col_c,col_r]
            col = np.r_[col1,col2,col3,col4]
            data1 = 2*np.r_[X[c_p1]-X[col_c],-X[c_p1]+X[col_c],-X[col_r]]
            data2 = 2*np.r_[X[c_p2]-X[col_c],-X[c_p2]+X[col_c],-X[col_r]]
            data3 = 2*np.r_[X[c_p3]-X[col_c],-X[c_p3]+X[col_c],-X[col_r]]
            data4 = 2*np.r_[X[c_p4]-X[col_c],-X[c_p4]+X[col_c],-X[col_r]]
            data = np.r_[data1,data2,data3,data4]
            r1 = np.linalg.norm((X[c_p1]-X[col_c]).reshape(-1,3, order='F'),axis=1)-X[col_r]**2
            r2 = np.linalg.norm((X[c_p2]-X[col_c]).reshape(-1,3, order='F'),axis=1)-X[col_r]**2
            r3 = np.linalg.norm((X[c_p3]-X[col_c]).reshape(-1,3, order='F'),axis=1)-X[col_r]**2
            r4 = np.linalg.norm((X[c_p4]-X[col_c]).reshape(-1,3, order='F'),axis=1)-X[col_r]**2
            r = np.r_[r1,r2,r3,r4]
            H = sparse.coo_matrix((data,(row,col)), shape=(4*numf, var))
            return H,r
        def _con_quad_edge_length(X,L0,var): # could use
            "(P1-P2)*(P3-P2)=l1*l2*cos0; (P4-P1)*(P2-P1)=l1*l4*cosa"
            l1,l2,l3,l4 = L0[:numf],L0[numf:2*numf],L0[2*numf:3*numf],L0[3*numf:4*numf]
            l13,l24 = L0[4*numf:5*numf],L0[5*numf:]
            def _edge(c1,c2,l1):
                "(P1-P2)^2 = l1**2"
                col = np.r_[c1,c2]
                row = np.tile(arr0,6)
                data = 2*np.r_[X[c1]-X[c2],X[c2]-X[c1]]
                r = np.linalg.norm((X[c1]-X[c2]).reshape(-1,3, order='F'),axis=1)**2+l1**2
                H = sparse.coo_matrix((data,(row,col)), shape=(numf, var))
                return H,r
            H1,r1 = _edge(c_p1,c_p2,l1)
            H2,r2 = _edge(c_p2,c_p3,l2)
            H3,r3 = _edge(c_p3,c_p4,l3)
            H4,r4 = _edge(c_p4,c_p1,l4)
            H5,r5 = _edge(c_p3,c_p1,l13)
            H6,r6 = _edge(c_p4,c_p2,l24)
            H = sparse.vstack((H1,H2,H3,H4,H5,H6))
            r = np.r_[r1,r2,r3,r4,r5,r6]
            return H ,r
        def _con_nonoverlapping(coo=2):
            """n1=(Q1-Q3)X(Q2-Q3); n2=(Q4-Q3)X(Q1-Q3)
               n1*n0=a1^2; n2*n0=a2^2
               c1xc2=[x2y3-x3y2,x3y1-x1y3,x1y2-x2y1]
               if coo=2: x1y2-x2y1=a**2
               (q1x-q3x,q1y-q3y,q1z-q3z)x(q2x-q3x,q2y-q3y,q2z-q3z)*(0,0,1)
               = (q1x-q3x)*(q2y-q3y)-(q1y-q3y)*(q2x-q3x)=a**2
               =q1x*q2y-q1x*q3y - q1y*q2x+q1y*q3x - q3x*q2y+q3y*q2x=a^2
            """
            arrq0 = np.arange(numq)
            arrq = np.arange(2*numq)
            c_aa, c_aaa = self.__No+arrq, self.__No+2*numq+arrq
            c_1x,c_2x,c_3x,c_4x = cq1[arrq0],cq2[arrq0],cq3[arrq0],cq4[arrq0]
            c_1xx,c_2xx,c_3xx,c_4xx = cq11[arrq0],cq22[arrq0],cq33[arrq0],cq44[arrq0]
            def _cross(c_1x,c_2x,c_3x,c_4x,ca):
                "(Q1-Q3)X(Q2-Q3)"
                "q1x*q2y-q1x*q3y-q3x*q2y-q1y*q2x+q1y*q3x+q3y*q2x=a^2"
                c_1y,c_2y,c_3y,c_4y = numq+c_1x,numq+c_2x,numq+c_3x,numq+c_4x
                c1x,c1y = np.r_[c_1x,c_4x], np.r_[c_1y,c_4y]
                c2x,c2y = np.r_[c_2x,c_1x], np.r_[c_2y,c_1y]
                c3x,c3y = np.tile(c_3x,2), np.tile(c_3y,2)
                row = np.tile(np.arange(len(c1x)),7)
                col = np.r_[c1x,c1y,c2x,c2y,c3x,c3y,ca]
                d1 = X[c2y]-X[c3y]
                d2 = X[c3x]-X[c2x]
                d3 = X[c3y]-X[c1y]
                d4 = X[c1x]-X[c3x]
                d5 = X[c1y]-X[c2y]
                d6 = X[c2x]-X[c1x]
                d7 = -2*X[ca]
                data = np.r_[d1,d2,d3,d4,d5,d6,d7]
                H = sparse.coo_matrix((data,(row,col)), shape=(2*numq, var))
                r =(X[c1x]-X[c3x])*(X[c2y]-X[c3y])-(X[c1y]-X[c3y])*(X[c2x]-X[c3x])-X[ca]**2
                return H,r
            H1,r1 = _cross(c_1x,c_2x,c_3x,c_4x,c_aa)
            H2,r2 = _cross(c_1xx,c_2xx,c_3xx,c_4xx,c_aaa)
            H = sparse.vstack((H1,H2))
            r = np.r_[r1,r2]
            return H,r

        def _fair_ortho(X,var):
            "(v_l-v_r)*(v_u-v_d)=0"
            c_l = np.r_[cq2,cq22]
            c_r = np.r_[cq4,cq44]
            c_u = np.r_[cq1,cq11]
            c_d = np.r_[cq3,cq33]
            n = numq
            col = np.r_[c_l,c_r,c_u,c_d]
            row = np.tile(np.arange(n),12)
            data = np.r_[X[c_u]-X[c_d],X[c_d]-X[c_u],X[c_l]-X[c_r],X[c_r]-X[c_l]]
            K = sparse.coo_matrix((data,(row,col)), shape=(n, var))
            s = np.einsum('ij,ij->i',(X[c_l]-X[c_r]).reshape(-1,3, order='F'),(X[c_u]-X[c_d]).reshape(-1,3, order='F'))
            return K,s

        H1,r1 = _con_quad_edge_length(X,L0,var)
        #H1,r1 = self.con_auxetic_edge_length(X,numf,c_p1,c_p2,c_p3,c_p4,L0,var)
        idb = np.arange(numq)
        H2,r2 = self.con_auxetic_touch(c_p1,c_p2,c_p3,c_p4,idb,var)
        H = sparse.vstack((H1*10,H2))
        r = np.r_[r1*10,r2]

        if plane:
            H3,r3 = _con_in_plane(var,coo)
            H = sparse.vstack((H,H3))
            r = np.r_[r,r3]
        if sphere:
            Hs,rs = _con_on_sphere(var)
            H = sparse.vstack((H,Hs))
            r = np.r_[r,rs]
        if shrink:
            Hs,rs = self.con_shrink_points(numv,c_p1,c_p2,c_p3,c_p4,var)
            H = sparse.vstack((H,Hs*10))
            r = np.r_[r,rs*10]
        if nonoverpalling:
            H4,r4 = _con_nonoverlapping()
            H = sparse.vstack((H,H4))
            r = np.r_[r,r4]
        if ortho:
            K,s = _fair_ortho(X,var)
            H = sparse.vstack((H,K*0.01))
            r = np.r_[r,s*0.01]
        opt = np.sum(np.square((H*X)-r))
        return H,r,opt

    def opt_unfold_quad(self,quad,L0,shrink,plane,sphere,nonoverpalling=False,ee=0.001,itera=20):
        n = 0
        X,self._var = self.initial_unfold(quad,sphere,nonoverpalling)
        self.matrix_fair(self._var,efair=0.005)
        opt_num, opt = 100, 100
        while n < itera and (opt_num>0.0001 or opt>0.0001) and opt_num<1000000:
            K = self.matrixK
            I = self.matrixI
            H, r, opt = self.opt_unfold_joint_quad(quad,X,L0,self._var,shrink,plane,sphere,nonoverpalling)
            X = sparse.linalg.spsolve(H.T*H+K.T*K+I, H.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
            #X = sparse.linalg.spsolve(H.T*H+I, H.T*r+np.dot(ee**2,X).T,permc_spec=None, use_umfpack=True)
            n += 1
            opt_num = np.sum(np.square((H*X)-r))
            print('Optimize the unfolding mesh:',n, opt_num)
        numf = quad.F
        P1 = X[:3*numf].reshape(-1,3,order='F')
        P2 = X[3*numf:6*numf].reshape(-1,3,order='F')
        P3 = X[6*numf:9*numf].reshape(-1,3,order='F')
        P4 = X[9*numf:12*numf].reshape(-1,3,order='F')
        V = np.vstack((P1,P2,P3,P4))
        return V,P1,P2,P3,P4,X

    def opt_unfold_quad_step(self,quad,sphere,nonoverpalling,efair):
        X, self._var = self.initial_unfold(quad,sphere,nonoverpalling)
        self.matrix_fair(self._var,efair)
        K = self.matrixK
        I = self.matrixI
        return quad,X,K,I,self._var

    def matrix_unit(self, var, ee=0.001):
        I = sparse.eye(var,format='coo')
        I = ee**2*I
        self._matrixI = I

    def matrix_fair(self,var,efair=0.005):
        "for vertex star: v_left+v_right=v_up+v_down"
        #numf = quad.F
        numf = self.num_rrf
        arr = np.arange(3*numf)
        c_p1,c_p2,c_p3,c_p4 = arr,3*numf+arr,6*numf+arr,9*numf+arr
        numq = self.num_rrv
        idb = np.arange(numq)
        cq1,cq11,cq2,cq22,cq3,cq33,cq4,cq44 = self.get_all_qi(c_p1,c_p2,c_p3,c_p4,idb)
        c_l = np.r_[cq2,cq22]
        c_r = np.r_[cq4,cq44]
        c_u = np.r_[cq1,cq11]
        c_d = np.r_[cq3,cq33]
        row = np.tile(np.arange(6*numq),4)
        col = np.r_[c_l,c_r,c_u,c_d]
        data = np.r_[np.ones(6*numq),np.ones(6*numq),-np.ones(6*numq),-np.ones(6*numq)]
        K = sparse.coo_matrix((data,(row,col)), shape=(6*numq, var))
        K = efair * K
        self._matrixK = K

    def matrix_fair2(self,quad,var,efair=0.001):
        """
        blue: Q1=Q3. a=|Q1Q2|=|Q2Q3|, b=|Q1Q4|=|Q3Q4|
             VC = (Q1+Q3)/2 = b/(a+b)*Q2 + a/(a+b)*Q4;
        red : Q2=Q4. a=|Q1Q2|=|Q1Q4|, b=|Q2Q3|=|Q3Q4|
             VC = (Q2+Q4)/2 = b/(a+b)*Q1 + a/(a+b)*Q3
        """

    def matrix_fair0(self,quad,var,efair=0.005,bary=False): # Hui: problem!
        """for each quad face f, two neighbour f_left/right/up/down
        Barycenters: Bc = (Bl+Br)/2
                    2*(vc1+vc2+vc3+vc4) = (vl1+vl2+vl3+vl4) + (vr1+vr2+vr3+vr4)
        """
        farr = self.rr_quadface
        fc,fl,fr = self.rr_quadface_tristar.T
        numv = quad.V
        vc1,vc2,vc3,vc4 = farr[fc].T
        vl1,vl2,vl3,vl4 = farr[fl].T
        vr1,vr2,vr3,vr4 = farr[fr].T
        n = len(fc)
        if bary:
            row = np.tile(np.arange(3*n),12)
            def _col(vc1,vc2,vc3,vc4):
                c1 = self.columnnew(vc1,0,numv)
                c2 = self.columnnew(vc2,0,numv)
                c3 = self.columnnew(vc3,0,numv)
                c4 = self.columnnew(vc4,0,numv)
                cc = np.r_[c1,c2,c3,c4]
                return cc
            c_c = _col(vc1,vc2,vc3,vc4)
            c_l = _col(vl1,vl2,vl3,vl4)
            c_r = _col(vr1,vr2,vr3,vr4)
            col = np.r_[c_c,c_l,c_r]
            data = np.r_[2*np.ones(12*n),-np.ones(24*n)]
            K = sparse.coo_matrix((data,(row,col)), shape=(3*n, var))
        else:
            "2*pc_i = pl_i+pr_i, i=1,2,3,4"
            def _each(vc1,vl1,vr1):
                row = np.tile(np.arange(3*n),3)
                cc1 = self.columnnew(vc1,0,numv)
                cl1 = self.columnnew(vl1,0,numv)
                cr1 = self.columnnew(vr1,0,numv)
                col = np.r_[cc1,cl1,cr1]
                data = np.r_[2*np.ones(3*n),-np.ones(6*n)]
                K1 = sparse.coo_matrix((data,(row,col)), shape=(3*n, var))
                return K1
            K1 = _each(vc1,vl1,vr1)
            K2 = _each(vc2,vl2,vr2)
            K3 = _each(vc3,vl3,vr3)
            K4 = _each(vc4,vl4,vr4)
            K = sparse.vstack((K1,K2,K3,K4))
        K = efair * K
        self._matrixK = K
