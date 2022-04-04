#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(path)

from quadrings import MMesh

import numpy as np
#------------------------------------------------------------------------------

''' used for AAG-project, linking to Grasshopper by Hops for VISUALIZATION
'''
__author__ = 'Hui Wang'
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                        VISUALIZATION
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class Gridshell_AGNet(MMesh):

    def __init__(self,**kwargs):

        MMesh.__init__(self)
        
        self.opt_Anet = kwargs.get('Anet', False)
        
        self.opt_Anet_diagnet = kwargs.get('AnetDiagnet', False)
        
        self.opt_Gnet = kwargs.get('Gnet', False)
        
        self.opt_Gnet_diagnet = kwargs.get('GnetDiagnet', False)
        
        self.set_another_polyline = kwargs.get('direction_poly', 0) #in gd_agnet.py

        self.opt_AAG = kwargs.get('AAG', False)
        
        self.opt_GAA = kwargs.get('GAA', False)
        
        self.opt_AGG = kwargs.get('AGG', False)
        
        self.opt_GGA = kwargs.get('GGA', False)
        
        self.opt_AAGG = kwargs.get('AAGG', False)
        
        self.opt_GGAA = kwargs.get('GGAA', False)
        
        self.opt_Voss = kwargs.get('Voss', False)
        
        self.opt_diag_Voss = kwargs.get('diagVoss', False)        
        
        
        self.VN = kwargs.get('VN',None)
        
        self.choose_which_poly = kwargs.get('set_Poly',1)
        
        self.set_bezier_ctrlp_fairness = kwargs.get('weight_CtrlP',0.005)
        
        self.set_smooth_vertices_fairness = kwargs.get('weight_SmoothVertex',0.005)
        #self.set_unroll_strip_fairness = kwargs.get('weight_UnrollStrips',0.005)
        
        self.switch_interpolate_checker = kwargs.get('is_Checker', True) ##NOTE: CHECKER VERTICES OR ALL-INNER

        self.num_checker_interval = kwargs.get('num_Checker', 4)
        
        self.switch_if_ruling_dense = kwargs.get('is_DenserRuling', False) ##NOTE: SPARSE OR DENSE

        self.switch_if_ruling_rectify = kwargs.get('is_RulingRectify', False)
        
        self.switch_unroll_midaxis = kwargs.get('is_UnrollMid', True) ##NOTE: UNROLL ALONG MIDPOINT OR STARTINGPOINT

        self.num_bezier_divide = kwargs.get('num_DenserRuling', 20)
        
        self.scale_dist_offset = kwargs.get('set_ScaleOffset',0.5)

        self.dist_interval = kwargs.get('set_DistInterval',1.5)

    # -------------------------------------------------------------------------
    #                                Properties
    # -------------------------------------------------------------------------
    @property
    def type(self):
        return 'Asymptotic Geodesic Hybrid Networks'
        
    @property
    def vertexlist(self):
        return self.vertices

    @property
    def facelist(self):
        return self.faces_list()
    
    @property
    def switch_singular_mesh(self):
        _,_,lj = self.vertex_ring_vertices_iterators(return_lengths=True)
        order = np.where(lj>4)[0]
        if len(order)!=0:
            return True
        return False

    # -------------------------------------------------------------------------
    #                         Visualization
    # -------------------------------------------------------------------------
    
    ##@on_trait_change('show_checker_tian')
    def plot_checker_group_tian_select_vertices(self):
        blue,yel = self.checker_vertex_tian
        Pb,Py = self.vertices[blue],self.vertices[yel]
        return Pb,Py

    ##@on_trait_change('show_1st_geodesic,show_2nd_geodesic')
    def plot_4family_polylines(self):
        "note: include multiple pls, donot find way to remove"
        # v,v1,v2,v3,v4 = self.rr_star[self.ind_rr_star_v4f4].T
        v,va,vb,vc,vd = self.rr_star_corner
        v,v1,v2,v3,v4 = self.rr_star.T
        return np.r_[v1,v],np.r_[v,v3],np.r_[v2,v],np.r_[v,v4],\
            np.r_[va,v],np.r_[v,vc],np.r_[vb,v],np.r_[v,vd]

    def get_vertices_matrix(self,rot=False):
        "Note: problem to work in GH"
        if rot:
            M = self.rot_patch_matrix
        else:
            M = self.patch_matrix
        return M

    def plot_4family_vertices(self,rot=False):
        "Note: problem to work in GH"
        if rot:
            M = self.rot_patch_matrix
        else:
            M = self.patch_matrix
        ipt1,ipt2 = M, M.T
        # ipt3,ipt4 = [],[]
        # def diagnet(M):
        #     ip = []
        #     a,b = M.shape
        #     for i in range(a-1)+1:
        #         j = np.arange(i)
        ipt3,_,_,_ = self.get_diagonal_vertex_list(interval=1,another_direction=True)
        ipt4,_,_,_ = self.get_diagonal_vertex_list(interval=1,another_direction=False)
        return ipt1.tolist(),ipt2.tolist(),ipt3,ipt4

    def plot_2family_polylines(self,rot=False):
        if rot:
            M = self.rot_patch_matrix
        else:
            M = self.patch_matrix
        v1l,v1r = M[:,:-1].flatten(), M[:,1:].flatten()
        v2l,v2r = M[:-1,:].flatten(), M[1:,:].flatten()      
        return v1l,v1r,v2l,v2r

    def plot_2family_diagonal_polylines(self):
        ipt3,_,_,_ = self.get_diagonal_vertex_list(interval=1,another_direction=True)
        ipt4,_,_,_ = self.get_diagonal_vertex_list(interval=1,another_direction=False)
        v1l=v1r=v2l=v2r=np.array([],dtype=int)
        for pl in ipt3:
            "different list length"
            v1l = np.r_[v1l, pl[:-1]]
            v1r = np.r_[v1r, pl[1:]]
        for pl in ipt4:
            "different list length"
            v2l = np.r_[v2l, pl[:-1]]
            v2r = np.r_[v2r, pl[1:]]
        return v1l,v1r,v2l,v2r

    def plot_boundary_polylines(self,i):
        v,_ = self.get_i_boundary_vertex_indices(i)
        return v[:-1],v[1:]
    
    def plot_corner_vertices(self):
        v = self.corner
        return v.tolist(),self.vertices[v]
    
    def plot_selected_vertices(self,Pi):
        "from Rhino/Grasshopper vertex return the mesh vertex"
        index = self.closest_vertices(Pi)
        return index.tolist(), self.vertices[index]
        
    def _set_ith_polylines(self):
        if self.choose_which_poly==1:
            "1st isoline"
            diagpoly = False
            is_another_poly = False
            if self.opt_AAG:
                asym_or_geo=True
            elif self.opt_GAA:
                asym_or_geo=None if self.set_another_polyline else False
            elif self.opt_GGA:
                asym_or_geo=False
            elif self.opt_AGG:
                asym_or_geo=None if self.set_another_polyline else True
            elif self.opt_AAGG:
                asym_or_geo=True
            elif self.opt_GGAA:
                asym_or_geo=False
            elif self.opt_Anet:
                asym_or_geo=True    
            elif self.opt_Gnet:
                asym_or_geo=False
            
        elif self.choose_which_poly==2:
            "2nd isoline"
            diagpoly = False
            is_another_poly = True
            if self.opt_AAG:
                asym_or_geo=True
            elif self.opt_GAA:
                asym_or_geo=False if self.set_another_polyline else None
            elif self.opt_GGA:
                asym_or_geo=False
            elif self.opt_AGG:
                asym_or_geo=True if self.set_another_polyline else None
            elif self.opt_AAGG:
                asym_or_geo=True
            elif self.opt_GGAA:
                asym_or_geo=False
            elif self.opt_Anet:
                asym_or_geo=True    
            elif self.opt_Gnet:
                asym_or_geo=False
        elif self.choose_which_poly==3:
            "3rd diagonal"
            diagpoly = True
            is_another_poly = False
            if self.opt_AAG or self.switch_singular_mesh:
                asym_or_geo=None if self.set_another_polyline else False
            elif self.opt_GAA:
                asym_or_geo=True
            elif self.opt_GGA:
                asym_or_geo=None if self.set_another_polyline else True
            elif self.opt_AGG:
                asym_or_geo=False
            elif self.opt_AAGG:
                asym_or_geo=False
            elif self.opt_GGAA:
                asym_or_geo=True
            elif self.opt_Anet_diagnet:
                asym_or_geo=True    
            elif self.opt_Gnet_diagnet:
                asym_or_geo=False
        elif self.choose_which_poly==4:
            "4th diagonal"
            diagpoly = True
            is_another_poly = True
            if self.opt_AAG:
                asym_or_geo=False if self.set_another_polyline else None
            elif self.opt_GAA:
                asym_or_geo=True
            elif self.opt_GGA:
                asym_or_geo=True if self.set_another_polyline else None
            elif self.opt_AGG:
                asym_or_geo=False
            elif self.opt_AAGG:
                asym_or_geo=False
            elif self.opt_GGAA:
                asym_or_geo=True
            elif self.opt_Anet_diagnet:
                asym_or_geo=True    
            elif self.opt_Gnet_diagnet:
                asym_or_geo=False    
        return asym_or_geo,diagpoly,is_another_poly

    def _set_asy_or_geo_net(self,ctrlnet_or_diagnet=True):
        if ctrlnet_or_diagnet:
            if self.opt_AAG or self.switch_singular_mesh:
                asym_or_geo,diagpoly=True,False
            elif self.opt_GGA:
                asym_or_geo,diagpoly=False,False
            elif self.opt_AAGG:
                asym_or_geo,diagpoly=True,False
            elif self.opt_GGAA:
                asym_or_geo,diagpoly=False,False
            elif self.Anet:
                asym_or_geo,diagpoly=True,False   
            elif self.Gnet:
                asym_or_geo,diagpoly=False,False
        else:
            if self.opt_GAA:
                asym_or_geo,diagpoly=True,True
            elif self.opt_AGG:
                asym_or_geo,diagpoly=False,True
            elif self.opt_AAGG:
                asym_or_geo,diagpoly=False,True
            elif self.opt_GGAA:
                asym_or_geo,diagpoly=True,True
            elif self.Anet_diagnet:
                asym_or_geo,diagpoly=True,True 
            elif self.Gnet_diagnet:
                asym_or_geo,diagpoly=False,True
        return asym_or_geo,diagpoly 
                
    
    ##@on_trait_change('show_rectify_strip_ctrlnet,show_rectify_strip_diagnet')
    def set_quintic_bezier_splines(self):
        asym_or_geo,diagpoly,is_another_poly = self._set_ith_polylines()
        is_one_or_another = not is_another_poly
        
        if not self.switch_singular_mesh:
            ick = int(self.num_checker_interval) if self.switch_interpolate_checker else 1
            P,_,_,_,frame,annr,_ = self.get_poly_quintic_Bezier_spline_crvs_checker(
                self,#self.save_new_mesh, # offset mesh with same geometry
                normal=self.VN,
                efair=self.set_bezier_ctrlp_fairness,
                is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
                is_one_or_another=is_one_or_another,
                is_checker=1,
                interval=ick, ##4 is the num of subdivision quadfaces
                is_dense=self.switch_if_ruling_dense,
                num_divide=self.num_bezier_divide,
                is_modify=self.switch_if_ruling_rectify,
                is_smooth=self.set_smooth_vertices_fairness)
        else:
            from singularMesh import get_singular_quintic_Bezier_spline_crvs_checker
            ick = 3 if self.switch_interpolate_checker else 1
            P,_,_,_,frame,annr,_ = get_singular_quintic_Bezier_spline_crvs_checker(
                self,
                normal=self.VN,
                efair=self.set_bezier_ctrlp_fairness,
                is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
                is_one_or_another=is_one_or_another,
                is_checker=ick, ##2,3 is the num of subdivision quadfaces
                is_dense=self.switch_if_ruling_dense,
                num_divide=self.num_bezier_divide,
                is_smooth=self.set_smooth_vertices_fairness)
    
        v,an,r,arr,nmlist,denselist = annr
        mlist=denselist if self.switch_if_ruling_dense else nmlist
        _,E1,E2,E3 = frame
        cos = np.abs(np.einsum('ij,ij->i',E3,r))
        if asym_or_geo and not self.switch_if_ruling_dense:
            Nv = - self.vertex_normals()[v] ##Note sign
            i = np.where(np.einsum('ij,ij->i',Nv,r)<0)[0]
            r[i] = -r[i]  
        if self.switch_if_ruling_rectify and not self.switch_if_ruling_dense:
            cos = np.abs(np.einsum('ij,ij->i',E3,r))
            i = np.where(cos<np.mean(cos))[0]
            r[i] = E3[i]
    
        dist = self.mean_edge_length() * self.scale_dist_offset  
        is_smooth = self.set_smooth_vertices_fairness
        if self.switch_if_ruling_rectify:
            if asym_or_geo:
                "strip bounded by asym.crv."
                unitN = r#E3
                centerline_symmetric = False
            else:
                "strip's center line is geodesic"
                unitN = E3
                an = an-E3*dist/2
                centerline_symmetric = True
            if True:
                "more general case"
                sm = self.get_strip_from_rulings(an,unitN*dist,mlist,is_smooth)
            else:
                "getting the developable strip"
                from developablestrip import StraightStrip
                num_seg = self.num_bezier_divide
                strip = StraightStrip(an, unitN, E1, r, num_seg, dist, mlist,
                                    efair=is_smooth,  # self.set_ply_rectify_strip_fairness,
                                    itera=50, ee=0.001)
                sm = strip.get_strip(centerline_symmetric)   
        else:
            if asym_or_geo:
                "strip bounded by asym.crv."
                xrr = dist / cos   
                unitN = r * xrr[:,None]
                centerline_symmetric = False
            else:
                centerline_symmetric = True
                if True:
                    "strip's center line is geodesic"
                    xrr = dist / cos / 2
                    an = an-r*xrr[:,None]
                    unitN = r*xrr[:,None]*2
                else:
                    "not centerline but boundaryline"
                    xrr = dist / cos   
                    unitN = r * xrr[:,None]
            "more general case"
            sm = self.get_strip_from_rulings(an,unitN,mlist,is_smooth)   

        "getting the unrolling strip:"
        from unroll import unroll_multiple_strips
        unm = unroll_multiple_strips(sm,mlist,dist,
                                      step=self.dist_interval,coo=2,
                                      anchor=0,efair=is_smooth,
                                      itera=100,
                                      is_midaxis=self.switch_unroll_midaxis,
                                      w_straight=is_smooth)

        return P,nmlist,an,E1,E2,E3,r,sm,unm