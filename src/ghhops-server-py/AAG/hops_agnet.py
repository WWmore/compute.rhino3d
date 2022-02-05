#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import os

# import sys

# path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sys.path.append(path)

from guidedprojection_agnet import GuidedProjection_AGNet

#------------------------------------------------------------------------------

''' optimization used for AAG-project, linking to Grasshopper by Hops  
'''
__author__ = 'Hui Wang'

import numpy as np
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                        GuidedProjection
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class AGNet(GuidedProjection_AGNet):
    
    # -------------------------------------------------------------------------
    #                                Initialize
    # -------------------------------------------------------------------------
    def __init__(self,**kwargs):
        
        GuidedProjection_AGNet.__init__(self)
        
        self.threshold = 1e-20
       
        self.epsilon = 0.001
        
        self.step = 1

        self.iter = int(float(kwargs.get('num_itera', 5)))
        
        self.weights['mesh_fairness'] = kwargs.get('weight_fairness', 0.005)
        self.weights['boundary_fairness'] = kwargs.get('weight_fairness', 0.005)
        #self.weights['corner_fairness'] = kwargs.get('weight_fairness', 0.005)
        self.weights['tangential_fairness'] = kwargs.get('weight_fairness', 0.005)
        self.weights['spring_fairness'] = kwargs.get('weight_fairness', 0.005)
    
        self.weights['self_closeness'] = kwargs.get('weight_closeness', 0)
        
        self.weights['fix_point'] = kwargs.get('weight_fix', 0)
        self.ind_fixed_point = kwargs.get('ifixV', None)##NOTE: index-number
        
        self.weights['i_boundary_glide'] = kwargs.get('weight_gliding', 0)
        self.ind_glide_boundary = kwargs.get('iGlideBdry', None) ##NOTE:1,2for rot, 1,2,3,4 for rectangular-shape

        self.weight_checker = kwargs.get('weight_checker', 1) ##NOTE: NEED TO CHECK
        
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

        self.restart = kwargs.get('Restart', True)
        
        self.vertexlist = None
        self.facelist = None

    # -------------------------------------------------------------------------
    #                                Properties
    # -------------------------------------------------------------------------
    @property
    def type(self):
        return 'Asymptotic Geodesic Hybrid Networks'

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
        #self.initialization()

    @property
    def switch_singular_mesh(self):
        _,_,lj = self.mesh.vertex_ring_vertices_iterators(return_lengths=True)
        order = np.where(lj>4)[0]
        if len(order)!=0:
            return True
        return False   

    # -------------------------------------------------------------------------
    #                         Reset + Optimization
    # -------------------------------------------------------------------------
    
    #### OPTIMIZATION MAIN FUNCTION ------------------
    def optimize_mesh(self):
        if self.restart:
            self.do_settings()
            self.mesh.vertices = self.mesh.vertices_0
            self.is_initial = True 
            self.reinitialize = True
            self.initialization()
            print("---restart---\n")

        for i in range(self.iter):
            X = self.optimize() #in gpbase.py
        ver = X[:3*self.mesh.V].reshape(-1,3,order='F')
        self.vertexlist = ver#self.mesh.vertices
        self.facelist = self.mesh.faces_list()

    #### ---------------------------------------------    

    def do_settings(self):
        self._settings_noconstraints()
        
        if self.switch_singular_mesh:
            self.is_singular = True
            self.set_weight('Anet', 1)
            self.set_weight('AAG_singular', 1)
        else:
            if self.opt_Anet:
                self.switch_diagmeth = False
                self.set_weight('Anet', 1)
            elif self.opt_Anet_diagnet:
                self.switch_diagmeth = True
                self.set_weight('Anet_diagnet', 1)
            elif self.opt_Gnet:
                self.switch_diagmeth = False
                self.set_weight('Gnet', 1)
            elif self.opt_Gnet_diagnet:
                self.switch_diagmeth = True
                self.set_weight('Gnet_diagnet', 1)
            if self.opt_AAG:
                self.switch_diagmeth = False
                self.set_weight('Anet', 1)
                self.set_weight('AAGnet', 1)
                #print('AAGnet: Anet + diagGeodesic')
            elif self.opt_GAA:
                self.switch_diagmeth = True
                self.set_weight('Anet_diagnet', 1)
                self.set_weight('GAAnet', 1)
               # print('GAAnet: Geodesic + diagAnet')
            elif self.opt_GGA:
                self.switch_diagmeth = False
                self.set_weight('Gnet', 1) 
                self.set_weight('GGAnet', 1)
                #print('GGAnet: Gnet + diagAsymptotic')
            elif self.opt_AGG:
                self.switch_diagmeth = True
                self.set_weight('Gnet_diagnet', 1)
                self.set_weight('AGGnet', 1)
                #print('AGGnet: Asymptotic + diagGnet')
            elif self.opt_AAGG:
                self.switch_diagmeth = False
                self.set_weight('Anet', 1)
                self.set_weight('Gnet_diagnet', 1)
                self.set_weight('AAGGnet', 1)
                #print('AAGGnet: Anet + diagGnet')
            elif self.opt_GGAA:
                self.switch_diagmeth = True
                self.set_weight('Gnet', 1) 
                self.set_weight('Anet_diagnet', 1)
                self.set_weight('GGAAnet', 1)
                #print('GGAAnet: Gnet + diagAnet')
            elif self.opt_Voss:
                "if GGAA, it is Voss-ctrol"
                self.set_weight('geometric', 1) 
                self.set_weight('planarity', 1) 
                self.set_weight('Gnet', 1) 
                self.set_weight('Anet_diagnet', 1)
            elif self.opt_diag_Voss: #TODO
                "if AAGG, it is Voss-diag"
                self.set_weight('Gnet_diagnet', 1)
                self.set_weight('Anet', 1)

            if self.get_weight('i_boundary_glide') and self.ind_glide_boundary is not None:
                from polyline import Polyline
                self.i_glide_bdry_crv = []
                self.i_glide_bdry_ver = []
                N = 5
                if 1 in self.ind_glide_boundary:
                    v,B = self.mesh.get_i_boundary_vertex_indices(i=0)
                    poly = Polyline(B,closed=False)  
                    poly.refine(steps=N)
                    self.i_glide_bdry_crv.append(poly)
                    self.i_glide_bdry_ver.append(v)
                if 2 in self.ind_glide_boundary:
                    v,B = self.mesh.get_i_boundary_vertex_indices(i=1)
                    poly = Polyline(B,closed=False)  
                    poly.refine(steps=N)
                    self.i_glide_bdry_crv.append(poly)
                    self.i_glide_bdry_ver.append(v)
                if 3 in self.ind_glide_boundary:
                    v,B = self.mesh.get_i_boundary_vertex_indices(i=2)
                    poly = Polyline(B,closed=False)  
                    poly.refine(steps=N)
                    self.i_glide_bdry_crv.append(poly)
                    self.i_glide_bdry_ver.append(v)
                if 4 in self.ind_glide_boundary:
                    v,B = self.mesh.get_i_boundary_vertex_indices(i=3)
                    poly = Polyline(B,closed=False)  
                    poly.refine(steps=N)
                    self.i_glide_bdry_crv.append(poly)
                    self.i_glide_bdry_ver.append(v)
            if self.get_weight('fix_point'):
                self.fixed_value = self.mesh.vertices_0[self.ind_fixed_point].flatten('F')
                
            
    def _settings_noconstraints(self):
        self.set_weight('AAG_singular', 0)
        self.set_weight('Anet', 0)
        self.set_weight('Anet_diagnet', 0)
        self.set_weight('Gnet', 0)
        self.set_weight('Gnet_diagnet', 0)
        self.set_weight('AAGnet', 0)
        self.set_weight('GAAnet', 0)
        self.set_weight('AGGnet', 0)
        self.set_weight('GGAnet', 0)
        self.set_weight('AAGGnet', 0)
        self.set_weight('GGAAnet', 0)
        self.set_weight('geometric', 0) 
        self.set_weight('planarity', 0)
        
        
    def get_agweb_an_n_on(self):
        V = self.mesh.vertices
        v = self.mesh.rr_star_corner[0]
        n = self.mesh.vertex_normals()[v]
        num = len(self.mesh.ind_rr_star_v4f4)
        if not self.is_initial:
            X = self.X
            if self.opt_AAG or self.opt_GAA:
                "X=+[oNg]"
                v = v[self.mesh.ind_rr_star_v4f4]
                n = X[self._Nanet-3*num:self._Nanet].reshape(-1,3,order='F') 
            elif self.opt_GGA or self.opt_AGG:
                d = self._Ndgeo-9*num
                n = X[d:d+3*num].reshape(-1,3,order='F')
            elif self.opt_AAGG or self.opt_GGAA:
                v = v[self.mesh.ind_rr_star_v4f4]
                n = X[self._Nanet-3*num:self._Nanet].reshape(-1,3,order='F') 
            elif self.opt_Anet:
                v = v[self.mesh.ind_rr_star_v4f4]
                n = X[self._Nanet-3*num:self._Nanet].reshape(-1,3,order='F')      
        alln = self.mesh.vertex_normals()
        alln[v] = n
        return V,alln