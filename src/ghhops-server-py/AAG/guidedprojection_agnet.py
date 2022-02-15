#!/usr/bin/env python

# -*- coding: utf-8 -*-

import numpy as np
# -----------------------------------------------------------------------------
from guidedprojectionbase import GuidedProjectionBase
# -----------------------------------------------------------------------------

# try:
from constraints_basic import columnnew,con_planarity_constraints,\
    con_isometry

from constraints_net import con_unit_edge,con_orthogonal_midline,\
    con_isogonal,con_isogonal_diagnet,\
    con_anet,con_anet_diagnet,con_gnet,con_gnet_diagnet,\
    con_anet_geodesic,con_polyline_ruling,con_osculating_tangents,\
    con_planar_1familyof_polylines,con_nonsquare_quadface,\
    con_singular_Anet_diag_geodesic,con_gonet, \
    con_diag_1_asymptotic_or_geodesic,\
    con_ctrlnet_symmetric_1_diagpoly, con_AGnet

from singularMesh import quadmesh_with_1singularity
from constraints_glide import con_alignment,con_alignments,con_fix_vertices
# -----------------------------------------------------------------------------
__author__ = 'Hui Wang'
# -----------------------------------------------------------------------------

class GuidedProjection_AGNet(GuidedProjectionBase):
    _N1 = 0
    _N5 = 0
    _N6 = 0
    
    _Nanet = 0
    _Ndgeo = 0
    _Ndgeoliou = 0
    _Ndgeopc = 0
    _Nruling = 0
    _Noscut = 0
    _Nnonsym = 0
    _Npp = 0
    _Ncd = _Ncds = 0
    _Nag = 0
    
    def __init__(self):
        GuidedProjectionBase.__init__(self)

        weights = {
            
        ## Commen setting:
        'geometric' : 0, ##NOTE SHOULD BE 1 ONCE planarity=1
        'planarity' : 0,

        ## shared used:
        'unit_edge' : 0, 
        'unit_diag_edge' : 0,
        
        'orthogonal' :0,
        'isogonal' : 0,
        'isogonal_diagnet' :0,
        
        'Anet' : 0,  
        'Anet_diagnet' : 0,  
        
        'Gnet' : 0, 
        'Gnet_diagnet' : 0, 
        
        'GOnet' : 0, 
        
        'diag_1_asymptotic': 0,
        'diag_1_geodesic': 0,
        
        'ctrlnet_symmetric_1diagpoly': 0,
        
        'nonsymmetric' :0,
        
        'isometry' : 0,
        
        'z0' : 0,
        
        'boundary_glide' :0, #Hui in gpbase.py doesn't work, replace here.
        'i_boundary_glide' :0,
        'fix_point' :0,
        
        ## only for AGNet:
        'GGGnet': 0,
        'GGG_diagnet': 0, #TODO
        
        'AGnet': 0,
        
        'AAGnet': 0,
        'GAAnet': 0,
        'GGAnet': 0,
        'AGGnet': 0,
        'AAGGnet': 0,
        'GGAAnet': 0,
        
        
        'planar_geodesic' : 0,
        'agnet_liouville': 0,
        'ruling': 0,# opt for ruling quadratic mesh, straight lines-mesh     
        'oscu_tangent' :0,

        'AAG_singular' :0,
        
        'planar_ply1' : 0,
        'planar_ply2' : 0,
        
        }

        self.add_weights(weights)
        
        self.switch_diagmeth = False
        
        self.is_initial = True
        
        self.if_angle = False
        self._angle = 90
        
        self._glide_reference_polyline = None
        self.i_glide_bdry_crv, self.i_glide_bdry_ver = [],[]
        
        self.ind_fixed_point, self.fixed_value = None,None
        
        self.set_another_polyline = 0
        
        self._ver_poly_strip1,self._ver_poly_strip2 = None,None
        
        self.nonsym_eps = 0.01
        self.ind_nonsym_v124,self.ind_nonsym_l12 = None,None
        
        self.is_singular = False
        self._singular_polylist = None
        self._ind_rr_vertex = None
        self.weight_checker = 1
        
        ### isogonal AGnet:
        self.is_AG_or_GA = True
        self.opt_AG_ortho = False
        self.opt_AG_const_rii = False
        self.opt_AG_const_r0 = False
        
    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
        self.initialization()

    @property
    def max_weight(self):    
        return max(self.get_weight('boundary_glide'),
                   self.get_weight('i_boundary_glide'),
                   self.get_weight('geometric'),
                   self.get_weight('planarity'),
                   self.get_weight('unit_edge'),
                   self.get_weight('unit_diag_edge'),
                   self.get_weight('orthogonal'),
                   self.get_weight('isometry'),
                   
                   self.get_weight('oscu_tangent'),
                   self.get_weight('Anet'),
                   self.get_weight('Anet_diagnet'),
                   
                   self.get_weight('diag_1_asymptotic'), #n defined from only ctrl-net
                   self.get_weight('diag_1_geodesic'),
                   
                   self.get_weight('ctrlnet_symmetric_1diagpoly'),
                   
                   self.get_weigth('nonsymmetric'),
                   
                   self.get_weight('AAG_singular'),
                   
                   self.get_weight('planar_plys'),
                   1)
    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self,angle):
        if angle != self._angle:
            self.mesh.angle=angle
        self._angle = angle
        
    @property
    def glide_reference_polyline(self):
        if self._glide_reference_polyline is None:
            polylines = self.mesh.boundary_curves(corner_split=False)[0]   
            N = 5
            for polyline in polylines:
                polyline.refine(N)
            self._glide_reference_polyline = polyline
        return self._glide_reference_polyline

    # @glide_reference_polyline.setter##NOTE: used by reference-mesh case
    # def glide_reference_polyline(self,polyline):
    #     self._glide_reference_polyline = polyline        
        
    @property
    def ver_poly_strip1(self):
        if self._ver_poly_strip1 is None:
            if self.get_weight('planar_ply1') or self.opt_AG_const_rii:
                self.index_of_mesh_polylines()
            else:
                self.index_of_strip_along_polyline()
        return self._ver_poly_strip1  
    
    @property
    def ver_poly_strip2(self):
        if self._ver_poly_strip2 is None:
            if self.get_weight('planar_ply2'):
                self.index_of_mesh_polylines()
        return self._ver_poly_strip2  

    @property
    def singular_polylist(self):
        if self._singular_polylist is None:
            self.get_singularmesh_diagpoly()
        return self._singular_polylist

    @property
    def ind_rr_vertex(self):
        if self._ind_rr_vertex is None:
            self.get_singularmesh_diagpoly()
        return self._ind_rr_vertex  
    
    #--------------------------------------------------------------------------
    #                               Initialization
    #--------------------------------------------------------------------------

    def set_weights(self):
        #------------------------------------
        if self.get_weight('isogonal'): 
            self.set_weight('unit_edge', 1*self.get_weight('isogonal'))
        elif self.get_weight('isogonal_diagnet'): 
            self.set_weight('unit_diag_edge', 1*self.get_weight('isogonal_diagnet')) 
            
        if self.get_weight('Gnet') or self.get_weight('GOnet'):  
            self.set_weight('unit_edge', 1)
        elif self.get_weight('Gnet_diagnet'):  
            self.set_weight('unit_diag_edge', 1)  
 
        if self.get_weight('GGGnet'):
            self.set_weight('Gnet', 1)
            self.set_weight('diag_1_geodesic',1)
            
        if self.get_weight('AAGnet'):
            self.set_weight('Anet', 1)
        elif self.get_weight('GAAnet'):
            self.set_weight('Anet_diagnet', 1) 
        elif self.get_weight('GGAnet'):
            self.set_weight('Gnet', 1)
        elif self.get_weight('AGGnet'):
            self.set_weight('Gnet_diagnet', 1)
        elif self.get_weight('AAGGnet'):
            self.set_weight('Anet', 1)
            self.set_weight('Gnet_diagnet', 1)
        elif self.get_weight('GGAAnet'):
            self.set_weight('Gnet', 1)
            self.set_weight('Anet_diagnet', 1)
            
        if self.get_weight('AGnet'): 
            self.set_weight('oscu_tangent', self.get_weight('AGnet'))    
            
        if self.get_weight('AAG_singular'):
            self.set_weight('Anet', 1*self.get_weight('AAG_singular'))
        if self.get_weight('diag_1_asymptotic') or self.get_weight('diag_1_geodesic'):
            self.set_weight('unit_edge',1)

        if self.get_weight('ctrlnet_symmetric_1diagpoly'):
            pass
        #--------------------------------------
          
    def set_dimensions(self): # Huinote: be used in guidedprojectionbase
        V = self.mesh.V
        F = self.mesh.F
        num_regular = self.mesh.num_regular
        N  = 3*V
        N1 = N5 = N6 = N
        Nanet = N
        Ndgeo = Ndgeoliou = Ndgeopc = Nruling = Noscut = N
        Nnonsym = N
        Npp = N
        Ncd = Ncds = N
        Nag = N
        #---------------------------------------------
        if self.get_weight('planarity'):
            N += 3*F
            N1 = N
        if self.get_weight('unit_edge'): #Gnet
            "le1,le2,le3,le4,ue1,ue2,ue3,ue4 "
            if self.get_weight('isogonal'):
                N += 16*num_regular
            else:
                "for Anet"
                N += 16*len(self.mesh.ind_rr_star_v4f4)
            N5 = N
        elif self.get_weight('unit_diag_edge'): #Gnet_diagnet
            "le1,le2,le3,le4,ue1,ue2,ue3,ue4 "
            N += 16*len(self.mesh.ind_rr_star_v4f4)
            N5 = N
            
        if self.get_weight('isogonal'):
            "lt1,lt2, ut1,ut2, cos0"
            N += 8*num_regular+1
            N6 = N
        elif self.get_weight('isogonal_diagnet'):
            "lt1,lt2, ut1,ut2, cos0"
            N += 8*len(self.mesh.ind_rr_star_v4f4)+1
            N6 = N       

        if self.get_weight('Anet') or self.get_weight('Anet_diagnet'):
            N += 3*len(self.mesh.ind_rr_star_v4f4)#3*num_regular
            Nanet = N
            
        if self.get_weight('AAGnet') or self.get_weight('GAAnet'):
            N += 3*len(self.mesh.ind_rr_star_v4f4)
            Ndgeo = N
        elif self.get_weight('GGAnet') or self.get_weight('AGGnet'):
            N += 9*len(self.mesh.ind_rr_star_v4f4)
            Ndgeo = N
        elif self.get_weight('AAGGnet') or self.get_weight('GGAAnet'):
            N += 6*len(self.mesh.ind_rr_star_v4f4)
            Ndgeo = N   
            
        if self.get_weight('oscu_tangent'):
            "X +=[ll1,ll2,ll3,ll4,lu1,lu2,u1,u2]"
            N += 12*len(self.mesh.ind_rr_star_v4f4)
            Noscut = N      
            
        if self.get_weight('AGnet'):
            "osculating tangents; X += [surfN; ogN]; if const.ri, X+=[Ri]"
            N += 6*len(self.mesh.ind_rr_star_v4f4)
            if self.opt_AG_const_rii:
                "const rii for each geodesic polylines, default v2-v-v4"
                N += len(self.ver_poly_strip1)#TODO
            elif self.opt_AG_const_r0:
                "unique r"
                N += 1
            Nag = N

        if self.get_weight('agnet_liouville'):
            "X +=[lu1,tu1; lla,llc,g1, lg1,tg1, c]"
            N += 13*len(self.mesh.ind_rr_star_v4f4) +1 
            Ndgeoliou = N
            
        if self.get_weight('planar_geodesic'):
            N += 3*len(self.ver_poly_strip1[0])
            Ndgeopc = N
        if self.get_weight('ruling'):
            N += 3*len(self.mesh.get_both_isopolyline(self.switch_diagmeth))
            Nruling = N
        if self.get_weight('nonsymmetric'):
            "X += [E,s]"
            N += self.mesh.E + len(self.ind_nonsym_v124[0]) ##self.mesh.num_rrf ##len=self.rr_quadface
            Nnonsym = N
        if self.get_weight('AAG_singular'):
            "X += [on]"
            N += 3*len(self.singular_polylist[1])
            Ndgeo = N
        
        ### PPQ-project:
        if self.get_weight('planar_ply1'):
            N += 3*len(self.ver_poly_strip1)
            ## only for \obj_cheng\every_5_PPQ.obj'
            ##matrix = self.ver_poly_strip1
            #matrix = self.mesh.rot_patch_matrix[:,::5].T
            #N += 3*len(matrix)
            Nppq = N
        if self.get_weight('planar_ply2'):
            N += 3*len(self.ver_poly_strip2)
            Nppo = N   
            
        ### CG / CA project:
        if self.get_weight('diag_1_asymptotic') or self.get_weight('diag_1_geodesic'):
            if self.get_weight('diag_1_asymptotic'):
                "[ln,v_N]"
                N += 4*len(self.mesh.ind_rr_star_v4f4)
            elif self.get_weight('diag_1_geodesic'):
                if self.is_singular:
                    "[ln,v_N;la[ind],lc[ind],ea[ind],ec[ind]]"
                    N += (1+3)*len(self.mesh.ind_rr_star_v4f4)+8*len(self.ind_rr_vertex)
                else:
                    "[ln,v_N;la,lc,ea,ec]"
                    N += (1+3+3+3+1+1)*len(self.mesh.ind_rr_star_v4f4)
            Ncd = N #ctrl-diag net
        if self.get_weight('ctrlnet_symmetric_1diagpoly'):
            N += (1+1+3+3+1+3)*len(self.mesh.ind_rr_star_v4f4) #[e_ac,l_ac]
            Ncds = N
        #---------------------------------------------
        if N1 != self._N1:
            self.reinitialize = True
        if N5 != self._N5 or N6 != self._N6:
            self.reinitialize = True
        if Nanet != self._Nanet:
            self.reinitialize = True
        if Ndgeo != self._Ndgeo:
            self.reinitialize = True
        if Nag != self._Nag:
            self.reinitialize = True
        if Ndgeoliou != self._Ndgeoliou:
            self.reinitialize = True
        if Ndgeopc != self._Ndgeopc:
            self.reinitialize = True
        if Nruling != self._Nruling:
            self.reinitialize = True
        if Noscut != self._Noscut:
            self.reinitialize = True
        if Nnonsym != self._Nnonsym:
            self.reinitialize = True
            
        if Npp != self._Npp:
            self.reinitialize = True
        if Ncd != self._Ncd:
            self.reinitialize = True
        if Ncds != self._Ncds:
            self.reinitialize = True
        #----------------------------------------------
        self._N = N
        self._N1 = N1
        self._N5 = N5
        self._N6 = N6
        self._Nanet = Nanet
        self._Ndgeo = Ndgeo
        self._Ndgeoliou = Ndgeoliou
        self._Ndgeopc = Ndgeopc
        self._Nruling = Nruling
        self._Noscut = Noscut
        self._Nnonsym = Nnonsym
        self._Npp = Npp
        self._Ncd = Ncd
        self._Ncds = Ncds
        self._Nag = Nag
        self.build_added_weight() # Hui add
        
        
    def initialize_unknowns_vector(self):
        X = self.mesh.vertices.flatten('F')
        if self.get_weight('planarity'):
            normals = self.mesh.face_normals()
            normals = normals.flatten('F')
            X = np.hstack((X, normals))       
            
        if self.get_weight('unit_edge'):
            if True:
                "self.get_weight('Gnet')"
                rr=True
            l1,l2,l3,l4,E1,E2,E3,E4 = self.mesh.get_v4_unit_edge(rregular=rr)
            X = np.r_[X,l1,l2,l3,l4]
            X = np.r_[X,E1.flatten('F'),E2.flatten('F'),E3.flatten('F'),E4.flatten('F')]
        elif self.get_weight('unit_diag_edge'):
            l1,l2,l3,l4,E1,E2,E3,E4 = self.mesh.get_v4_diag_unit_edge()
            X = np.r_[X,l1,l2,l3,l4]
            X = np.r_[X,E1.flatten('F'),E2.flatten('F'),E3.flatten('F'),E4.flatten('F')]
            
        if self.get_weight('isogonal'):
            lt1,lt2,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents()
            cos0 = np.mean(np.einsum('ij,ij->i', ut1, ut2))
            X = np.r_[X,lt1,lt2,ut1.flatten('F'),ut2.flatten('F'),cos0]
        elif self.get_weight('isogonal_diagnet'):
            lt1,lt2,ut1,ut2,_,_ = self.mesh.get_v4_diag_unit_tangents()
            cos0 = np.mean(np.einsum('ij,ij->i', ut1, ut2))
            X = np.r_[X,lt1,lt2,ut1.flatten('F'),ut2.flatten('F'),cos0]   

        if self.get_weight('Anet'):
            if True:
                "only r-regular vertex"
                v = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4][:,0]
            else:
                v = self.mesh.ver_regular
            V4N = self.mesh.vertex_normals()[v]
            X = np.r_[X,V4N.flatten('F')]  
            
        elif self.get_weight('Anet_diagnet'):
            v = self.mesh.rr_star_corner[0]
            V4N = self.mesh.vertex_normals()[v]
            X = np.r_[X,V4N.flatten('F')]

        if self.get_weight('AAGnet'):
            on = self.get_agweb_initial(diagnet=False,
                                another_poly_direction=self.set_another_polyline,
                                        AAG=True)
            X = np.r_[X,on]

        elif self.get_weight('GAAnet'):
            on = self.get_agweb_initial(diagnet=True,
                                another_poly_direction=self.set_another_polyline,
                                        AAG=True)
            X = np.r_[X,on] 
        elif self.get_weight('GGAnet'):
            vNoN1oN2 = self.get_agweb_initial(diagnet=False,
                                another_poly_direction=self.set_another_polyline,
                                        GGA=True)
            X = np.r_[X,vNoN1oN2] 
        elif self.get_weight('AGGnet'):
            vNoN1oN2 = self.get_agweb_initial(diagnet=True,
                                another_poly_direction=self.set_another_polyline,
                                        GGA=True)
            X = np.r_[X,vNoN1oN2]  
        elif self.get_weight('AAGGnet'):
            oN1oN2 = self.get_agweb_initial(diagnet=False,
                                another_poly_direction=self.set_another_polyline,
                                        AAGG=True)
            X = np.r_[X,oN1oN2] 
        elif self.get_weight('GGAAnet'):
            oN1oN2 = self.get_agweb_initial(diagnet=True,
                                another_poly_direction=self.set_another_polyline,
                                        AAGG=True)
            X = np.r_[X,oN1oN2]
        
        if self.get_weight('oscu_tangent'):
            "X +=[ll1,ll2,ll3,ll4,lu1,lu2,u1,u2]"
            if self.get_weight('GAAnet') or self.get_weight('AGGnet') or self.get_weight('GGAAnet'):
                diag=True
            else:
                diag=False
            l,t,lt1,lt2 = self.mesh.get_net_osculating_tangents(diagnet=diag)
            [ll1,ll2,ll3,ll4],[lt1,t1],[lt2,t2] = l,lt1,lt2
            X = np.r_[X,ll1,ll2,ll3,ll4]
            X = np.r_[X,lt1,lt2,t1.flatten('F'),t2.flatten('F')] 
            
        if self.get_weight('AGnet'):
            "osculating tangent"
            v,v1,v2,v3,v4 = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4].T
            V = self.mesh.vertices
            _,_,lt1,lt2 = self.mesh.get_net_osculating_tangents()
            srfN = np.cross(lt1[1],lt2[1]) 
            srfN = srfN / np.linalg.norm(srfN,axis=1)[:,None]
            if not self.is_AG_or_GA:
                v2,v4 = v1,v3
            biN = np.cross(V[v2]-V[v], V[v4]-V[v])
            ogN = biN / np.linalg.norm(biN,axis=1)[:,None]
            X = np.r_[X,srfN.flatten('F'),ogN.flatten('F')]
            if self.opt_AG_const_rii:
                "const rii for each geodesic polylines, default v2-v-v4"
                pass #TODO
            elif self.opt_AG_const_r0:
                "unique r"
                from frenet_frame import FrenetFrame
                allr = FrenetFrame(V[v],V[v2],V[v4]).radius
                X = np.r_[X,np.mean(allr)]
            
        if self.get_weight('agnet_liouville'): # no need now
            "X +=[lu1,tu1;  lla,llc,g1, lg1,tg1, c]"
            lulg = self.get_agweb_liouville(diagnet=True)
            X = np.r_[X,lulg]    
        if self.get_weight('planar_geodesic'):
            sn = self.get_poly_strip_normal()
            X = np.r_[X,sn.flatten('F')]  
            
        if self.get_weight('ruling'): # no need now
            sn = self.get_poly_strip_ruling_tangent()
            X = np.r_[X,sn.flatten('F')]      

        if self.get_weight('nonsymmetric'):
            E, s = self.get_nonsymmetric_edge_ratio(diagnet=False)
            X = np.r_[X, E, s]  

        if self.get_weight('AAG_singular'):
            "X += [on]"
            on = self.get_initial_singular_diagply_normal(is_init=True)
            X = np.r_[X,on.flatten('F')]
            
        if self.get_weight('planar_ply1'):
            sn = self.get_poly_strip_normal(pl1=True)
            X = np.r_[X,sn.flatten('F')]
        if self.get_weight('planar_ply2'):
            sn = self.get_poly_strip_normal(pl2=True)
            X = np.r_[X,sn.flatten('F')]

        ### CG / CA project:
        if self.get_weight('diag_1_asymptotic') or self.get_weight('diag_1_geodesic'):
            "X += [ln,uN;la,lc,ea,ec]"
            v,v1,v2,v3,v4 = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4].T
            V = self.mesh.vertices
            v4N = np.cross(V[v3]-V[v1], V[v4]-V[v2])
            ln = np.linalg.norm(v4N,axis=1)
            un = v4N / ln[:,None]
            if self.get_weight('diag_1_asymptotic'):
                "X += [ln,un]"
                X = np.r_[X,ln,un.flatten('F')]
            elif self.get_weight('diag_1_geodesic'):
                "X += [ln,un; la,lc,ea,ec]"
                if self.is_singular:
                    "new, different from below"
                    vl,vc,vr = self.singular_polylist
                    la = np.linalg.norm(V[vl]-V[vc],axis=1)
                    lc = np.linalg.norm(V[vr]-V[vc],axis=1)
                    ea = (V[vl]-V[vc]) / la[:,None]
                    ec = (V[vr]-V[vc]) / lc[:,None]
                    X = np.r_[X,ln,un.flatten('F'),la,lc,ea.flatten('F'),ec.flatten('F')]
                else:
                    "no singular case"
                    l1,l2,l3,l4,E1,E2,E3,E4 = self.mesh.get_v4_diag_unit_edge()
                    if self.set_another_polyline:
                        "switch to another diagonal polyline"
                        ea,ec,la,lc = E2,E4,l2,l4
                    else:
                        ea,ec,la,lc = E1,E3,l1,l3
                    X = np.r_[X,ln,un.flatten('F'),la,lc,ea.flatten('F'),ec.flatten('F')]
       
        if self.get_weight('ctrlnet_symmetric_1diagpoly'):
            "X += [lt1,lt2,ut1,ut2; lac,ud1]"
            lt1,lt2,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents()
            ld1,ld2,ud1,ud2,_,_ = self.mesh.get_v4_diag_unit_tangents()
            if self.set_another_polyline:
                "switch to another diagonal polyline"
                eac,lac = ud2,ld2
            else:
                eac,lac = ud1,ld1
            X = np.r_[X,lt1,lt2,ut1.flatten('F'),ut2.flatten('F')]
            X = np.r_[X,lac,eac.flatten('F')]          
            
        self._X = X
        self._X0 = np.copy(X)
            
        self.build_added_weight() # Hui add
    #--------------------------------------------------------------------------
    #                                Errors strings
    #--------------------------------------------------------------------------
    def make_errors(self):
        self.planarity_error()
        self.isogonal_error()
        self.isogonal_diagnet_error()
        self.anet_error()    
        self.gnet_error()
        self.gonet_error()  
        #self.oscu_tangent_error() # good enough: mean=meax=90
        #self.liouville_error()   

    def planarity_error(self):
        if self.get_weight('planarity') == 0:
            return None
        P = self.mesh.face_planarity()
        Emean = np.mean(P)
        Emax = np.max(P)
        self.add_error('planarity', Emean, Emax, self.get_weight('planarity'))

    def isogonal_error(self):
        if self.get_weight('isogonal') == 0:
            return None
        cos,cos0 = self.unit_tangent_vectors()
        err = np.abs(cos-cos0) # no divided by cos
        emean = np.mean(err)
        emax = np.max(err)
        self.add_error('isogonal', emean, emax, self.get_weight('isogonal'))

    def isogonal_diagnet_error(self):
        if self.get_weight('isogonal_diagnet') == 0:
            return None
        cos,cos0 = self.unit_tangent_vectors_diagnet()
        err = np.abs(cos-cos0) # no divided by cos
        emean = np.mean(err)
        emax = np.max(err)
        self.add_error('isogonal_diagnet', emean, emax, self.get_weight('isogonal_diagnet'))

    def isometry_error(self): # Hui
        "compare all edge_lengths"
        if self.get_weight('isometry') == 0:
            return None
        L = self.edge_lengths_isometry()
        L0 = self.edge_lengths_isometry(initialized=True)
        norm = np.mean(L)
        Err = np.abs(L-L0) / norm
        Emean = np.mean(Err)
        Emax = np.max(Err)
        self.add_error('isometry', Emean, Emax, self.get_weight('isometry'))
        
    def anet_error(self):
        if self.get_weight('Anet') == 0 and self.get_weight('Anet_diagnet')==0:
            return None
        if self.get_weight('Anet'):
            name = 'Anet'
            if True:
                star = self.mesh.rr_star
                v,v1,v2,v3,v4 = star[self.mesh.ind_rr_star_v4f4].T
            else:
                v,v1,v2,v3,v4 = self.mesh.ver_regular_star.T
        elif self.get_weight('Anet_diagnet'):
            name = 'Anet_diagnet'
            v,v1,v2,v3,v4 = self.mesh.rr_star_corner
            
        if self.is_initial:    
            Nv = self.mesh.vertex_normals()[v]
        else:
            num = len(v)
            c_n = self._Nanet-3*num+np.arange(3*num)
            Nv = self.X[c_n].reshape(-1,3,order='F')        
        V = self.mesh.vertices
        err1 = np.abs(np.einsum('ij,ij->i',Nv,V[v1]-V[v]))
        err2 = np.abs(np.einsum('ij,ij->i',Nv,V[v2]-V[v]))
        err3 = np.abs(np.einsum('ij,ij->i',Nv,V[v3]-V[v]))
        err4 = np.abs(np.einsum('ij,ij->i',Nv,V[v4]-V[v]))
        Err = err1+err2+err3+err4
        Emean = np.mean(Err)
        Emax = np.max(Err)
        self.add_error(name, Emean, Emax, self.get_weight(name))  

    def gnet_error(self):
        if self.get_weight('Gnet') == 0 and self.get_weight('Gnet_diagnet')==0:
            return None
        
        if self.get_weight('Gnet'):
            name = 'Gnet'
            if True:
                star = self.mesh.rr_star
                v,v1,v2,v3,v4 = star[self.mesh.ind_rr_star_v4f4].T
            else:
                v,v1,v2,v3,v4 = self.mesh.ver_regular_star.T
        elif self.get_weight('Gnet_diagnet'):
            name = 'Gnet_diagnet'
            v,v1,v2,v3,v4 = self.mesh.rr_star_corner
        
        V = self.mesh.vertices
        E1 = (V[v1]-V[v]) / np.linalg.norm(V[v1]-V[v],axis=1)[:,None]
        E2 = (V[v2]-V[v]) / np.linalg.norm(V[v2]-V[v],axis=1)[:,None]
        E3 = (V[v3]-V[v]) / np.linalg.norm(V[v3]-V[v],axis=1)[:,None]
        E4 = (V[v4]-V[v]) / np.linalg.norm(V[v4]-V[v],axis=1)[:,None]

        err1 = np.abs(np.einsum('ij,ij->i',E1,E2)-np.einsum('ij,ij->i',E3,E4))
        err2 = np.abs(np.einsum('ij,ij->i',E2,E3)-np.einsum('ij,ij->i',E4,E1))
        Err = err1+err2
        Emean = np.mean(Err)
        Emax = np.max(Err)
        self.add_error(name, Emean, Emax, self.get_weight(name))  

    def gonet_error(self):
        if self.get_weight('GOnet') == 0:
            return None
        name = 'GOnet'
        if True:
            star = self.mesh.rr_star
            v,v1,v2,v3,v4 = star[self.mesh.ind_rr_star_v4f4].T
        else:
            v,v1,v2,v3,v4 = self.mesh.ver_regular_star.T

        V = self.mesh.vertices
        E1 = (V[v1]-V[v]) / np.linalg.norm(V[v1]-V[v],axis=1)[:,None]
        E2 = (V[v2]-V[v]) / np.linalg.norm(V[v2]-V[v],axis=1)[:,None]
        E3 = (V[v3]-V[v]) / np.linalg.norm(V[v3]-V[v],axis=1)[:,None]
        E4 = (V[v4]-V[v]) / np.linalg.norm(V[v4]-V[v],axis=1)[:,None]
        if self.is_AG_or_GA:
            err1 = np.abs(np.einsum('ij,ij->i',E1,E2)-np.einsum('ij,ij->i',E2,E3))
            err2 = np.abs(np.einsum('ij,ij->i',E3,E4)-np.einsum('ij,ij->i',E4,E1))
        else:
            err1 = np.abs(np.einsum('ij,ij->i',E1,E2)-np.einsum('ij,ij->i',E1,E4))
            err2 = np.abs(np.einsum('ij,ij->i',E2,E3)-np.einsum('ij,ij->i',E3,E4))
        Err = err1+err2
        Emean = np.mean(Err)
        Emax = np.max(Err)
        self.add_error(name, Emean, Emax, self.get_weight(name))  
        
    def oscu_tangent_error(self):
        if self.get_weight('oscu_tangent') == 0:
            return None
        if self.get_weight('GAAnet') or self.get_weight('AGGnet') or self.get_weight('GGAAnet'):
            diag=True
        else:
            diag=False
        angle = self.mesh.get_net_osculating_tangents(diagnet=diag,printerr=True)        
        emean = '%.2f' % np.mean(angle)
        emax = '%.2f' % np.max(angle)
        print('ortho:',emean,emax)
        #self.add_error('orthogonal', emean, emax, self.get_weight('oscu_tangent'))        

    def liouville_error(self):
        if self.get_weight('agnet_liouville') == 0:
            return None
        cos,cos0 = self.agnet_liouville_const_angle()
        err = np.abs(cos-cos0) # no divided by cos
        emean = np.mean(err)
        emax = np.max(err)
        self.add_error('Liouville', emean, emax, self.get_weight('agnet_liouville'))   

    def planarity_error_string(self):
        return self.error_string('planarity')

    def isogonal_error_string(self):
        return self.error_string('isogonal')
    
    def isogonal_diagnet_error_string(self):
        return self.error_string('isogonal_diagnet')

    def isometry_error_string(self):
        return self.error_string('isometry')
    
    def anet_error_string(self):
        return self.error_string('Anet')

    def liouville_error_string(self):
        return self.error_string('agnet_liouville')
    
    #--------------------------------------------------------------------------
    #                       Getting (initilization + Plotting):
    #--------------------------------------------------------------------------
    def unit_tangent_vectors(self, initialized=False):
        if self.get_weight('isogonal') == 0:
            return None
        if initialized:
            X = self._X0
        else:
            X = self.X
        N6 = self._N6
        num = self.mesh.num_regular
        ut1 = X[N6-6*num-1:N6-3*num-1].reshape(-1,3,order='F')
        ut2 = X[N6-3*num-1:N6-1].reshape(-1,3,order='F')
        cos = np.einsum('ij,ij->i',ut1,ut2)
        cos0 = X[N6-1]
        return cos,cos0

    def unit_tangent_vectors_diagnet(self, initialized=False):
        if self.get_weight('isogonal_diagnet') == 0:
            return None
        if initialized:
            X = self._X0
        else:
            X = self.X
        N6 = self._N6
        num = len(self.mesh.ind_rr_star_v4f4)
        ut1 = X[N6-6*num-1:N6-3*num-1].reshape(-1,3,order='F')
        ut2 = X[N6-3*num-1:N6-1].reshape(-1,3,order='F')
        cos = np.einsum('ij,ij->i',ut1,ut2)
        cos0 = X[N6-1]
        return cos,cos0
    
    def edge_lengths_isometry(self, initialized=False): # Hui
        "isometry: keeping all edge_lengths"
        if self.get_weight('isometry') == 0:
            return None
        if initialized:
            X = self._X0
        else:
            X = self.X
        vi, vj = self.mesh.vertex_ring_vertices_iterators(order=True) # later should define it as global
        Vi = X[columnnew(vi,0,self.mesh.V)].reshape(-1,3,order='F')
        Vj = X[columnnew(vj,0,self.mesh.V)].reshape(-1,3,order='F')
        el = np.linalg.norm(Vi-Vj,axis=1)
        return el    
    
    def get_agweb_initial(self,diagnet=False,another_poly_direction=False,
                          AAG=False,GGA=False,AAGG=False):
        "initilization of AG-net project"
        V = self.mesh.vertices
        v,v1,v2,v3,v4 = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4].T # regular
        v,va,vb,vc,vd = self.mesh.rr_star_corner# in diagonal direction
        V0,V1,V2,V3,V4,Va,Vb,Vc,Vd = V[v],V[v1],V[v2],V[v3],V[v4],V[va],V[vb],V[vc],V[vd]
        vnn = self.mesh.vertex_normals()[v]
        
        if diagnet:
            "GGAA / GAA"
            Vg1,Vg2,Vg3,Vg4 = V1,V2,V3,V4
        else:
            "AAGG / AAG"
            Vg1,Vg2,Vg3,Vg4 = Va,Vb,Vc,Vd
            
        "X +=[ln, vN] + [oNi]; oNi not need to be unit; all geodesics matter"
        if AAGG:
            "oN1,oN2 from Gnet-osculating_normals,s.t. anetN*oN1(oN2)=0"
            oN1,oN2 = np.cross(Vg3-V0,Vg1-V0),np.cross(Vg4-V0,Vg2-V0)
            X = np.r_[oN1.flatten('F'),oN2.flatten('F')] 
        elif AAG:
            "oN from geodesic-osculating-normal (not unit)"
            if another_poly_direction:
                Vl,Vr = Vg2, Vg4
            else:
                Vl,Vr = Vg1, Vg3
            oN = np.cross(Vr-V0,Vl-V0)
            X = np.r_[oN.flatten('F')] 
        elif GGA:
            "X +=[vN, oN1, oN2]; oN1,oN2 from Gnet-osculating_normals"
            if diagnet:
                "AGG"
                Vg1,Vg2,Vg3,Vg4 = Va,Vb,Vc,Vd # different from above
            else:
                "GGA"
                Vg1,Vg2,Vg3,Vg4 = V1,V2,V3,V4 # different from above
            oN1,oN2 = np.cross(Vg3-V0,Vg1-V0),np.cross(Vg4-V0,Vg2-V0)
            vn = np.cross(oN1,oN2)
            vN = vn / np.linalg.norm(vn,axis=1)[:,None]
            ind = np.where(np.einsum('ij,ij->i',vnn,vN)<0)[0]
            vN[ind]=-vN[ind]
            X = np.r_[vN.flatten('F'),oN1.flatten('F'),oN2.flatten('F')]  
        return X
    
    def get_agweb_an_n_on(self,is_n=False,is_on=False,is_all_n=False):
        V = self.mesh.vertices
        v = self.mesh.rr_star[:,0]#self.mesh.rr_star_corner[0]
        an = V[v]
        n = self.mesh.vertex_normals()[v]
        on1=on2=n
        num = len(self.mesh.ind_rr_star_v4f4)
        if self.is_initial:
            if self.get_weight('AAGnet') or self.get_weight('GAAnet'):
                "vertex normal from A-net"
                X = self.get_agweb_initial(AAG=True)
                #on = X[:3*num].reshape(-1,3,order='F')
            elif self.get_weight('GGAnet') or self.get_weight('AGGnet'):
                "X=+[N,oN1,oN2]"
                X = self.get_agweb_initial(GGA=True)
                n = X[:3*num].reshape(-1,3,order='F')
                on1 = X[3*num:6*num].reshape(-1,3,order='F')
                on2 = X[6*num:9*num].reshape(-1,3,order='F')
            elif self.get_weight('AAGGnet') or self.get_weight('GGAAnet'):
                "vertex-normal from Anet, X+=[on1,on2]"
                X = self.get_agweb_initial(AAGG=True)
                on1 = X[:3*num].reshape(-1,3,order='F')
                on2 = X[3*num:6*num].reshape(-1,3,order='F')
            elif self.get_weight('Anet'):
                pass
                # v = v[self.mesh.ind_rr_star_v4f4]
                # n = n[v]
            elif self.get_weight('AGnet'):
                if False:
                    _,_,lt1,lt2 = self.mesh.get_net_osculating_tangents()
                    n = np.cross(lt1[1],lt2[1])
                    n = n / np.linalg.norm(n,axis=1)[:,None]
                else:
                    _,_,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents(False,True)
                    n = np.cross(ut1,ut2)
                    n = n / np.linalg.norm(n,axis=1)[:,None]
        else:
            X = self.X
            if self.get_weight('AAGnet') or self.get_weight('GAAnet'):
                "X=+[oNg]"
                ##print(v,self.mesh.ind_rr_star_v4f4,len(v),len(self.mesh.ind_rr_star_v4f4))
                v = v[self.mesh.ind_rr_star_v4f4]
                n = X[self._Nanet-3*num:self._Nanet].reshape(-1,3,order='F') 
                d = self._Ndgeo-3*num
                #on = X[d:d+3*num].reshape(-1,3,order='F') 
            elif self.get_weight('GGAnet') or self.get_weight('AGGnet'):
                d = self._Ndgeo-9*num
                n = X[d:d+3*num].reshape(-1,3,order='F')
                on1 = X[d+3*num:d+6*num].reshape(-1,3,order='F')
                on2 = X[d+6*num:d+9*num].reshape(-1,3,order='F')
            elif self.get_weight('AAGGnet') or self.get_weight('GGAAnet'):
                v = v[self.mesh.ind_rr_star_v4f4]
                n = X[self._Nanet-3*num:self._Nanet].reshape(-1,3,order='F') 
                d = self._Ndgeo-6*num
                on1 = X[d:d+3*num].reshape(-1,3,order='F')
                on2 = X[d+3*num:d+6*num].reshape(-1,3,order='F')
            elif self.get_weight('Anet'):
                v = v[self.mesh.ind_rr_star_v4f4]
                n = X[self._Nanet-3*num:self._Nanet].reshape(-1,3,order='F')
            elif self.get_weight('AGnet'):
                if False:
                    Nag = self._Nag
                    arr3 = np.arange(3*num)
                    if self.opt_AG_const_rii or self.opt_AG_const_r0:
                        if self.opt_AG_const_rii:
                            #k = len(igeo)
                            #c_ri = Nag-k+np.arange(k)
                            pass
                            #c_srfN = Nag-6*num+arr3-k
                            #c_ogN = Nag-4*num+arr3-k
                        elif self.opt_AG_const_r0:
                            #c_r = Nag-1
                            c_srfN = Nag-6*num+arr3-1
                            #c_ogN = Nag-4*num+arr3-1
                    else:
                        c_srfN = Nag-6*num+arr3
                        #c_ogN = Nag-3*num+arr3
                    n = X[c_srfN].reshape(-1,3,order='F')
                    #on = X[c_ogN].reshape(-1,3,order='F')
                elif False:
                    ie1 = self._N5-12*num+np.arange(3*num)
                    ue1 = X[ie1].reshape(-1,3,order='F')
                    ue2 = X[ie1+3*num].reshape(-1,3,order='F')
                    ue3 = X[ie1+6*num].reshape(-1,3,order='F')
                    ue4 = X[ie1+9*num].reshape(-1,3,order='F')
                    #try:
                    if self.is_AG_or_GA:
                        n = ue2+ue4
                    else:
                        n = ue1+ue3
                    n = n / np.linalg.norm(n,axis=1)[:,None]
                    # except:
                    #     t1,t2 = ue1-ue3,ue2-ue4
                    #     n = np.cross(t1,t2)
                    #     n = n / np.linalg.norm(n,axis=1)[:,None]
                    v = v[self.mesh.ind_rr_star_v4f4]
                else:
                    c_srfN = self._Nag-3*num+np.arange(3*num)
                    n = X[c_srfN].reshape(-1,3,order='F')
        if is_n:
            n = n / np.linalg.norm(n,axis=1)[:,None]
            alln = self.mesh.vertex_normals()
            n0 = alln[v]
            j = np.where(np.einsum('ij,ij->i',n0,n)<0)[0]
            n[j] = -n[j]
            return V[v],n
        elif is_on:
            on1 = on1 / np.linalg.norm(on1,axis=1)[:,None]
            on2 = on2 / np.linalg.norm(on2,axis=1)[:,None]
            return an,on1,on2
        elif is_all_n:
            alln = self.mesh.vertex_normals()
            n0 = alln[v]
            j = np.where(np.einsum('ij,ij->i',n0,n)<0)[0]
            n[j] = -n[j]
            alln[v] = n
            return alln

    def get_agnet_normal(self,is_biN=False):
        V = self.mesh.vertices
        v,v1,v2,v3,v4 = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4].T
        an = V[v]
        if is_biN:
            "AGnet: Asy(v1-v-v3), Geo(v2-v-v4), binormal of geodesic crv"
            if self.is_AG_or_GA:
                eb = (V[v2]-V[v])#/np.linalg.norm(V[v2]-V[v],axis=1)[:,None]
                ed = (V[v4]-V[v])#/np.linalg.norm(V[v4]-V[v],axis=1)[:,None]
            else:
                eb = (V[v1]-V[v])#/np.linalg.norm(V[v1]-V[v],axis=1)[:,None]
                ed = (V[v3]-V[v])#/np.linalg.norm(V[v3]-V[v],axis=1)[:,None]
            n = np.cross(eb,ed)
            i = np.where(np.linalg.norm(n,axis=1)==0)[0]
            if len(i)!=0:
                n[i]=np.zeros(3)
            else:
                n = n / np.linalg.norm(n,axis=1)[:,None]
            return an, n
        
        if False:
            _,_,lt1,lt2 = self.mesh.get_net_osculating_tangents()
            n = np.cross(lt1[1],lt2[1])
            n = n / np.linalg.norm(n,axis=1)[:,None]
        else:
            _,_,ut1,ut2,_,_ = self.mesh.get_v4_unit_tangents(False,True)
            n = np.cross(ut1,ut2)
            n = n / np.linalg.norm(n,axis=1)[:,None]
        return an, n

    def index_of_strip_along_polyline(self):
        "ver_poly_strip1: 2-dim list with different length, at least 2"
        w3 = self.get_weight('AAGnet')
        w4 = self.get_weight('AAGGnet')
        diag = True if w3 or w4 else False
        d = self.set_another_polyline
        if diag:
            iall,iind,_,_ = self.mesh.get_diagonal_vertex_list(5,d) # interval is random
        else:
            iall,iind,_,_ = self.mesh.get_isoline_vertex_list(5,d) # updated, need to check
        self._ver_poly_strip1 = [iall,iind]   
        
    def index_of_mesh_polylines(self):
        "index_of_strip_along_polyline without two bdry vts, this include full"
        if self.is_singular:
            self._ver_poly_strip1,_,_ = quadmesh_with_1singularity(self.mesh)
        else:
            "ver_poly_strip1,ver_poly_strip2"
            iall = self.mesh.get_both_isopolyline(diagpoly=self.switch_diagmeth,
                                                  is_one_or_another=self.set_another_polyline)
            self._ver_poly_strip1 = iall   
            iall = self.mesh.get_both_isopolyline(diagpoly=self.switch_diagmeth,
                                                  is_one_or_another=not self.set_another_polyline)
            self._ver_poly_strip2 = iall        
    
    def get_initial_singular_diagply_normal(self,is_init=False,AGnet=False,CCnet=False):
        V = self.mesh.vertices
        vl,vc,vr = self.singular_polylist
        Vl,Vc,Vr = V[vl], V[vc], V[vr]
        if is_init:
            on = np.cross(Vl-Vc, Vr-Vc)
            return on / np.linalg.norm(on,axis=1)[:,None]
        else:
            if self.is_initial:
                v = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4][:,0]
                vN = self.mesh.vertex_normals()[v] ##approximate.
            else:
                if AGnet:
                    #num = self.mesh.num_regular
                    num = len(self.mesh.ind_rr_star_v4f4)
                    arr = self._Nanet-3*num+np.arange(3*num)
                    vN = self.X[arr].reshape(-1,3,order='F')
                elif CCnet:
                    num1 = len(self.mesh.ind_rr_star_v4f4)
                    num2 = len(self.ind_rr_vertex)
                    arr = self._Ncd-3*num1-8*num2+np.arange(3*num1)
                    vN = self.X[arr].reshape(-1,3,order='F')
            Nc = vN[self.ind_rr_vertex]
            return Nc,Vl,Vc,Vr
        
    def get_poly_strip_normal(self,pl1=False,pl2=False):
        "for planar strip: each strip 1 normal as variable, get mean n here"
        V = self.mesh.vertices
        
        if pl1:
            iall = self.ver_poly_strip1
        elif pl2:
            iall = self.ver_poly_strip2    
        else:
            iall = self.ver_poly_strip1[0]    
        n = np.array([0,0,0])
        for iv in iall:
            if len(iv)==2:
                ni = np.array([(V[iv[1]]-V[iv[0]])[1],-(V[iv[1]]-V[iv[0]])[0],0]) # random orthogonal normal
            elif len(iv)==3:
                vl,v0,vr = iv[0],iv[1],iv[2]
                ni = np.cross(V[vl]-V[v0],V[vr]-V[v0])
            else:
                vl,v0,vr = iv[:-2],iv[1:-1],iv[2:]
                ni = np.cross(V[vl]-V[v0],V[vr]-V[v0])
                ni = ni / np.linalg.norm(ni,axis=1)[:,None]
                ni = np.mean(ni,axis=0)
            ni = ni / np.linalg.norm(ni)
            n = np.vstack((n,ni))
        return n[1:,:]

    def get_poly_strip_ruling_tangent(self):
        "ruling"
        V = self.mesh.vertices
        iall = self.mesh.get_both_isopolyline(diagpoly=self.switch_diagmeth,is_one_or_another=True)
        t = np.array([0,0,0])
        for iv in iall:
            ti = V[iv[1:]]-V[iv[:-1]]
            ti = np.mean(ti,axis=0)
            t = np.vstack((t,ti))
        return t[1:,:]
    
    def get_mesh_planar_normal_or_plane(self,pl1=False,pl2=False,pln=False,scale=None):
        V = self.mesh.vertices
        if pl1:
            iall = self.ver_poly_strip1
        elif pl2:
            iall = self.ver_poly_strip2    
        else:
            iall = self.ver_poly_strip1[0]    
        num = len(iall)
        
        if not pln:
            an=vn = np.array([0,0,0])
            i= 0
            for iv in iall:
                vl,v0,vr = iv[:-2],iv[1:-1],iv[2:]
                an = np.vstack((an,V[iv]))
                if self.get_weight('planar_geodesic'):
                    nx = self.X[self._Ndgeopc-3*num+i]
                    ny = self.X[self._Ndgeopc-2*num+i]
                    nz = self.X[self._Ndgeopc-1*num+i]
                    ni = np.tile(np.array([nx,ny,nz]),len(iv)).reshape(-1,3) 
                elif self.get_weight('planar_plys'):
                    nx = self.X[self._Npp-3*num+i]
                    ny = self.X[self._Npp-2*num+i]
                    nz = self.X[self._Npp-1*num+i]
                    ni = np.tile(np.array([nx,ny,nz]),len(iv)).reshape(-1,3) 
                else:
                    "len(an)=len(ni)=len(iv)-2"
                    an = np.vstack((an,V[v0]))
                    ni = np.cross(V[vl]-V[v0],V[vr]-V[v0])
                    ni = ni / np.linalg.norm(ni,axis=1)[:,None]
                vn = np.vstack((vn,ni))
                i+= 1
            return an[1:,:],vn[1:,:]
        else:
            "planar strip passing through ply-vertices with above uninormal"
            P1=P2=P3=P4 = np.array([0,0,0])
            i= 0
            for iv in iall:
                vl,vr = iv[:-1],iv[1:]
                vec = V[vr]-V[vl]
                vec = np.vstack((vec,vec[-1])) #len=len(iv)
                if scale is None:
                    scale = np.mean(np.linalg.norm(vec,axis=1)) * 0.1
                if self.get_weight('planar_geodesic'):
                    nx = self.X[self._Ndgeopc-3*num+i]
                    ny = self.X[self._Ndgeopc-2*num+i]
                    nz = self.X[self._Ndgeopc-1*num+i]
                    oni = np.array([nx,ny,nz])
                    Ni = np.cross(vec,oni)
                elif self.get_weight('planar_plys'):
                    nx = self.X[self._Npp-3*num+i]
                    ny = self.X[self._Npp-2*num+i]
                    nz = self.X[self._Npp-1*num+i]
                    oni = np.array([nx,ny,nz])
                    Ni = np.cross(vec,oni)
                else:
                    il,i0,ir = iv[:-2],iv[1:-1],iv[2:]
                    oni = np.cross(V[il]-V[i0],V[ir]-V[i0])
                    oni = np.vstack((oni[0],oni,oni[-1])) #len=len(iv)
                    oni = oni / np.linalg.norm(oni,axis=1)[:,None]
                    Ni = np.cross(vec,oni)
                uNi = Ni / np.linalg.norm(Ni,axis=1)[:,None] * scale  
                i+= 1    
                P1,P2 = np.vstack((P1,V[vl]-uNi[:-1])),np.vstack((P2,V[vr]-uNi[1:]))
                P4,P3 = np.vstack((P4,V[vl]+uNi[:-1])),np.vstack((P3,V[vr]+uNi[1:]))
            pm = self.mesh.make_quad_mesh_pieces(P1[1:],P2[1:],P3[1:],P4[1:])   
            return pm

    def get_singularmesh_diagpoly(self,is_poly=False,is_rr_vertex=True):
        plylist,vlr,vlcr = quadmesh_with_1singularity(self.mesh)
        self._singular_polylist = vlcr ##==[vl,vc,vr]
        
        ##### AAG-SINGULAR / CG / CA project:
        if is_rr_vertex:
            rrv = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4][:,0]
        else: ##no use now
            rrv = self.mesh.ver_regular
        ind = []
        for i in range(len(vlcr[1])):
            ck = np.where(rrv==vlcr[1][i])[0]
            ind.append(ck[0])
        self._ind_rr_vertex = np.array(ind,dtype=int)
        
        if is_poly:
            Vl,Vr = self.mesh.vertices[vlr[0]], self.mesh.vertices[vlr[1]]
            return self.mesh.make_polyline_from_endpoints(Vl,Vr)

    def get_nonsymmetric_edge_ratio(self,diagnet=False):
        """each quadface, oriented edge1,edge2 
            l1 > l2 or l1<l2<==> (l1-l2)^2 = s^2 + eps"""
        if diagnet:
            pass
        else:
            "suppose edge1: v1v2; edge2: v1v4"
            V = self.mesh.vertices
            
            if self.is_singular:
                v1,v2,_,v4 = quadmesh_with_1singularity(self.mesh,False,True)
            else:
                v1,v2,_,v4 = self.mesh.rr_quadface.T # in odrder
                
            mean1 = np.mean(np.linalg.norm(V[v4]-V[v1],axis=1))
            mean2 = np.mean(np.linalg.norm(V[v2]-V[v1],axis=1))
            print(mean1,mean2)
            print('%.2g'%(mean1-mean2)**2)
            
            
            self.ind_nonsym_v124 = [v1,v4,v2]    
            il1 = self.mesh.edge_from_connected_vertices(v1,v4)
            il2 = self.mesh.edge_from_connected_vertices(v1,v2)
            self.ind_nonsym_l12 = [il1,il2]
            
            allv1, allv2 = self.mesh.edge_vertices()
            eL = np.linalg.norm(V[allv2]-V[allv1],axis=1)
            
            "(l1-l2)^2 = s^2 + eps"
            s = np.zeros(len(il1)) ## len(il1)=len(il2)=len(s)
            ind = np.where((eL[il1]-eL[il2])**2-self.nonsym_eps>0)[0]
            s[ind] = np.sqrt((eL[il1][ind]-eL[il2][ind])**2-self.nonsym_eps)
            print('%.2g'% np.mean((eL[il1]-eL[il2])**2))
            return eL,s    

    def get_conjugate_diagonal_net(self,normal=True):
        "CCD-net vertex normal"
        V = self.mesh.vertices
        v,v1,v2,v3,v4 = self.mesh.rr_star[self.mesh.ind_rr_star_v4f4].T
        num = len(v)
        if self.get_weight('diag_1_asymptotic'):
            arr = self._Ncd-3*num+np.arange(3*num)
            n = self.X[arr].reshape(-1,3,order='F')
        elif self.get_weight('diag_1_geodesic'):
            if self.is_singular:
                arr = self._Ncd-3*num-8*len(self.ind_rr_vertex)+np.arange(3*num)
            else:
                arr = self._Ncd-11*num+np.arange(3*num)
            n = self.X[arr].reshape(-1,3,order='F')
        else:
            n = np.cross(V[v3]-V[v1], V[v4]-V[v2])
        if normal:
            return V[v], n
              
        
    # -------------------------------------------------------------------------
    #                                 Build
    # -------------------------------------------------------------------------

    def build_iterative_constraints(self):
        self.build_added_weight() # Hui change
        
        H, r = self.mesh.iterative_constraints(**self.weights) ##NOTE: in gridshell.py
        self.add_iterative_constraint(H, r, 'mesh_iterative')
        
        if self.get_weight('planarity'):
            H,r = con_planarity_constraints(**self.weights)  
            self.add_iterative_constraint(H, r, 'planarity')
            
        if self.get_weight('AAGnet') or self.get_weight('GAAnet') \
            or self.get_weight('GGAnet')or self.get_weight('AGGnet') \
            or self.get_weight('AAGGnet')or self.get_weight('GGAAnet') :
           "agnet_liouville, planar_geodesic"
           if self.get_weight('planar_geodesic'):
               strip = self.ver_poly_strip1
           else:
               strip = None
           if self.weight_checker<1:
               idck = self.mesh.ind_ck_tian_rr_vertex,#self.mesh.ind_ck_rr_vertex,
           else:
               idck = None
           H,r = con_anet_geodesic(strip,self.set_another_polyline,
                                   checker_weight=self.weight_checker,
                                   id_checker=idck, #self.mesh.ind_ck_tian_rr_vertex,#self.mesh.ind_ck_rr_vertex,
                                   **self.weights)  
           self.add_iterative_constraint(H, r, 'AG-web')
    
        if self.get_weight('ruling'):
            H,r = con_polyline_ruling(switch_diagmeth=False,
                                      **self.weights)
            self.add_iterative_constraint(H, r, 'ruling')
            
        if self.get_weight('oscu_tangent'):
            if self.get_weight('GAAnet') or self.get_weight('AGGnet') or self.get_weight('GGAAnet'):
                diag=True
                is_orthonet = True
            else:
                diag=False
                is_orthonet = True
            H,r = con_osculating_tangents(diag,is_ortho=is_orthonet,**self.weights)
            self.add_iterative_constraint(H, r, 'oscu_tangent')
        
        if self.get_weight('GOnet'):
            d = True if self.is_AG_or_GA else False
            H,r = con_gonet(rregular=True,is_direction24=d,**self.weights)
            self.add_iterative_constraint(H, r, 'GOnet')
            
        if self.get_weight('AGnet'):
            H,r = con_AGnet(self.is_AG_or_GA,self.opt_AG_ortho,
                            self.opt_AG_const_rii,self.opt_AG_const_r0,
                            **self.weights)  
            self.add_iterative_constraint(H, r, 'AGnet')    
            
        ###-------partially shared-used codes:---------------------------------
        if self.get_weight('unit_edge'): 
            H,r = con_unit_edge(rregular=True,**self.weights)
            self.add_iterative_constraint(H, r, 'unit_edge')
        elif self.get_weight('unit_diag_edge'): 
            H,r = con_unit_edge(rregular=True,**self.weights)
            self.add_iterative_constraint(H, r, 'unit_diag_edge')
            
        if self.get_weight('fix_point'):
            ind,Vf = self.ind_fixed_point, self.fixed_value
            H,r = con_fix_vertices(ind, Vf,**self.weights)
            self.add_iterative_constraint(H,r, 'fix_point')
            
            
        if self.get_weight('boundary_glide'):
            "the whole boundary"
            refPoly = self.glide_reference_polyline
            glideInd = self.mesh.boundary_curves(corner_split=False)[0] 
            w = self.get_weight('boundary_glide')
            H,r = con_alignment(w, refPoly, glideInd,**self.weights)
            self.add_iterative_constraint(H, r, 'boundary_glide')
        elif self.get_weight('i_boundary_glide'):
            "the i-th boundary"
            refPoly = self.i_glide_bdry_crv
            glideInd = self.i_glide_bdry_ver
            if len(glideInd)!=0:
                w = self.get_weight('i_boundary_glide')
                H,r = con_alignments(w, refPoly, glideInd,**self.weights)
                self.add_iterative_constraint(H, r, 'iboundary_glide')
                
        if self.get_weight('orthogonal'):
            #H,r = con_orthogonal(**self.weights)
            H,r = con_orthogonal_midline(**self.weights)
            self.add_iterative_constraint(H, r, 'orthogonal')
            
        if self.get_weight('isogonal'): # todo: mayhas problem of unit-edge(rregular)
            H,r = con_isogonal(np.cos(self.angle/180.0*np.pi),
                               assign=self.if_angle,**self.weights)
            self.add_iterative_constraint(H, r, 'isogonal')
            
        elif self.get_weight('isogonal_diagnet'): 
            H,r = con_isogonal_diagnet(np.cos(self.angle/180.0*np.pi),
                                       assign=self.if_angle,**self.weights) 
            self.add_iterative_constraint(H, r, 'isogonal_diagnet')

        if self.get_weight('isometry'):
            l0 = self.edge_lengths_isometry(initialized=True)
            H,r = con_isometry(l0,**self.weights)
            self.add_iterative_constraint(H, r, 'isometry')

        if self.get_weight('z0') !=0:
            from constraints_glide import con_glide_in_plane
            H,r = con_glide_in_plane(2,**self.weights)
            self.add_iterative_constraint(H,r, 'z0')        
            
        if self.get_weight('Anet'):
            if self.weight_checker<1:
                idck = self.mesh.ind_ck_tian_rr_vertex,#self.mesh.ind_ck_rr_vertex,
            else:
                idck = None
            H,r = con_anet(rregular=True,checker_weight=self.weight_checker,
                           id_checker=idck,
                           **self.weights)
            self.add_iterative_constraint(H, r, 'Anet')
            
        elif self.get_weight('Anet_diagnet'):
            if self.weight_checker<1:
                idck = self.mesh.ind_ck_tian_rr_vertex,#self.mesh.ind_ck_rr_vertex,
            else:
                idck = None
            H,r = con_anet_diagnet(checker_weight=self.weight_checker,
                                   id_checker=idck,
                                     **self.weights)
            self.add_iterative_constraint(H, r, 'Anet_diagnet')

        if self.get_weight('Gnet'):
            if self.weight_checker<1:
                idck = self.mesh.ind_ck_tian_rr_vertex,#self.mesh.ind_ck_rr_vertex,
            else:
                idck = None
            H,r = con_gnet(rregular=True,checker_weight=self.weight_checker,
                           id_checker=idck,
                           **self.weights)
            self.add_iterative_constraint(H, r, 'Gnet')
            
        elif self.get_weight('Gnet_diagnet'):
            if self.weight_checker<1:
                idck = self.mesh.ind_ck_tian_rr_vertex,#self.mesh.ind_ck_rr_vertex,
            else:
                idck = None
            H,r = con_gnet_diagnet(checker_weight=self.weight_checker,
                                   id_checker=idck,**self.weights)
            self.add_iterative_constraint(H, r, 'Gnet_diagnet')
            
        if self.get_weight('nonsymmetric'):
            v012,eps = self.ind_nonsym_v124,self.nonsym_eps
            il12 = self.ind_nonsym_l12
            H,r = con_nonsquare_quadface(v012,il12,eps,**self.weights)
            self.add_iterative_constraint(H, r, 'nonsymmetric')
        
        if self.get_weight('AAG_singular'):
            H,r = con_singular_Anet_diag_geodesic(self.singular_polylist,
                                                   self.ind_rr_vertex,**self.weights)   
            self.add_iterative_constraint(H, r, 'singular')

        if self.get_weight('planar_ply1'):
            ## only for \obj_cheng\every_5_PPQ.obj'
            matrix = self.ver_poly_strip1
            #matrix = self.mesh.rot_patch_matrix[:,::5].T
            H,r = con_planar_1familyof_polylines(self._Nppq,matrix,
                                                 is_parallxy_n=False, #TO SET
                                                 **self.weights)              
            self.add_iterative_constraint(H, r, 'planar_ply1')
        if self.get_weight('planar_ply2'):
            H,r = con_planar_1familyof_polylines(self._Nppo,self.ver_poly_strip2,
                                                 is_parallxy_n=False, #TO SET
                                                 **self.weights)              
            self.add_iterative_constraint(H, r, 'planar_ply2')   
           
        
        if self.get_weight('diag_1_asymptotic') or self.get_weight('diag_1_geodesic'):
            if self.get_weight('diag_1_asymptotic'):
                name = 'diag_1_asymptotic'
                is_switch = True
            elif self.get_weight('diag_1_geodesic'):
                name = 'diag_1_geodesic'
                is_switch = False
            H,r = con_diag_1_asymptotic_or_geodesic(
                                   singular_polylist=self._singular_polylist,
                                   ind_rrv=self._ind_rr_vertex,
                                   another_poly_direction=self.set_another_polyline,
                                   is_asym_or_geod = is_switch,
                                   **self.weights)
            self.add_iterative_constraint(H, r, name)

        if self.get_weight('ctrlnet_symmetric_1diagpoly'):
            H,r = con_ctrlnet_symmetric_1_diagpoly(self.set_another_polyline,
                                                   **self.weights)
            self.add_iterative_constraint(H, r, 'ctrlnet_symmetric_1diagpoly')   
            
        ###--------------------------------------------------------------------                
   
        self.is_initial = False   
            
        #print('-'*10)
        print(' Err_total: = ','%.3e' % np.sum(np.square(self._H*self.X-self._r)))
        #print('-'*10)
        
    def build_added_weight(self): # Hui add
        self.add_weight('mesh', self.mesh)
        self.add_weight('N', self.N)
        self.add_weight('X', self.X)
        self.add_weight('N1', self._N1)
        self.add_weight('N5', self._N5)
        self.add_weight('N6', self._N6)
        self.add_weight('Nanet', self._Nanet)
        self.add_weight('Ndgeo', self._Ndgeo)
        self.add_weight('Ndgeoliou', self._Ndgeoliou)
        self.add_weight('Ndgeopc', self._Ndgeopc)
        self.add_weight('Nruling', self._Nruling)
        self.add_weight('Noscut', self._Noscut)
        self.add_weight('Nnonsym', self._Nnonsym)
        
        self.add_weight('Nag', self._Nag)
        
        self.add_weight('Npp', self._Npp)
        self.add_weight('Ncd', self._Ncd)
        self.add_weight('Ncds', self._Ncds)

    #--------------------------------------------------------------------------
    #                                  Results
    #--------------------------------------------------------------------------
    def post_iteration_update(self):
        V = self.mesh.V
        self.mesh.vertices[:,0] = self.X[0:V]
        self.mesh.vertices[:,1] = self.X[V:2*V]
        self.mesh.vertices[:,2] = self.X[2*V:3*V]

    def vertices(self):
        V = self.mesh.V
        vertices = self.X[0:3*V]
        vertices = np.reshape(vertices, (V,3), order='F')
        return vertices        
        