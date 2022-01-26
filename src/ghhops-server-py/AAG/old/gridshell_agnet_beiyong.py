# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:44:44 2022

@author: WANGH0M
"""



    # @on_trait_change('show_rectify_strip_ctrlnet,show_rectify_strip_diagnet')
    # def plot_polyline_E3_developable_strip_net(self):
    #     from huilab.huimesh.developablestrip import StraightStrip
    #     dist = self.mean_edge_length() * self.scale_dist_offset
    #     num_seg = self.num_bezier_divide
        
    #     if self.show_rectify_strip_ctrlnet or self.show_rectify_strip_diagnet:
    #         asym_or_geo,diagpoly = self._set_asy_or_geo_net(self.show_rectify_strip_ctrlnet)
    #         name='RNet12' if self.show_rectify_strip_ctrlnet else 'RNet34'
            
    #         if self.switch_singular_mesh:
    #             from huilab.huimesh.singularMesh import get_singular_quintic_Bezier_spline_crvs_checker
                
    #         def _get_sm(is_1st_or_2nd):
    #             if self.switch_interpolate_checker:
    #                 "this is checker case"
    #                 if self.switch_singular_mesh:
    #                     _,_,_,_,frame,annr,crvature = get_singular_quintic_Bezier_spline_crvs_checker(
    #                         self,
    #                         normal=self.VN,
    #                         efair=self.set_bezier_ctrlp_fairness,
    #                         is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                         is_one_or_another=is_1st_or_2nd,
    #                         is_checker=3, ##3 is the num of subdivision quadfaces
    #                         is_dense=True,num_divide=self.num_bezier_divide,
    #                         is_smooth=self.set_smooth_vertices_fairness)
    #                 else:
    #                     _,_,_,_,frame,annr,crvature = self.get_poly_quintic_Bezier_spline_crvs_checker(
    #                         self,
    #                         normal=self.VN,
    #                         efair=self.set_bezier_ctrlp_fairness,
    #                         is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                         is_one_or_another=is_1st_or_2nd,
    #                         is_checker=4, ##4 is the num of subdivision quadfaces
    #                         is_onlyinner=self.switch_if_ruling_denseinner,
    #                         num_divide=self.num_bezier_divide,
    #                         is_modify=self.switch_if_ruling_rectify,
    #                         is_smooth=self.set_smooth_vertices_fairness)
    #             else:
    #                 "this is not checker case"
    #                 _,_,_,_,frame,annr,crvature = self.get_poly_quintic_Bezier_spline_crvs(
    #                     self,
    #                     normal=self.VN,
    #                     efair=self.set_bezier_ctrlp_fairness,
    #                     is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                     is_one_or_another=is_1st_or_2nd,
    #                     is_interpolate = self.switch_interpolate_checker,
    #                     is_onlyinner=self.switch_if_ruling_denseinner,
    #                     is_dense=True,num_divide=self.num_bezier_divide,
    #                     is_modify=self.switch_if_ruling_rectify,
    #                     is_smooth=self.set_smooth_vertices_fairness)
                    
    #             _,an,r,_,nmlist = annr
    #             _,E1,_,E3 = frm
    #             unitN = E3
    #             if True:
    #                 "envelope directly by normals"
    #                 is_smooth = self.set_smooth_vertices_fairness
    #                 sm = self.get_strip_from_rulings(an,unitN*dist,nmlist,is_smooth)
    #             else:
    #                 centerline_symmetric = False if asym_or_geo else True
    #                 strip = StraightStrip(an,unitN,E1,r,num_seg,dist,nmlist,
    #                                      efair=self.set_ply_rectify_strip_fairness,
    #                                      itera=50,ee=0.001)
    #                 sm = strip.get_strip(centerline_symmetric)
    #             return sm
                
    #         sm1 = _get_sm(True)
    #         sm2 = _get_sm(False) ##geo:red:(240,114,114);asy:blue:(98,113,180)
    #         showf1 = Faces(sm1,color =(98,113,180),name=name+'f1') #'golden_yellow'
    #         showe1 = Edges(sm1,color =(98,113,180),name=name+'e1')#'gold'
    #         showf2 = Faces(sm2,color =(98,113,180),name=name+'f2')
    #         showe2 = Edges(sm2,color =(98,113,180),name=name+'e2')
    #         poly1 = self.get_boundary_polyline(sm1) 
    #         selfmanager.plot_polyline(polyline=poly1,
    #                                        tube_radius=0.5*selfmanager.r ,
    #                                        glossy=1,
    #                                        color='black',
    #                                        sampling=100,
    #                                        name=name+'pl1') 
    #         poly2 = self.get_boundary_polyline(sm2) 
    #         selfmanager.plot_polyline(polyline=poly2,
    #                                        tube_radius=0.5*selfmanager.r ,
    #                                        glossy=1,
    #                                        color='black',
    #                                        sampling=100,
    #                                        name=name+'pl2') 
    #         selfmanager.add([showf1,showe1,showf2,showe2])
    #     else:
    #         selfmanager.remove(['RNet12e1','RNet12f1','RNet12e2','RNet12f2'])
    #         selfmanager.remove(['RNet34e1','RNet34f1','RNet34e2','RNet34f2'])
    #         selfmanager.remove(['RNet12pl1','RNet12pl2','RNet34pl1','RNet34pl2']) 

    # ##@on_trait_change('show_bezier_ctrlnet,show_bezier_diagnet')
    # def plot_Bezier_curvenet(self):
    #     asym_or_geo,diagpoly = self._set_asy_or_geo_net(self.show_bezier_ctrlnet)
        
    #     if not self.switch_singular_mesh:
    #         ick = 4 if self.switch_interpolate_checker else 1
    #         _,_,crv1,_,_,_,_ = self.get_poly_quintic_Bezier_spline_crvs_checker(
    #             self,
    #             normal=self.VN,
    #             is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #             is_one_or_another=True, ##NOTE: different
    #             is_checker=ick, ##4 is the num of subdivision quadfaces
    #             efair=self.set_bezier_ctrlp_fairness)
    #         _,_,crv2,_,_,_,_ = self.get_poly_quintic_Bezier_spline_crvs_checker(
    #             self,
    #             normal=self.VN,
    #             is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #             is_one_or_another=False, ##NOTE: different
    #             is_checker=ick, ##4 is the num of subdivision quadfaces
    #             efair=self.set_bezier_ctrlp_fairness)
    #     else:
    #         from huilab.huimesh.singularMesh import get_singular_quintic_Bezier_spline_crvs_checker
    #         ick = 3 if self.switch_interpolate_checker else 1
    #         _,_,crv1,_,_,_,_ = get_singular_quintic_Bezier_spline_crvs_checker(
    #             self,
    #             normal=self.VN,
    #             is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #             is_one_or_another=True,
    #             is_checker=ick, ##4 is the num of subdivision quadfaces
    #             efair=self.set_bezier_ctrlp_fairness)
    #         _,_,crv2,_,_,_,_ = get_singular_quintic_Bezier_spline_crvs_checker(
    #             self,
    #             normal=self.VN,
    #             is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #             is_one_or_another=False,
    #             is_checker=ick, ##4 is the num of subdivision quadfaces
    #             efair=self.set_bezier_ctrlp_fairness)


        
    # @on_trait_change('show_bezier_ctrlnet_ss,show_bezier_diagnet_ss')
    # def plot_Bezier_curvenet_strip(self):
    #     if self.show_bezier_ctrlnet_ss or self.show_bezier_diagnet_ss:
    #         asym_or_geo,diagpoly = self._set_asy_or_geo_net(self.show_bezier_ctrlnet_ss)
    #         name='NetSS12' if self.show_bezier_ctrlnet_ss else 'NetSS34'
    #         dist = self.mean_edge_length() * self.scale_dist_offset
    #         num_seg = self.num_bezier_divide
            
    #         if False:
    #             #-------------------------------------------------------------
    #             "Optimize developable strip by normal vectors"
    #             from huilab.huimesh.developablestrip import StraightStrip
    #             def _get_sm(is_1st_or_2nd):
    #                 if self.switch_interpolate_checker:
    #                     _,_,_,_,frame,annr,crvature = self.get_poly_quintic_Bezier_spline_crvs_checker(
    #                         self,
    #                         normal=self.VN,
    #                         efair=self.set_bezier_ctrlp_fairness,
    #                         is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                         is_one_or_another=is_1st_or_2nd, ##NOTE: different
    #                         is_checker=4, ##4 is the num of subdivision quadfaces
    #                         is_onlyinner=self.switch_if_ruling_denseinner,
    #                         is_dense=True,num_divide=self.num_bezier_divide,
    #                         is_modify=self.switch_if_ruling_rectify,
    #                         is_smooth=self.set_smooth_vertices_fairness)
    #                 else:
    #                     _,_,_,_,frame,annr,crvature = self.get_poly_quintic_Bezier_spline_crvs(
    #                         self,
    #                         normal=self.VN,
    #                         efair=self.set_bezier_ctrlp_fairness,
    #                         is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                         is_one_or_another=is_1st_or_2nd, ##NOTE: different
    #                         is_interpolate = self.switch_interpolate_checker,
    #                         is_onlyinner=self.switch_if_ruling_denseinner,
    #                         is_dense=True,num_divide=self.num_bezier_divide,
    #                         is_modify=self.switch_if_ruling_rectify,
    #                         is_smooth=self.set_smooth_vertices_fairness)
                    
    #                 _,an,r,_,nmlist = annr
    #                 _,E1,_,E3 = frm
    #                 if asym_or_geo:
    #                     "strip bounded by asym.crv."
    #                     unitN = E3
    #                 else:
    #                     "strip's center line is geodesic"
    #                     an = an-E3*dist/2
    #                     unitN = E3
    
    #                 strip = StraightStrip(an,unitN,E1,r,num_seg,dist,nmlist,
    #                                      efair=self.set_ply_rectify_strip_fairness,
    #                                      itera=50,ee=0.001)
    #                 sm = strip.get_strip()
    #                 return sm
    #             #-------------------------------------------------------------
    #         elif True:
    #             def _get_sm(is_1st_or_2nd):
    #                 if self.switch_singular_mesh:
    #                     from huilab.huimesh.singularMesh import get_singular_quintic_Bezier_spline_crvs_checker
                        
    #                 if self.switch_interpolate_checker:
    #                     "this is checker case"
    #                     if self.self.switch_singular_mesh:
    #                         _,_,_,_,frame,annr,crvature = get_singular_quintic_Bezier_spline_crvs_checker(
    #                             self,
    #                             normal=self.VN,
    #                             efair=self.set_bezier_ctrlp_fairness,
    #                             is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                             is_one_or_another=is_1st_or_2nd,
    #                             is_checker=3, ##3 is the num of subdivision quadfaces
    #                             is_dense=True,num_divide=self.num_bezier_divide,
    #                             is_smooth=self.set_smooth_vertices_fairness)
    #                     else:
    #                         _,_,_,_,frame,annr,crvature = self.get_poly_quintic_Bezier_spline_crvs_checker(
    #                             self,
    #                             normal=self.VN,
    #                             efair=self.set_bezier_ctrlp_fairness,
    #                             is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                             is_one_or_another=is_1st_or_2nd,
    #                             is_checker=4, ##4 is the num of subdivision quadfaces
    #                             is_onlyinner=self.switch_if_ruling_denseinner,
    #                             is_dense=True,num_divide=self.num_bezier_divide,
    #                             is_modify=self.switch_if_ruling_rectify,
    #                             is_smooth=self.set_smooth_vertices_fairness)
    #                 else:
    #                     "this is not checker case"
    #                     annr,frm,crvature = self.get_poly_quintic_Bezier_spline_crvs(
    #                         self,
    #                         normal=self.VN,
    #                         efair=self.set_bezier_ctrlp_fairness,
    #                         is_asym_or_geo=asym_or_geo,diagpoly=diagpoly,
    #                         is_one_or_another=is_1st_or_2nd,
    #                         is_interpolate = self.switch_interpolate_checker,
    #                         is_onlyinner=self.switch_if_ruling_denseinner,
    #                         is_dense=True,num_divide=self.num_bezier_divide,
    #                         is_modify=self.switch_if_ruling_rectify,
    #                         is_smooth=self.set_smooth_vertices_fairness)
                        
    #                 _,an,r,_,nmlist = annr
    #                 _,E1,_,E3 = frm
    #                 cos = np.abs(np.einsum('ij,ij->i',E3,r))
    #                 if self.switch_if_ruling_rectify and not self.switch_if_ruling_dense:
    #                     "rectify ruling by adding normals"
    #                     #print(np.mean(cos))
    #                     i = np.where(cos<np.mean(cos))[0]
    #                     r[i] = E3[i]
    #                     cos[i] = 1 
    #                 if asym_or_geo:
    #                     "strip bounded by asym.crv."
    #                     xrr = dist / cos   
    #                     vN = r * xrr[:,None]
    #                 else:
    #                     "strip's center line is geodesic"
    #                     xrr = dist / cos / 2
    #                     an = an-r*xrr[:,None]
    #                     vN = r*xrr[:,None]*2
    #                 is_smooth = self.set_smooth_vertices_fairness
    #                 sm = self.get_strip_from_rulings(an,vN,nmlist,is_smooth)
    #                 return sm
    #         sm1 = _get_sm(True)
    #         sm2 = _get_sm(False) ##geo:red:(240,114,114);asy:blue:(98,113,180)   
    #         showf1 = Faces(sm1,color =(240,114,114),name=name+'f1') #'golden_yellow'
    #         showe1 = Edges(sm1,color =(240,114,114),name=name+'e1')#'gold'
    #         showf2 = Faces(sm2,color =(240,114,114),name=name+'f2')
    #         showe2 = Edges(sm2,color =(240,114,114),name=name+'e2')
    #         poly1 = self.get_boundary_polyline(sm1) 
    #         selfmanager.plot_polyline(polyline=poly1,
    #                                        tube_radius=0.5*selfmanager.r ,
    #                                        glossy=1,
    #                                        color='black',
    #                                        sampling=100,
    #                                        name=name+'pl1') 
    #         poly2 = self.get_boundary_polyline(sm2) 
    #         selfmanager.plot_polyline(polyline=poly2,
    #                                        tube_radius=0.5*selfmanager.r ,
    #                                        glossy=1,
    #                                        color='black',
    #                                        sampling=100,
    #                                        name=name+'pl2') 
    #         selfmanager.add([showf1,showe1,showf2,showe2])
    #     else:
    #         selfmanager.remove(['NetSS12e1','NetSS12f1','NetSS12e2','NetSS12f2'])
    #         selfmanager.remove(['NetSS34e1','NetSS34f1','NetSS34e2','NetSS34f2'])
    #         selfmanager.remove(['NetSS12pl1','NetSS12pl2','NetSS34pl1','NetSS34pl2'])                 

        
    # ##@on_trait_change('show_1st_geo_strips_big,show_2nd_geo_strips_big,show_1st_geo_strips_small,show_2nd_geo_strips_small')
    # def plot_geodesic_strips(self):
    #     if self.show_1st_geo_strips_big or self.show_1st_geo_strips_small or self.show_2nd_geo_strips_big or self.show_2nd_geo_strips_small:
    #         if self.show_1st_geo_strips_big or self.show_1st_geo_strips_small:
    #               d = False  
    #               self.show_2nd_geo_strips_big = False
    #               self.show_2nd_geo_strips_small = False
    #         elif self.show_2nd_geo_strips_big or self.show_2nd_geo_strips_small:
    #               d = True
    #               self.show_1st_geo_strips_big = False
    #               self.show_1st_geo_strips_small = False
    #         diag = True if self.switch_opt_diag_or_iso_geodesic else False        
            
    #         if self.show_1st_geo_strips_big or self.show_2nd_geo_strips_big:
    #             big = True
    #         else:
    #             big=False
            
    #         sm,nmlist = self.get_strips_along_polylines(self.interval,diag,d,big)
    #         data = sm.face_planarity()

            
    # ##@on_trait_change('show_1st_planar_osculating_plane,show_2nd_planar_osculating_plane')
    # def plot_geodesic_osculating_plane(self):
    #     if self.show_1st_planar_osculating_plane or self.show_2nd_planar_osculating_plane:
    #         diag = True if self.switch_opt_diag_or_iso_geodesic else False    
    #         d = True if self.show_2nd_planar_osculating_plane else False
    #         sm,nmlist = self.get_normal_strips_along_polylines(
    #             self.interval,diag,another_direction=d,
    #             s=self.scale_dist_offset) # d should be same with opt-direction
    #         data = sm.face_planarity()
