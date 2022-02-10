
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
            if self.n==2:
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
                    #S0,S5 = np.cross(N0,E1[0]), np.cross(N5,E1[-1])
                    ##print(di.shape,S0.shape,S5.shape,N0.shape,N5.shape,E1.shape)
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
                    di = fair_vertices_or_vectors(di,itera=10,efair=weight)
                    di = di / np.linalg.norm(di,axis=1)[:,None]
                    E3 = fair_vertices_or_vectors(E3,itera=10,efair=weight)
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
                                if np.dot(N0[i],di[0])<0.8:
                                    di[0] = N0[i]
                                if np.dot(N5[i],di[-1])<0:
                                    di[-1] = N5[i]
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
                            if np.dot(S0[i],di[0])<0.8:
                                di[0] = S0[i]
                            if np.dot(S5[i],di[-1])<0:
                                di[-1] = S5[i]
                    #--------------------------
                    
                    #--------------------------
                    if is_smooth:
                        weight = is_smooth
                        di = fair_vertices_or_vectors(di,itera=10,efair=weight*1)
                        di = di / np.linalg.norm(di,axis=1)[:,None]
                        E3 = fair_vertices_or_vectors(E3,itera=10,efair=weight*1)
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

    # def get_poly_quintic_Bezier_spline_crvs_checker(self,mesh,normal,
    #                                                 efair=0.01,
    #                                                 is_asym_or_geo=True,
    #                                                 diagpoly=False,
    #                                                 is_one_or_another=False,
    #                                                 is_checker=1,
    #                                                 is_dense=False,num_divide=5,
    #                                                 is_modify=False,
    #                                                 is_smooth=0.0):
    #     "For each polyline, 1.[Pi,Ti,Ni,ki] 2.opt to get ctrl-p,polygon,crv"
    #     from curvature import frenet_frame
    #     from bezierC2Continuity import BezierSpline
    #     V = mesh.vertices
    #     N = normal ### surf_normal n
    #     kck = is_checker
    #     inner = False if kck!=1 else True 
    #     iall = self.get_both_isopolyline(diagpoly,is_one_or_another,only_inner=inner)
    #     an = np.array([0,0,0])
    #     ruling = np.array([0,0,0])
    #     all_kg=all_kn=all_k=all_tau=np.array([])
    #     arr = np.array([],dtype=int)
    #     varr = np.array([],dtype=int)
    #     num = 0
    #     P=Pl=Pr = np.array([0,0,0])
    #     crvPl=crvPr = np.array([0,0,0])
    #     frm1=frm2=frm3 = np.array([0,0,0])
 
    #     #seg_q1234,seg_vl, seg_vr = np.array([0,0,0]),[],[]
    #     ctrlP = []

    #     if kck !=1:
    #         if self._rot_patch_matrix is not None:
    #             num_poly = 1
    #             if diagpoly:
    #                 num_poly -= 1 ## -=2 IF PATCH, ELIF ROTATIOANL =-1
    #         elif self.patch_matrix is not None:
    #             num_poly,num_allpoly = 0,len(iall)-1
    #             if diagpoly:
    #                 ##NOTE: TOSELECT:
    #                 num_poly -= 2 ## -=2 IF PATCH, ELIF ROTATIOANL =-1
    #                 num_allpoly += 2
            
    #     row_list = []
    #     dense_row_list = []
    #     for iv in iall:
    #         "Except two endpoints on the boudnary"
    #         bool_value=False
    #         if kck !=1:
    #             if len(self.corner)!=0:
    #                 "if only-patch-shape:"
    #                 bool_value = kck !=0 and 0<num_poly and num_poly<num_allpoly and num_poly%(kck)==0 and len(iv)>=(kck*3+1)
    #             else:
    #                 "if rotational-shape"
    #                 bool_value = kck !=0 and num_poly%(kck)==0 and len(iv)>=(kck*3+1)
    #         else:
    #             bool_value = len(iv)>=4

    #         if bool_value:
    #             if kck==1:
    #                 iv_sub,iv_sub1,iv_sub3 = iv[1:-1],iv[:-2],iv[2:]
    #             else:
    #                 iv_sub,iv_sub1,iv_sub3 = iv[kck:-kck],iv[kck-1:-kck-1],iv[kck+1:-kck+1]
    #             Pi = V[iv_sub] # n-8
    #             frame = frenet_frame(Pi,V[iv_sub1],V[iv_sub3])
    #             "using surface normal computed by A-net or G-net"
    #             Ti,Ef2,Ef3 = frame ### Frenet frame (E1,E2,E3)
    #             Ni = N[iv_sub] ### SURF-NORMAL
    #             "if asym: binormal==Ni; elif geo: binormal == t x Ni"
    #             #E3i = Ni if is_asym_or_geo else np.cross(Ti,Ni)
    #             if is_asym_or_geo:
    #                 "asymptotic; orient binormal with surf-normal changed at inflections"
    #                 E3i = Ni
    #                 #i = np.where(np.einsum('ij,ij->i',Ef3,E3i)<0)[0]
    #                 #E3i[i] = -E3i[i]
    #             else:
    #                 "geodesic"
    #                 E3i = np.cross(Ti,Ni)

    #             if kck !=1:
    #                 "checker_vertex_partial_of_submesh case"
    #                 Pi = Pi[::kck]
    #                 Ti = Ti[::kck]
    #                 E3i = E3i[::kck]
    #                 iv_ck = iv_sub[::kck]
    #                 #seg_vl.extend(iv_ck[:-1])
    #                 #seg_vr.extend(iv_ck[1:])
    #             else:
    #                 iv_ck = iv[1:-1]
    #                 #seg_vl.extend(iv[1:-2])
    #                 #seg_vr.extend(iv[2:-1])

    #             bs = BezierSpline(degree=5,continuity=3,
    #                               efair=efair,itera=200,
    #                               endpoints=Pi,tangents=Ti,normals=E3i)

    #             p, pl, pr = bs.control_points(is_points=True,is_seg=True)
    #             P,Pl,Pr = np.vstack((P,p)), np.vstack((Pl,pl)), np.vstack((Pr,pr))
    #             ctrlP.append(p)
    #             crvp,crvpl,crvpr = bs.control_points(is_curve=True,is_seg=True)
    #             crvPl, crvPr= np.vstack((crvPl,crvpl)), np.vstack((crvPr,crvpr))
                
    #             row_list.append(len(Pi))
    #             if not is_dense:
    #                 kg,kn,k,tau,d = bs.get_curvature(is_asym_or_geo)
    #                 an = np.vstack((an,Pi))
    #                 oNi = np.cross(E3i,Ti)
    #                 frm1 = np.vstack((frm1,Ti))  ## Frenet-E1
    #                 frm2 = np.vstack((frm2,oNi)) ## Frenet-E2
    #                 frm3 = np.vstack((frm3,E3i)) ## Frenet-E3

    #                 if False:
    #                     i = np.where(np.einsum('ij,ij->i',E3i,d)<0)[0]
    #                     d[i] = -d[i]

    #                 arr = np.r_[arr, np.arange(len(iv_ck)-1) + num]
    #                 num += len(iv_ck)
    #                 varr = np.r_[varr,iv_ck]

    #             else:
    #                 kg,kn,k,tau,pts,d,frmi = bs.get_curvature(is_asym_or_geo,
    #                                                         True,num_divide,
    #                                                         is_modify,
    #                                                         is_smooth)
    #                 dense_row_list.append(len(d))
    #                 an = np.vstack((an,pts))
    #                 frm1 = np.vstack((frm1,frmi[0])) ## Frenet-E1
    #                 frm2 = np.vstack((frm2,frmi[1])) ## Frenet-E2
    #                 frm3 = np.vstack((frm3,frmi[2])) ## Frenet-E3
                    
    #                 # if False:
    #                 #     i = np.where(np.einsum('ij,ij->i',frmi[2],d)<0)[0]
    #                 #     d[i] = -d[i]
    #                 # elif False:
    #                 #     d = -d
                    
    #                 if is_smooth:
    #                     from smooth import fair_vertices_or_vectors
    #                     Pup = fair_vertices_or_vectors(pts+d,itera=10,efair=is_smooth)
    #                     d = Pup-pts
    #                     d = d / np.linalg.norm(d,axis=1)[:,None]
                        
    #                 arr = np.r_[arr, np.arange(len(kg)-1) + num]
    #                 num += len(kg)
                    
    #             ruling = np.vstack((ruling,d))
    #             all_kg,all_kn = np.r_[all_kg,kg],np.r_[all_kn,kn]
    #             all_k,all_tau = np.r_[all_k,k],np.r_[all_tau,tau]
    #             #seg_q1234 = np.vstack((seg_q1234,bs.control_points(is_Q1234=True)))
                
    #         if kck !=1:        
    #             num_poly +=1
        
    #     P, Pl, Pr, crvPl, crvPr = P[1:],Pl[1:],Pr[1:],crvPl[1:],crvPr[1:]   
    #     polygon = None #self.make_polyline_from_endpoints(Pl, Pr)
    #     crv = None #self.make_polyline_from_endpoints(crvPl, crvPr)
    #     return P,polygon,crv,np.array(ctrlP,dtype=object),\
    #            [an[1:],frm1[1:],frm2[1:],frm3[1:]],\
    #            [varr,an[1:],ruling[1:],arr,row_list,dense_row_list],\
    #            [all_kg,all_kn,all_k,all_tau]
    #     #return seg_q1234[1:],[seg_vl,seg_vr]
        
        

    # def regular_vertex_regular_neighbour_checkerboard_order(self): try
    #     "vertex_rr_check_ind: seperate regular-regular vertex into blue/red"
    #     H = self.halfedges
    #     idb,idr = self.vertex_check_ind
    #     blue,red = self.vertex_check
    #     a,b=[],[]
    #     for i in range(len(blue)):
    #         v = blue[i]
    #         ei = np.where(H[:,0]==v)[0]
    #         ej = H[H[H[H[ei,2],2],2],2]
    #         if any(list(ej-ei)):
    #             continue
    #         else:
    #             a.append(i)
    #     for i in range(len(red)):
    #         v = red[i]
    #         ei = np.where(H[:,0]==v)[0]
    #         ej = H[H[H[H[ei,2],2],2],2]
    #         if any(list(ej-ei)):
    #             continue
    #         else:
    #             b.append(i)
    #     #self._vertex_rr_check_ind=[idb[a],idr[b]]
    #     self._vertex_rr_check_ind=[np.array(a),np.array(b)]
    