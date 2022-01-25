"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
import rhino3dm
import numpy as np

# #-------------------------------------------------------
# import os
# import sys
# path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# path = path+'\ArchitecturalGeometry'
# sys.path.append(path)###==C:\Users\WANGH0M\Documents\GitHub\ArchitecturalGeometry   

# from huilab.huigridshell.gridshell_agnet import Gridshell_AGNet
# from HOPS.hops_agnet import AGNet
# #-------------------------------------------------------

# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)
#-------------------------------------------------------

#==================================================================
### OUTPUT MULTI-POINTS FROM MUTI-LINES:
@hops.component(
    "/lines",
    name="Lines",
    inputs=[
        hs.HopsLine("Line", "L", access=hs.HopsParamAccess.LIST),
        hs.HopsNumber("i", "i"),
    ],
    outputs=[
        hs.HopsPoint("P", "P")
    ]
)
def lines(lines: rhino3dm.Line, i):
    points = [line.PointAt(i) for line in lines]
    return points
#==================================================================



#==================================================================
### TEST TREE (NOT WORK)!
def np_float_array_to_hops_tree(np_array: np.array, paths: list = []):
    """
    Converts a numpy array to a Hops DataTree (dict with paths as keys).
    """
    if not paths:
        paths = ["{0;" + str(x) + "}" for x in range(np_array.shape[0])]
    tree = {}
    for i, branch in enumerate(np_array):
        try:
            tree[paths[i].strip("}{")] = [float(v) for v in branch]
        except:
            tree[paths[i].strip("}{")] = [float(branch)]
    return tree

@hops.component(
    "/arrayss",
    name="arr",
    nickname="arr",
    description="arr.",
    inputs=[hs.HopsInteger("iter","n","num of iteration.")],
    #outputs=[hs.HopsString("x","x","x")]
    outputs=[hs.HopsNumber("x","x","x"), hs.HopsParamAccess.TREE]
)
def arrayss(n):
    m = np.arange(n)
    x = np_float_array_to_hops_tree(m)
    print(x)
    return x
#==================================================================

# pt1,pt2,pt3,pt4 = [],[],[],[]
    # for pl in ipl1:
    #     pt1.append([rhino3dm.Point3d(V[ip][0],V[ip][1],V[ip][2]) for ip in pl])
    # for pl in ipl2:
    #     pt2.append([rhino3dm.Point3d(V[ip][0],V[ip][1],V[ip][2]) for ip in pl])
    # for pl in ipl3:
    #     pt3.append([rhino3dm.Point3d(V[ip][0],V[ip][1],V[ip][2]) for ip in pl])
    # for pl in ipl4:
    #     pt4.append([rhino3dm.Point3d(V[ip][0],V[ip][1],V[ip][2]) for ip in pl])
# @hops.component(
#     "/input_mesh",
#     name="processMesh",
#     nickname="pM",
#     description="Process a mesh.",
#     inputs=[
#         hs.HopsMesh("mesh", "m", "The mesh to process."),
#         hs.HopsInteger("iter","n","num of iteration."),
#         hs.HopsNumber("fairness","w1","weight of fairness.",default=0.005),
#         hs.HopsNumber("closness","w2","weight of self-closness.",default=0.01),
#         hs.HopsBoolean("Anet","Anet","constraint of A-net",default=False),
#         hs.HopsBoolean("diag_Anet","AnetDiag","constraint of diagonal A-net",default=False),
#         hs.HopsBoolean("Gnet","Gnet","constraint of G-net",default=False),
#         hs.HopsBoolean("diag_Gnet","GnetDiag","constraint of diagonal G-net",default=False),
#         hs.HopsNumber("direction","d","direction of polyline.",default=0),
#         hs.HopsBoolean("AAG","AAG","constraint of AAG-web",default=False),
#         hs.HopsBoolean("GAA","GAA","constraint of GAA-web",default=False),
#         hs.HopsBoolean("AGG","AGG","constraint of AGG-web",default=False),
#         hs.HopsBoolean("GGA","GGA","constraint of GGA-web",default=False),
#         hs.HopsBoolean("AAGG","AAGG","constraint of AAGG-web",default=False),
#         hs.HopsBoolean("GGAA","GGAA","constraint of GGAA-web",default=False),
#         hs.HopsBoolean("Restart","Restart","Restart the optimization",default=False),
#     ],
#     outputs=[hs.HopsMesh("mesh", "m", "The new mesh."),
#             hs.HopsNumber("a","a","parameter a")]
# )
# def input_mesh(mesh:rhino3dm.Mesh,n,w1,w2,\
#     anet=False,anet_diagnet=False,gnet=False,gnet_diagnet=False,\
#     d=0,aag=False,gaa=False,agg=False,gga=False,aagg=False,ggaa=False,\
#     restart=False):
#     #assert isinstance(mesh, rhino3dm.Mesh)
#     #m=rhino3dm.Mesh()

#     ###-------------------------------------------
#     ### GET THE VERTEXLIST+FACELIST OF INPUT MESH:
#     vlist,flist = [],[]
#     for i in range(len(mesh.Vertices)):
#         vlist.append([mesh.Vertices[i].X,mesh.Vertices[i].Y,mesh.Vertices[i].Z])

#     for i in range(mesh.Faces.Count):
#         flist.append(list(mesh.Faces[i]))

#     varr = np.array(vlist)
#     farr = np.array(flist)
#     ###-------------------------------------------

#     ###-------------------------------------------
#     ### ASSIGN TO GEOMETRYLAB:
#     M = Gridshell()
#     M.make_mesh(varr, farr) ###Refer gui_basic.py/open_obj_file()
#     ###-------------------------------------------
#     constraints = {
#         'num_itera' : n,
#         'weight_fairness' : w1,
#         'weight_closeness' : w2,
#         'direction_poly' : d,
#         'Anet' : anet,
#         'AnetDiagnet' : anet_diagnet,
#         'Gnet' : gnet,
#         'GnetDiagnet' : gnet_diagnet,
#         'AAG' : aag,
#         'GAA' : gaa,
#         'AGG' : agg,
#         'GGA' : gga,
#         'AAGG' : aagg,
#         'GGAA' : ggaa,
#         'Restart' : restart,
#     }
#     ###-------------------------------------------
#     ### OPTIMIZATION:
#     AG = AGNet(**constraints)
#     AG.mesh = M
#     AG.optimize_mesh()
#     ###-------------------------------------------
#     ###-------------------------------------------
#     ### MAKE RHINO MESH:
#     Mout=rhino3dm.Mesh()
#     for v in AG.vertexlist:
#         Mout.Vertices.Add(v[0],v[1],v[2])
    
#     for f in AG.facelist:
#         Mout.Faces.AddFace(f[0],f[1],f[2],f[3])
#     ###-------------------------------------------

#     b=3
#     return Mout, b



# @hops.component(
#     "/read_1mesh",
#     name="processMesh",
#     nickname="pM",
#     description="Process a mesh.",
#     inputs=[
#         hs.HopsMesh("mesh", "m", "The mesh to process."),
#         hs.HopsInteger("iter","n","num of iteration."),
#         hs.HopsNumber("fairness","w1","weight of fairness.",default=0.005),
#         hs.HopsNumber("closness","w2","weight of self-closness.",default=0.01),
#         hs.HopsNumber("direction","d","direction of polyline.",default=0),
#         hs.HopsString("web","web","constraint of net or web"),
#         hs.HopsBoolean("Restart","Restart","Restart the optimization",default=False),
#     ],
#     outputs=[hs.HopsMesh("mesh", "m", "The new mesh."),
#             hs.HopsPoint("an","V","all vertices"),
#             hs.HopsPoint("vn","N","normals at V")]
# )
# def read_1mesh(mesh:rhino3dm.Mesh,n,w1,w2,d,web,restart=False):
#     #assert isinstance(mesh, rhino3dm.Mesh)
#     #m=rhino3dm.Mesh()
#     ###-------------------------------------------
#     ### GET THE VERTEXLIST+FACELIST OF INPUT MESH:
#     vlist,flist = [],[]
#     for i in range(len(mesh.Vertices)):
#         vlist.append([mesh.Vertices[i].X,mesh.Vertices[i].Y,mesh.Vertices[i].Z])

#     for i in range(mesh.Faces.Count):
#         flist.append(list(mesh.Faces[i]))

#     varr = np.array(vlist)
#     farr = np.array(flist)
#     ###-------------------------------------------

#     ###-------------------------------------------
#     ### ASSIGN TO GEOMETRYLAB:
#     M = Gridshell()
#     M.make_mesh(varr, farr) ###Refer gui_basic.py/open_obj_file()
#     ###-------------------------------------------
#     constraints = {
#         'num_itera' : n,
#         'weight_fairness' : w1,
#         'weight_closeness' : w2,
#         'direction_poly' : d,
#         web : True,
#         'Restart' : restart,
#     }
#     ###-------------------------------------------
#     ### OPTIMIZATION:
#     AG = AGNet(**constraints)
#     AG.mesh = M
#     AG.optimize_mesh()
#     ###-------------------------------------------
#     ###-------------------------------------------
#     ### MAKE RHINO MESH:
#     Mout=rhino3dm.Mesh()
#     for v in AG.vertexlist:
#         Mout.Vertices.Add(v[0],v[1],v[2])
    
#     for f in AG.facelist:
#         Mout.Faces.AddFace(f[0],f[1],f[2],f[3])
#     ###-------------------------------------------

#     an,vn = AG.get_agweb_an_n_on()
#     anchor = rhino3dm.Point3dList()
#     for a in an:
#         anchor.Add(a[0],a[1],a[2])

#     normal = rhino3dm.Point3dList()
#     for n in vn:
#         normal.Add(n[0],n[1],n[2])

#     return Mout,anchor,normal




    def get_poly_quintic_Bezier_spline_crvs_checker(self,mesh,normal,
                                                    efair=0.01,
                                                    is_asym_or_geo=True,
                                                    diagpoly=False,
                                                    is_one_or_another=False,
                                                    is_checker=1,
                                                    is_Frame=False,
                                                    is_onlyinner=False,
                                                    is_ruling=False,
                                                    is_dense=False,num_divide=5,
                                                    is_modify=False,
                                                    is_smooth=0.0):
        "For each polyline, 1.[Pi,Ti,Ni,ki] 2.opt to get ctrl-p,polygon,crv"
        from huilab.huimesh.curvature import frenet_frame
        from huilab.huimesh.bezierC2Continuity import BezierSpline
        V = mesh.vertices
        N = normal ### surf_normal n
        iall = self.get_both_isopolyline(diagpoly,is_one_or_another,only_inner=False)
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
 
        seg_q1234,seg_vl, seg_vr = np.array([0,0,0]),[],[]
        ctrlP = []
        kck = is_checker
        
        if kck !=1:
            if self._rot_patch_matrix is not None:
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
        for iv in iall:
            "Except two endpoints on the boudnary"
            bool_value=False
            if kck !=1:
                ###"if only-patch-shape:"
                bool_value = kck !=0 and 0<num_poly and num_poly<num_allpoly and num_poly%(kck)==0 and len(iv)>=(kck*3+1)
                ###"if rotational-shape"#TODO
                #bool_value = kck !=0 and num_poly%(kck)==0 and len(iv)>=(kck*3+1):
            else:
                bool_value = len(iv)>=4

            if bool_value:
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
                    seg_vl.extend(iv_ck[:-1])
                    seg_vr.extend(iv_ck[1:])
                else:
                    iv_ck = iv[1:-1]
                    seg_vl.extend(iv[1:-2])
                    seg_vr.extend(iv[2:-1])

                bs = BezierSpline(degree=len(Pi)-1,continuity=3,
                                  efair=efair,itera=200,
                                  endpoints=Pi,tangents=Ti,normals=E3i)

                if is_Frame:
                    p, pl, pr = bs.control_points(is_points=True,is_seg=True)
                    P,Pl,Pr = np.vstack((P,p)), np.vstack((Pl,pl)), np.vstack((Pr,pr))
                    ctrlP.append(p)
                    crvp,crvpl,crvpr = bs.control_points(is_curve=True,is_seg=True)
                    crvPl, crvPr= np.vstack((crvPl,crvpl)), np.vstack((crvPr,crvpr))
                    
                    an = np.vstack((an,Pi))
                    oNi = np.cross(E3i,Ti)
                    frm1 = np.vstack((frm1,Ti))  ## Frenet-E1
                    frm2 = np.vstack((frm2,oNi)) ## Frenet-E2
                    frm3 = np.vstack((frm3,E3i)) ## Frenet-E3
                    #ruling:
                    kg,kn,k,tau,d = bs.get_curvature(is_asym_or_geo)
                    
                    row_list.append(len(d))
                    
                    if False:
                        i = np.where(np.einsum('ij,ij->i',E3i,d)<0)[0]
                        d[i] = -d[i]

                    ruling = np.vstack((ruling,d))
                    all_kg,all_kn = np.r_[all_kg,kg],np.r_[all_kn,kn]
                    all_k,all_tau = np.r_[all_k,k],np.r_[all_tau,tau]
                    arr = np.r_[arr, np.arange(len(iv_ck)-1) + num]
                    num += len(iv_ck)
                    varr = np.r_[varr,iv_ck]
                    
                elif is_ruling and is_dense:
                    kg,kn,k,tau,pts,d,frmi = bs.get_curvature(is_asym_or_geo,
                                                            True,num_divide,
                                                            is_onlyinner,
                                                            is_modify,
                                                            is_smooth)
                    an = np.vstack((an,pts))
                    frm1 = np.vstack((frm1,frmi[0])) ## Frenet-E1
                    frm2 = np.vstack((frm2,frmi[1])) ## Frenet-E2
                    frm3 = np.vstack((frm3,frmi[2])) ## Frenet-E3
                    
                    # if False:
                    #     i = np.where(np.einsum('ij,ij->i',frmi[2],d)<0)[0]
                    #     d[i] = -d[i]
                    # elif False:
                    #     d = -d
                    
                    if is_smooth:
                        from huilab.huimesh.smooth import fair_vertices_or_vectors
                        Pup = fair_vertices_or_vectors(pts+d,itera=10,efair=is_smooth)
                        d = Pup-pts
                        d = d / np.linalg.norm(d,axis=1)[:,None]
                    
                    ruling = np.vstack((ruling,d))
                    all_kg,all_kn = np.r_[all_kg,kg],np.r_[all_kn,kn]
                    all_k,all_tau = np.r_[all_k,k],np.r_[all_tau,tau]
                    arr = np.r_[arr, np.arange(len(kg)-1) + num]
                    num += len(kg)
                    
                    row_list.append(len(d))
                else:
                    seg_q1234 = np.vstack((seg_q1234,bs.control_points(is_Q1234=True)))
                    
            num_poly +=1
        
        if is_ruling and is_dense:
             return [varr,an[1:],ruling[1:],arr,row_list],\
                 [an[1:],frm1[1:],frm2[1:],frm3[1:]],\
                     [all_kg,all_kn,all_k,all_tau]
                    
        if is_Frame:
            P, Pl, Pr, crvPl, crvPr = P[1:],Pl[1:],Pr[1:],crvPl[1:],crvPr[1:]   
            polygon = self.make_polyline_from_endpoints(Pl, Pr)
            crv = self.make_polyline_from_endpoints(crvPl, crvPr)
            return P,polygon,crv,np.array(ctrlP,dtype=object),\
                   [an[1:],frm1[1:],frm2[1:],frm3[1:]],\
                   [varr,an[1:],ruling[1:],arr,row_list],\
                   [all_kg,all_kn,all_k,all_tau]
        else:
            return seg_q1234[1:],[seg_vl,seg_vr]