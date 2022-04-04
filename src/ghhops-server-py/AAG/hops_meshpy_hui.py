"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
#from numpy.lib.polynomial import poly1d
import rhino3dm
import numpy as np
# #-------------------------------------------------------
# import os
# import sys
# path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# path = path+'\ArchitecturalGeometry'
# sys.path.append(path)###==C:\Users\WANGH0M\Documents\GitHub\ArchitecturalGeometry   

from gridshell import Gridshell
from gridshell_agnet import Gridshell_AGNet
from hops_agnet import AGNet
#-------------------------------------------------------
#
# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)
#hops = hs.Hops(app,debug=True)
#-------------------------------------------------------



#==================================================================
#
#                      OPTIMIZATIONS
#
#==================================================================
def _reading_mesh(mesh,opt=False,**kwargs):
    ### GET THE VERTEXLIST+FACELIST OF INPUT MESH:
    vlist,flist = [],[]
    for i in range(len(mesh.Vertices)):
        vlist.append([mesh.Vertices[i].X,mesh.Vertices[i].Y,mesh.Vertices[i].Z])

    for i in range(mesh.Faces.Count):
        flist.append(list(mesh.Faces[i]))

    varr = np.array(vlist)
    farr = np.array(flist)

    if opt:
        ### meshpy.py-->quadrings.py-->gridshell.py (further related with guidedprojection)
        M = Gridshell()
    else:
        ### meshpy.py-->quadrings.py-->gridshell_agnet.py
        M = Gridshell_AGNet(**kwargs)
    M.make_mesh(varr, farr) ###Refer gui_basic.py/open_obj_file()
    return M

### MAIN OPTIMIZATION:
@hops.component(
    "/read_aaa_mesh",
    name="processMesh",
    nickname="pM",
    description="Process a mesh.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The mesh to process."),
        hs.HopsString("web","web","constraint of net or web"),
        hs.HopsNumber("direction","direction","0,1 for asy./geo. direction of AGG/GAA web.",default=0),
        hs.HopsInteger("iteration","iter","num of iteration.",default=10),
        hs.HopsNumber("fairness","w1(fair)","weight of fairness.",default=0.000),
        hs.HopsNumber("closness","w2(closeness)","weight of self-closness.",default=0.01),
        hs.HopsNumber("glide","w3(glide)","weight of gliding boundaries",default=0),
        hs.HopsInteger("glideBdry","index(bdry)","index of glided boundary(s)",access=hs.HopsParamAccess.LIST,default=0),
        hs.HopsNumber("fix","w4(fix)","weight of fixed vertices",default=0),
        hs.HopsInteger("fixVertices","index(fix)","index of fixed vertices",access=hs.HopsParamAccess.LIST,default=0),
        hs.HopsBoolean("Restart","Restart","Restart the optimization",default=False),
    ],
    outputs=[hs.HopsMesh("mesh", "mesh", "The new mesh."),
            hs.HopsPoint("an","Vertices","all vertices"),
            hs.HopsPoint("vn","Normal","normals at V")]
)
def main(mesh:rhino3dm.Mesh,web,d=0,n=10,w1=0.0000,w2=0.01,w3=0,ind3=0,w4=0,ind4=0,restart=True):
    #assert isinstance(mesh, rhino3dm.Mesh)
    #m=rhino3dm.Mesh()
    ###-------------------------------------------
    M = _reading_mesh(mesh,opt=True)
    ###-------------------------------------------
    constraints = {
        web : True,
        'direction_poly' : d,
        'num_itera' : n,
        'weight_fairness' : w1,
        'weight_closeness' : w2,
        'weight_gliding' : w3,
        'iGlideBdry' : ind3,
        'weight_fix' : w4,
        'ifixV' : ind4,
        'Restart' : restart,
    }
    ###-------------------------------------------
    ### OPTIMIZATION:
    AG = AGNet(**constraints)
    AG.mesh = M
    AG.optimize_mesh()
    ###-------------------------------------------
    ###-------------------------------------------
    ### MAKE RHINO MESH:
    Mout=rhino3dm.Mesh()

    for v in AG.vertexlist:
        Mout.Vertices.Add(v[0],v[1],v[2])
    
    for f in AG.facelist:
        Mout.Faces.AddFace(f[0],f[1],f[2],f[3])
    ###-------------------------------------------

    ### OUTPUT VERTICES + NORMALS:
    an,vn = AG.get_agweb_an_n_on()
    anchor = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in an]
    normal = [rhino3dm.Point3d(n[0],n[1],n[2]) for n in vn]
    return Mout,anchor,normal
#==================================================================



#==================================================================
#
#                      VISUALIZATIONS
#
#==================================================================
@hops.component(
    "/corner_vertex",
    name="cornerVertex",
    nickname="corner",
    description="Get corner vertices.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[
            hs.HopsInteger("index_corner","index","index of corner vertices",access=hs.HopsParamAccess.LIST),
            hs.HopsPoint("Pc","Pc","corner vertex"),
            ]
)
def corner_vertex(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    v,Vc = M.plot_corner_vertices()
    Pc = [rhino3dm.Point3d(r[0],r[1],r[2]) for r in Vc]
    print(v)
    return v,Pc

@hops.component(
    "/polyline_4_directions",
    name="polylines",
    nickname="pl",
    description="Get 4 familes of polylines.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[
            hs.HopsCurve("polyline1", "pl1", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline2", "pl2", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline3", "pl3", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline4", "pl4", access=hs.HopsParamAccess.LIST),
            ]
)
def polyline(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    V = M.vertexlist
    ipl11,ipl12,ipl21,ipl22,ipl31,ipl32,ipl41,ipl42 = M.plot_4family_polylines()
    pl1,pl2,pl3,pl4 = [],[],[],[]
    for i in range(len(ipl11)):
        a = rhino3dm.Point3d(V[ipl11][i][0],V[ipl11][i][1],V[ipl11][i][2])
        b = rhino3dm.Point3d(V[ipl12][i][0],V[ipl12][i][1],V[ipl12][i][2])
        pl1.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl21)):
        a = rhino3dm.Point3d(V[ipl21][i][0],V[ipl21][i][1],V[ipl21][i][2])
        b = rhino3dm.Point3d(V[ipl22][i][0],V[ipl22][i][1],V[ipl22][i][2])
        pl2.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl31)):
        a = rhino3dm.Point3d(V[ipl31][i][0],V[ipl31][i][1],V[ipl31][i][2])
        b = rhino3dm.Point3d(V[ipl32][i][0],V[ipl32][i][1],V[ipl32][i][2])
        pl3.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl41)):
        a = rhino3dm.Point3d(V[ipl41][i][0],V[ipl41][i][1],V[ipl41][i][2])
        b = rhino3dm.Point3d(V[ipl42][i][0],V[ipl42][i][1],V[ipl42][i][2])
        pl4.append(rhino3dm.LineCurve(a,b))

    return pl1,pl2,pl3,pl4
#==================================================================


@hops.component(
    "/vertices_matrix_and_flip",
    name="vertices",
    nickname="vs",
    description="Get vertices.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[
            hs.HopsPoint("pts","Vertices","all vertices"),
            hs.HopsPoint("pts_flip","Vertices_flip","all vertices_flip"),
            hs.HopsInteger("row","row","num of row."),
            hs.HopsInteger("column","column","num of column."),
            ]
)
def vertices(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    V = M.vertexlist
    m = M.get_vertices_matrix(rot=False)
    row,column = m.shape
    V1,V2 = V[m.flatten()],V[(np.flip(m, 1)).flatten()]
    Pt1 = [rhino3dm.Point3d(r[0],r[1],r[2]) for r in V1]
    Pt2 = [rhino3dm.Point3d(r[0],r[1],r[2]) for r in V2]
    return Pt1,Pt2,row,column
#==================================================================    
@hops.component(
    "/vertices_4_polyline_list",
    name="vertices",
    nickname="vs",
    description="Get 4 familes of polyline-vertices.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[
            hs.HopsPoint("pts","Vertices","all vertices"),
            hs.HopsNumber("net_pt1", "ipt1", access=hs.HopsParamAccess.LIST),
            hs.HopsNumber("net_pt2", "ipt2", access=hs.HopsParamAccess.LIST),
            hs.HopsNumber("diagnet_pt1", "ipt3", access=hs.HopsParamAccess.LIST),
            hs.HopsNumber("diagnet_pt2", "ipt4", access=hs.HopsParamAccess.LIST),
            ]
)
def polyline_vertices(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    V = M.vertexlist
    ipt1,ipt2,ipt3,ipt4 = M.plot_4family_vertices(rot=False)
    # pt1,pt2,pt3,pt4 = [],[],[],[]
    # for ip in ipt1:
    #     pl = []
    #     for i in ip:
    #         a = rhino3dm.Point3d(V[i][0],V[i][1],V[i][2])
    #         pl.append(a)
    #     pt1.append(pl)
    # for ip in ipt2:
    #     pl = []
    #     for i in ip:
    #         a = rhino3dm.Point3d(V[i][0],V[i][1],V[i][2])
    #         pl.append(a)
    #     pt2.append(pl)
    # for ip in ipt3:
    #     pl = []
    #     for i in ip:
    #         a = rhino3dm.Point3d(V[i][0],V[i][1],V[i][2])
    #         pl.append(a)
    #     pt3.append(pl)
    # for ip in ipt4:
    #     pl = []
    #     for i in ip:
    #         a = rhino3dm.Point3d(V[i][0],V[i][1],V[i][2])
    #         pl.append(a)
    #     pt4.append(pl)
    # print(type(ipt1[0]),type(ipt2[0]),type(ipt3[0]),type(ipt4[0]))
    # print(ipt1[0],ipt2[0],ipt3[0],ipt4[0])
    # print(V)
    Pt = [rhino3dm.Point3d(r[0],r[1],r[2]) for r in V]
    return Pt, ipt1,ipt2,ipt3,ipt4
#==================================================================

@hops.component(
    "/patch_net",
    name="patch_crvnetwork",
    nickname="network",
    description="Get 2 familes of polylines.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[
            hs.HopsCurve("polyline1", "pl1", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline2", "pl2", access=hs.HopsParamAccess.LIST),
            ]
)
def patch(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    V = M.vertexlist
    ipl11,ipl12,ipl21,ipl22 = M.plot_2family_polylines(rot=False)
    pl1,pl2 = [],[]
    for i in range(len(ipl11)):
        a = rhino3dm.Point3d(V[ipl11][i][0],V[ipl11][i][1],V[ipl11][i][2])
        b = rhino3dm.Point3d(V[ipl12][i][0],V[ipl12][i][1],V[ipl12][i][2])
        pl1.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl21)):
        a = rhino3dm.Point3d(V[ipl21][i][0],V[ipl21][i][1],V[ipl21][i][2])
        b = rhino3dm.Point3d(V[ipl22][i][0],V[ipl22][i][1],V[ipl22][i][2])
        pl2.append(rhino3dm.LineCurve(a,b))
    return pl1,pl2
#==================================================================
@hops.component(
    "/rotational_net",
    name="rot_crvnetwork",
    nickname="network",
    description="Get 2 familes of polylines.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[
            hs.HopsCurve("polyline1", "pl1", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline2", "pl2", access=hs.HopsParamAccess.LIST),
            ]
)
def rotational(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    V = M.vertexlist
    ipl11,ipl12,ipl21,ipl22 = M.plot_2family_polylines(rot=True)
    pl1,pl2 = [],[]
    for i in range(len(ipl11)):
        a = rhino3dm.Point3d(V[ipl11][i][0],V[ipl11][i][1],V[ipl11][i][2])
        b = rhino3dm.Point3d(V[ipl12][i][0],V[ipl12][i][1],V[ipl12][i][2])
        pl1.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl21)):
        a = rhino3dm.Point3d(V[ipl21][i][0],V[ipl21][i][1],V[ipl21][i][2])
        b = rhino3dm.Point3d(V[ipl22][i][0],V[ipl22][i][1],V[ipl22][i][2])
        pl2.append(rhino3dm.LineCurve(a,b))
    return pl1,pl2
#==================================================================
@hops.component(
    "/patch_diag_polylines",
    name="patch_2diag",
    nickname="2diagonals",
    description="Get 2 familes of diagonal polylines.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[
            hs.HopsCurve("polyline1", "pl1", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline2", "pl2", access=hs.HopsParamAccess.LIST),
            ]
)
def patch_diag(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    V = M.vertexlist
    ipl11,ipl12,ipl21,ipl22 = M.plot_2family_diagonal_polylines()
    pl1,pl2 = [],[]
    for i in range(len(ipl11)):
        a = rhino3dm.Point3d(V[ipl11][i][0],V[ipl11][i][1],V[ipl11][i][2])
        b = rhino3dm.Point3d(V[ipl12][i][0],V[ipl12][i][1],V[ipl12][i][2])
        pl1.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl21)):
        a = rhino3dm.Point3d(V[ipl21][i][0],V[ipl21][i][1],V[ipl21][i][2])
        b = rhino3dm.Point3d(V[ipl22][i][0],V[ipl22][i][1],V[ipl22][i][2])
        pl2.append(rhino3dm.LineCurve(a,b))
    return pl1,pl2
#==================================================================

@hops.component(
    "/boundary",
    name="boundaries",
    nickname="bdry",
    description="Get 4 or 2 boundary polylines.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "Input mesh.")],
    outputs=[
            hs.HopsCurve("polyline1", "pl1", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline2", "pl2", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline3", "pl3", access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("polyline4", "pl4", access=hs.HopsParamAccess.LIST),
            ]
)
def boundaries(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    V = M.vertexlist
    ipl11,ipl12 = M.plot_boundary_polylines(0)
    ipl21,ipl22 = M.plot_boundary_polylines(1)
    ipl31,ipl32 = M.plot_boundary_polylines(2)
    ipl41,ipl42 = M.plot_boundary_polylines(3)
    pl1,pl2,pl3,pl4 = [],[],[],[]
    for i in range(len(ipl11)):
        a = rhino3dm.Point3d(V[ipl11][i][0],V[ipl11][i][1],V[ipl11][i][2])
        b = rhino3dm.Point3d(V[ipl12][i][0],V[ipl12][i][1],V[ipl12][i][2])
        pl1.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl21)):
        a = rhino3dm.Point3d(V[ipl21][i][0],V[ipl21][i][1],V[ipl21][i][2])
        b = rhino3dm.Point3d(V[ipl22][i][0],V[ipl22][i][1],V[ipl22][i][2])
        pl2.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl31)):
        a = rhino3dm.Point3d(V[ipl31][i][0],V[ipl31][i][1],V[ipl31][i][2])
        b = rhino3dm.Point3d(V[ipl32][i][0],V[ipl32][i][1],V[ipl32][i][2])
        pl3.append(rhino3dm.LineCurve(a,b))
    for i in range(len(ipl41)):
        a = rhino3dm.Point3d(V[ipl41][i][0],V[ipl41][i][1],V[ipl41][i][2])
        b = rhino3dm.Point3d(V[ipl42][i][0],V[ipl42][i][1],V[ipl42][i][2])
        pl4.append(rhino3dm.LineCurve(a,b))
    return pl1,pl2,pl3,pl4
#==================================================================

#==================================================================

if __name__ == "__main__":
    #app.run(host="10.8.0.10",debug=True)
    #app.run(host="10.8.0.12",debug=True) ##Hui's VPN
    app.run(debug=True)

