"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
from numpy.lib.polynomial import poly1d
import rhino3dm
import numpy as np
#-------------------------------------------------------
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
path = path+'\ArchitecturalGeometry'
sys.path.append(path)###==C:\Users\WANGH0M\Documents\GitHub\ArchitecturalGeometry   

from geometrylab.optimization.gridshell import Gridshell
from huilab.huigridshell.gridshell_agnet import Gridshell_AGNet
from HOPS.hops_agnet import AGNet
#-------------------------------------------------------
#
# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)
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
    "/read_1_mesh",
    name="processMesh",
    nickname="pM",
    description="Process a mesh.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The mesh to process."),
        hs.HopsInteger("iteration","iter","num of iteration."),
        hs.HopsNumber("fairness","w1(fair)","weight of fairness.",default=0.005),
        hs.HopsNumber("closness","w2(closeness)","weight of self-closness.",default=0.01),
        hs.HopsNumber("direction","direction","direction of polyline.",default=0),
        hs.HopsString("web","web","constraint of net or web"),
        hs.HopsBoolean("Restart","Restart","Restart the optimization",default=False),
    ],
    outputs=[hs.HopsMesh("mesh", "mesh", "The new mesh."),
            hs.HopsPoint("an","Vertices","all vertices"),
            hs.HopsPoint("vn","Normal","normals at V")]
)
def main(mesh:rhino3dm.Mesh,n,w1,w2,d,web,restart=False):
    #assert isinstance(mesh, rhino3dm.Mesh)
    #m=rhino3dm.Mesh()
    ###-------------------------------------------
    M = _reading_mesh(mesh,opt=True)
    ###-------------------------------------------
    constraints = {
        'num_itera' : n,
        'weight_fairness' : w1,
        'weight_closeness' : w2,
        'direction_poly' : d,
        web : True,
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
### PLOTING CHECKER-VERTICES:
@hops.component(
    "/checker_tianvertex",
    name="checkerVertex",
    nickname="ckv",
    description="Get checker vertices.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh.")],
    outputs=[hs.HopsPoint("P1","P1","checker vertices P1"),
            hs.HopsPoint("P2","P2","checker vertices P2"),
            ]
)
def checker_tian_vertex(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    Vr,Vb = M.plot_checker_group_tian_select_vertices()
    Pr = [rhino3dm.Point3d(r[0],r[1],r[2]) for r in Vr]
    Pb = [rhino3dm.Point3d(b[0],b[1],b[2]) for b in Vb]
    return Pr,Pb
#==================================================================

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
    "/beziersss",
    name="bezier",
    nickname="bz",
    description="Get Bezier splines.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh."),
        hs.HopsString("web","web","constraint of net or web"),
        hs.HopsVector("VN","VN","vertex normals",access=hs.HopsParamAccess.LIST),
        hs.HopsInteger("i-th","ipoly","which polyline"), ## only 1,2,3,4
        hs.HopsNumber("weight","weight","which polyline",default=0.005),
        hs.HopsBoolean("checker","ck","switch if at checker-vertices",default=False),
        ],
    outputs=[
            hs.HopsPoint("ctrlP","P","all ctrl points"),
            hs.HopsPoint("an","V","anchor points"),
            hs.HopsPoint("e1","e1","unit tangent vector"),
            hs.HopsPoint("e2","e2","unit principal normal vector"),
            hs.HopsPoint("e3","e3","unit binormal vector"),
            hs.HopsPoint("r","r","unit ruling vector"),
            ]
)
def bezier(mesh:rhino3dm.Mesh,web,vn:rhino3dm.Vector3d,i,w,is_ck):
    x = [n.X for n in vn]
    y = [n.Y for n in vn]
    z = [n.Z for n in vn]
    VN = np.c_[x,y,z]
    setting = {
        web : True,
        'VN' : VN,
        'set_Poly' : i,
        'weight_CtrlP' : w,
        'is_Checker' : is_ck,
    }
    M = _reading_mesh(mesh,**setting)
    p,an,e1,e2,e3,r = M.set_quintic_bezier_splines()
    P = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in p]
    AN = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in an]
    E1 = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in e1]
    E2 = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in e2]
    E3 = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in e3]
    R = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in r]
    return P,AN,E1,E2,E3,R
#==================================================================

if __name__ == "__main__":
    #app.run(host="10.8.0.10",debug=True)
    app.run(debug=True)

