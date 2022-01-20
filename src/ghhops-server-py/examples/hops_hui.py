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
    "/read_a_mesh",
    name="processMesh",
    nickname="pM",
    description="Process a mesh.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The mesh to process."),
        hs.HopsBoolean("singulrMesh","singular/regular","switch if singular or regular mesh",default=False),
        hs.HopsInteger("iteration","iter","num of iteration."),
        hs.HopsNumber("fairness","w1(fair)","weight of fairness.",default=0.005),
        hs.HopsNumber("closness","w2(closeness)","weight of self-closness.",default=0.01),
        hs.HopsNumber("direction","direction","0,1 for asy./geo. direction of AGG/GAA web.",default=0),
        hs.HopsString("web","web","constraint of net or web"),
        hs.HopsBoolean("Restart","Restart","Restart the optimization",default=False),
    ],
    outputs=[hs.HopsMesh("mesh", "mesh", "The new mesh."),
            hs.HopsPoint("an","Vertices","all vertices"),
            hs.HopsPoint("vn","Normal","normals at V")]
)
def main(mesh:rhino3dm.Mesh,is_singular,n,w1,w2,d,web,restart=False):
    #assert isinstance(mesh, rhino3dm.Mesh)
    #m=rhino3dm.Mesh()
    ###-------------------------------------------
    M = _reading_mesh(mesh,opt=True)
    ###-------------------------------------------
    constraints = {
        'is_Singular' : is_singular,
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
    "/Bezier_Spline_strips",
    name="bezier",
    nickname="bz",
    description="Get Bezier splines.",
    inputs=[
        hs.HopsMesh("mesh", "Mesh", "The optimized mesh."),
        hs.HopsBoolean("singulrMesh","singular/regular","switch if singular or regular mesh",default=False),
        hs.HopsString("web","web","constraint of net or web"),
        hs.HopsVector("VN","VN","vertex normals",access=hs.HopsParamAccess.LIST),
        hs.HopsInteger("i-th","ipoly","which polyline"),
        hs.HopsNumber("weight1","w1(CtrlPoint)","fairness of Bezier ctrl-points",default=0.005),
        hs.HopsBoolean("checker","ck/all","switch if at checker-vertices",default=True),
        hs.HopsNumber("numChecker","numChecker","number of checker selection",default=4),
        hs.HopsBoolean("rectify by E3","optRuling/cmptRuling","switch if optimized or directly computed rulings",default=False),
        hs.HopsNumber("weight2","w2(Strip)","fairness of (unrolled) developable strip",default=0.005),
        hs.HopsBoolean("denser","dense/sparse","switch if sparse or denser rulings",default=False),
        hs.HopsInteger("numDenser","numDenser","number of denser rulings",default=20),
        hs.HopsNumber("width","width","width of developable strip",default=0.5),
        hs.HopsNumber("distInterval","interval","interval distance of unrolling strips",default=1.5),
        ],
    outputs=[
            hs.HopsPoint("ctrlP","P","all ctrl points"),
            hs.HopsInteger("indices","ilist","list of control points P",access=hs.HopsParamAccess.LIST),
            hs.HopsCurve("Bezier","Bs","list of quintic Bezier splines",access=hs.HopsParamAccess.LIST),
            hs.HopsPoint("an","V","anchor points"),
            hs.HopsPoint("e1","e1","unit tangent vector"),
            hs.HopsPoint("e2","e2","unit principal normal vector"),
            hs.HopsPoint("e3","e3","unit binormal vector"),
            hs.HopsPoint("r","r","unit ruling vector"),
            hs.HopsMesh("Strip", "Strip", "3D mesh of developable strips"),
            hs.HopsMesh("unrollStrip", "unrollment", "2D mesh of unrolled developable strips"),
            ]
)
def bezier(mesh:rhino3dm.Mesh,is_singular,web,vn:rhino3dm.Vector3d,i,w1,is_ck,num_ck,is_optruling,w2,is_dense,num_dense,width,dist):
    x = [n.X for n in vn]
    y = [n.Y for n in vn]
    z = [n.Z for n in vn]
    VN = np.c_[x,y,z]
    setting = {
        web : True,
        'VN' : VN,
        'set_Poly' : i,
        'weight_CtrlP' : w1,
        'weight_SmoothVertex' : w2,
        'num_DenserRuling' : num_dense,
        'set_ScaleOffset' : width,
        'set_DistInterval' : dist,
        'is_Checker' : is_ck,
        'num_Checker' : num_ck,
        'is_DenserRuling' : is_dense,
        'is_RulingRectify' : is_optruling,
        'is_Singular' : is_singular,
    }
    M = _reading_mesh(mesh,**setting)
    ctrl_pts,nmlist,an,e1,e2,e3,r,sm,unm = M.set_quintic_bezier_splines()

    ilist = [(i-1)*5+1 for i in nmlist] # ctrl-points ## for regular patch


    ### GET QUINTIC BEZIER SPLINES:
    bz = []
    k=0
    for n in nmlist:
        m = np.arange((n-1)*5).reshape(-1,5) ##num of a row of ctrl-points
        mm = np.c_[m,(1+np.arange(n-1))*5] + k
        pt = []
        for i in range(mm.shape[0]):
            "for each Bezier CRV WITH 6 CTRL-POINTS" 
            pt = [rhino3dm.Point3d(ctrl_pts[j][0],ctrl_pts[j][1],ctrl_pts[j][2]) for j in mm[i]]
            c = rhino3dm.NurbsCurve.Create(False,5,pt)
            bz.append(c)
        k += (n-1)*5+1

    ### GET MULTI-ROW WHOLE NURBS-CURVE:
    # bz = []
    # k=0
    # for n in nmlist:
    #     m = (n-1)*5+1
    #     pt = []
    #     for i in range(m):
    #         pt.append(rhino3dm.Point3d(ctrl_pts[k+i][0],ctrl_pts[k+i][1],ctrl_pts[k+i][2]))
    #     c = rhino3dm.NurbsCurve.Create(False,5,pt)
    #     bz.append(c)
    #     k += m

    ### GET LIST:
    # ilist = []
    # k=0
    # for n in nmlist:
    #     ilist.append([k+i for i in range(n)])
    #     #ilist.append([rhino3dm.Point3d(k+i,0,0) for i in range(n)])
    #     k += n
    #ilist=[i for i in range(nmlist[0])] ### work to get list of numbers
    #print(ilist)

    P = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in ctrl_pts]
    AN = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in an]
    E1 = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in e1]
    E2 = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in e2]
    E3 = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in e3]
    R = [rhino3dm.Point3d(a[0],a[1],a[2]) for a in r]

    ### MAKE RHINO MESH:
    strip=rhino3dm.Mesh()
    for v in sm.vertices:
        strip.Vertices.Add(v[0],v[1],v[2])
    for f in sm.faces_list():
        strip.Faces.AddFace(f[0],f[1],f[2],f[3])

    unroll=rhino3dm.Mesh()
    for v in unm.vertices:
        unroll.Vertices.Add(v[0],v[1],v[2])
    for f in unm.faces_list():
        unroll.Faces.AddFace(f[0],f[1],f[2],f[3])

    return P,ilist,bz,AN,E1,E2,E3,R,strip,unroll
#==================================================================

if __name__ == "__main__":
    #app.run(host="10.8.0.10",debug=True)
    app.run(debug=True)

