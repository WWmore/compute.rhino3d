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