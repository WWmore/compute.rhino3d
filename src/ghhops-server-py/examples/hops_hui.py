"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
import rhino3dm
import numpy as np
#-------------------------------------------------------
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
path = path+'\ArchitecturalGeometry'
sys.path.append(path)###==C:\Users\WANGH0M\Documents\GitHub\ArchitecturalGeometry   
#print(path)
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
def _reading_mesh(mesh,opt=False):
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
        M = Gridshell_AGNet()
    M.make_mesh(varr, farr) ###Refer gui_basic.py/open_obj_file()
    return M

### MAIN OPTIMIZATION:
@hops.component(
    "/read_1mesh",
    name="processMesh",
    nickname="pM",
    description="Process a mesh.",
    inputs=[
        hs.HopsMesh("mesh", "m", "The mesh to process."),
        hs.HopsInteger("iter","n","num of iteration."),
        hs.HopsNumber("fairness","w1","weight of fairness.",default=0.005),
        hs.HopsNumber("closness","w2","weight of self-closness.",default=0.01),
        hs.HopsNumber("direction","d","direction of polyline.",default=0),
        hs.HopsString("web","web","constraint of net or web"),
        hs.HopsBoolean("Restart","Restart","Restart the optimization",default=False),
    ],
    outputs=[hs.HopsMesh("mesh", "m", "The new mesh."),
            hs.HopsPoint("an","V","all vertices"),
            hs.HopsPoint("vn","N","normals at V")]
)
def read_1mesh(mesh:rhino3dm.Mesh,n,w1,w2,d,web,restart=False):
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
    "/checker_tian_vertex",
    name="checkerVertex",
    nickname="ckv",
    description="Get checker vertices.",
    inputs=[
        hs.HopsMesh("mesh", "m", "The optimized mesh.")],
    outputs=[hs.HopsPoint("P1","P1","checker vertices P1"),
            hs.HopsPoint("P2","P2","checker vertices P2"),
            ]
)
def main(mesh:rhino3dm.Mesh):
    M = _reading_mesh(mesh)
    Vr,Vb = M.plot_checker_group_tian_select_vertices()
    Pr = [rhino3dm.Point3d(r[0],r[1],r[2]) for r in Vr]
    Pb = [rhino3dm.Point3d(b[0],b[1],b[2]) for b in Vb]
    return Pr,Pb
#==================================================================

# @hops.component(
#     "/polyline",
#     name="checkerVertex",
#     nickname="ckv",
#     description="Get checker vertices.",
#     inputs=[
#         hs.HopsMesh("mesh", "m", "The optimized mesh."),
#         hs.HopsNumber("direction","d","direction of polyline.",default=0),
#         hs.HopsString("web","web","constraint of net or web"),
#     ],
#     outputs=[hs.HopsPoint("P1","P1","checker vertices P1"),
#             hs.HopsPoint("P2","P2","checker vertices P2"),
#             ]
# )
# def polyline(mesh:rhino3dm.Mesh,d,web):
#     ###-------------------------------------------
#     ### GET THE VERTEXLIST+FACELIST OF INPUT MESH:
#     vlist,flist = [],[]
#     for i in range(len(mesh.Vertices)):
#         vlist.append([mesh.Vertices[i].X,mesh.Vertices[i].Y,mesh.Vertices[i].Z])

#     for i in range(mesh.Faces.Count):
#         flist.append(list(mesh.Faces[i]))

#     varr = np.array(vlist)
#     farr = np.array(flist)

#     M = Gridshell_AGNet()
#     M.make_mesh(varr, farr) ###Refer gui_basic.py/open_obj_file()

#     constraints = {
#         'direction_poly' : d,
#         web : True,
#     }

#     AG = AGNet(**constraints)
#     AG.mesh = M
#     Pb,Py = AG.plot_checker_group_tian_select_vertices()
#     return Pb,Py





if __name__ == "__main__":
    #app.run(host="10.8.0.10",debug=True)
    app.run(debug=True)

