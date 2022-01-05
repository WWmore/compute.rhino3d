"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
import rhino3dm
import numpy

# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


@hops.component(
    "/processMesh",
    name="processMesh",
    nickname="pM",
    description="Process a mesh.",
    inputs=[
        hs.HopsMesh("mesh", "m", "The mesh to process."),
        hs.HopsNumber("a","a","parameter a"),
    ],
    outputs=[hs.HopsMesh("mesh", "m", "The new mesh."),
             hs.HopsNumber("a","a","parameter a")]
)
def processMesh(mesh: rhino3dm.Mesh, mynur):
    assert isinstance(mesh, rhino3dm.Mesh)
    
    m=rhino3dm.Mesh()
    print(type(m.Vertices))
    m.Vertices.Add(mynur,mynur,mynur)
    m.Vertices.Add(1,1,2)
    m.Vertices.Add(3,3,3)
    print(type(m.Faces))
    m.Faces.AddFace(0,1,2)
    
    import os
    import sys
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(path)
    print(path)

    #from geometrylab.optimization.gridshell import Gridshell

    b=3*mynur
    return m, b



@hops.component(
    "/processVF",
    name="processVF",
    nickname="vf",
    description="Process a mesh by vertexlist and facelist.",
    inputs=[
        hs.HopsNumber("x","vx","x-coordinate of vertexlist"),
        hs.HopsNumber("y","vy","y-coordinate of vertexlist"),
        hs.HopsNumber("z","vz","z-coordinate of vertexlist"),
        hs.HopsNumber("f1","f1","1st facelist"),
        hs.HopsNumber("f2","f2","2nd facelist"),
        hs.HopsNumber("f3","f3","3rd facelist"),
        hs.HopsNumber("f4","f4","4th facelist"),
    ],
    outputs=[#hs.HopsMesh("mesh", "m", "The new mesh."),
             hs.HopsNumber("a","a","parameter a")]
)
def processVFlist(vx,vy,vz,f1,f2,f3,f4):
    #assert isinstance(mesh, rhino3dm.Mesh)
    
    vertices_list = np.c_[vx,vy,vz]
    faces_list = np.c_[f1,f2,f3,f4]

    import os
    import sys
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    path = path+'\ArchitecturalGeometry'
    sys.path.append(path) ###==C:\Users\WANGH0M\Documents\GitHub\ArchitecturalGeometry   
    from geometrylab.optimization.gridshell import Gridshell
    
    print(vertices_list)
    print(faces_list)
    Gridshell.make_mesh(vertices_list, faces_list)

    b=3*vx
    return b




if __name__ == "__main__":
    #app.run(host="10.8.0.10",debug=True)
    app.run(debug=True)

