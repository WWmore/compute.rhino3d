
## flask app can be used for other stuff drectly
#@app.route("/help")
#def help():
#    return "Welcome to Grashopper Hops for CPython!"


#@hops.component(
#    "/binmult",
#    inputs=[hs.HopsNumber("A"), hs.HopsNumber("B")],
#    outputs=[hs.HopsNumber("Multiply")],
#)
#def BinaryMultiply(a: float, b: float):
#    return a * b


#@hops.component(
#    "/add",
#    name="Add",
#    nickname="Add",
#    description="Add numbers with CPython",
#    inputs=[
#        hs.HopsNumber("A", "A", "First number"),
#        hs.HopsNumber("B", "B", "Second number"),
#    ],
#    outputs=[hs.HopsNumber("Sum", "S", "A + B")]
#)
#def add(a: float, b: float):
#    return a + b


#@hops.component(
#    "/pointat",
#    name="PointAt",
#    nickname="PtAt",
#    description="Get point along curve",
#    icon="pointat.png",
#    inputs=[
#        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        
#        hs.HopsNumber("t", "t", "Parameter on Curve to evaluate")
#    ],
#    outputs=[hs.HopsPoint("P", "P", "Point on curve at t")]
#)
#def pointat(c: rhino3dm.Curve, t=0.0):  #Rhino.Geometry.Curve
        
#    return c.PointAt(t)

#def test(t:str):
    
#    pass

#@hops.component(
#    "/srf4pt",
#    name="4Point Surface",
#    nickname="Srf4Pt",
#    description="Create ruled surface from four points",
#    inputs=[
#        hs.HopsPoint("Corner A", "A", "First corner"),
#        hs.HopsPoint("Corner B", "B", "Second corner"),
#        hs.HopsPoint("Corner C", "C", "Third corner"),
#        hs.HopsPoint("Corner D", "D", "Fourth corner")
#    ],
#    outputs=[hs.HopsSurface("Surface", "S", "Resulting surface")]
#)
#def ruled_surface(a: rhino3dm.Point3d,
#                  b: rhino3dm.Point3d,
#                  c: rhino3dm.Point3d,
#                  d: rhino3dm.Point3d):
#    edge1 = rhino3dm.LineCurve(a, b)
#    edge2 = rhino3dm.LineCurve(c, d)
#    return rhino3dm.NurbsSurface.CreateRuledSurface(edge1, edge2)



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
    #assert isinstance(m,rhino3dm.MeshVertexList)
    print(type(m.Vertices))
    m.Vertices.Add(mynur,mynur,mynur)
    m.Vertices.Add(1,1,2)
    m.Vertices.Add(3,3,3)
    print(type(m.Faces))
    m.Faces.AddFace(0,1,2)
    

    b=3*mynur
    return m, b

if __name__ == "__main__":
    app.run(host="10.8.0.10",debug=True)

