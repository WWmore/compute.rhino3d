"""Hops default HTTP server example"""
import ghhops_server as hs
import rhino3dm

hops = hs.Hops()


@hops.component(
    "/add",
    name="Add",
    nickname="Add",
    description="Add numbers with CPython",
    inputs=[
        hs.HopsNumber("A", "A", "First number"),
        hs.HopsNumber("B", "B", "Second number"),
    ],
    outputs=[hs.HopsNumber("Sum", "S", "A + B")]
)
def add(a, b):
    return a + b


@hops.component(
    "/pointat",
    name="PointAt",
    nickname="PtAt",
    description="Get point along curve",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameter on Curve to evaluate"),
    ],
    outputs=[
        hs.HopsPoint("P", "P", "Point on curve at t")
    ]
)
def pointat(curve, t=0.0):
    return curve.PointAt(t)


# same as pointat, but with a list of numbers as input
# and list of points returned
@hops.component(
    "/pointsat",
    name="PointsAt",
    nickname="PtsAt",
    description="Get points along curve",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameters on Curve to evaluate", hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsPoint("P", "P", "Points on curve at t")
    ]
)
def pointsat(curve, t):
    points = [curve.PointAt(item) for item in t]
    return points


@hops.component(
    "/srf4pt",
    name="4Point Surface",
    nickname="Srf4Pt",
    description="Create ruled surface from four points",
    inputs=[
        hs.HopsPoint("Corner A", "A", "First corner"),
        hs.HopsPoint("Corner B", "B", "Second corner"),
        hs.HopsPoint("Corner C", "C", "Third corner"),
        hs.HopsPoint("Corner D", "D", "Fourth corner")
    ],
    outputs=[hs.HopsSurface("Surface", "S", "Resulting surface")]
)
def ruled_surface(a, b, c, d):
    edge1 = rhino3dm.LineCurve(a, b)
    edge2 = rhino3dm.LineCurve(c, d)
    return rhino3dm.NurbsSurface.CreateRuledSurface(edge1, edge2)



@hops.component(
    "/ttt",
    name="Mesh",
    nickname="mesh",
    description="Transfer into a mesh from input mesh",
    category="Mesh",
    subcategory="RectangularPatch",
    inputs=[
        hs.HopsNumber("ver_x","x"),
        hs.HopsNumber("ver_y","y"),
        hs.HopsNumber("ver_z","z"),
        hs.HopsNumber("if1", "fid1"),
        hs.HopsNumber("if2", "fid2"),
        hs.HopsNumber("if3", "fid3"),
        hs.HopsNumber("if4", "fid4"),
    ],
    #outputs=[hs.HopsPoint("P", "points", "points of polyline")],
    outputs=[hs.HopsNumber("x", "points", "points of polyline"),
            hs.HopsNumber("y", "points", "points of polyline"),
            hs.HopsNumber("z", "points", "points of polyline")],
    #outputs=[hs.HopsMesh("mesh")],
)
def get_mesh_polyline_points(verx,very,verz,if1,if2,if3,if4):
    #vertices = np.arange(12).reshape(-1,3)
    # vertices = np.c_[verx,very,verz]
    # mesh = Mesh()
    # flist = np.c_[if1,if2,if3,if4]
    # mesh.make_mesh(vertices,flist.tolist())
    # return mesh.vertices[:,0],mesh.vertices[:,1],mesh.vertices[:,2]
    return 1,2,3#vertices[:,0],vertices[:,1],vertices[:,2]

if __name__ == "__main__":
    hops.start(debug=True)
