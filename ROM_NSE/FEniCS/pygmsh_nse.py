import gmsh
import meshio

length = 2.2  # Length of the rectangle
width = 0.41    # Width of the rectangle
radius = 0.05   # Radius of the circular hole

dx_min = 0.005
dx_max = 0.02

# Initialize the Gmsh Python API
gmsh.initialize()

# Add a new model
gmsh.model.add('my_model')

# Define the rectangle
rect = gmsh.model.occ.addRectangle(0, 0, 0, length, width)

# Define the circle
circle = gmsh.model.occ.addDisk(0.2, 0.2, 0, radius, radius)

# # Define a box around the circle where you want to refine the mesh
# box = gmsh.model.occ.addBox(2, 2, 0, 1, 1, 1)

# Subtract the circle from the rectangle
diff = gmsh.model.occ.cut([(2, rect)], [(2, circle)])

# Synchronize the model (this creates the actual geometry)
gmsh.model.occ.synchronize()



# Define a new mesh size field
field = gmsh.model.mesh.field.add("Box")

# Set the box for the field
gmsh.model.mesh.field.setNumber(field, "VIn", dx_min)
gmsh.model.mesh.field.setNumber(field, "VOut", dx_max)

gmsh.model.mesh.field.setNumber(field, "XMin", 0.2-2*radius)
gmsh.model.mesh.field.setNumber(field, "XMax", 0.2+2*radius)
gmsh.model.mesh.field.setNumber(field, "YMin", 0.2-2*radius)
gmsh.model.mesh.field.setNumber(field, "YMax", 0.2+2*radius)
gmsh.model.mesh.field.setNumber(field, "ZMin", 0)
gmsh.model.mesh.field.setNumber(field, "ZMax", 0)

# Set the field as the background field
gmsh.model.mesh.field.setAsBackgroundMesh(field)


# # Define a new mesh size field
# field = gmsh.model.mesh.field.add("MathEval")

# # Set the formula for the field (e.g., "0.1" for a mesh size of 0.1)
# gmsh.model.mesh.field.setString(field, "F", "0.1")

# # Set the field as the background field
# gmsh.model.mesh.field.setAsBackgroundMesh(field)

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Save the mesh
gmsh.write("mesh.vtk")

# Finalize the Gmsh Python API
gmsh.finalize()

# Read the mesh from the VTK file
mesh = meshio.read("mesh.vtk")

# remove z-coordinate 
mesh = meshio.Mesh(mesh.points[:, :2], {"triangle":  mesh.get_cells_type("triangle")})

# Write the mesh to an XML file
meshio.write("mesh.xml", mesh)


