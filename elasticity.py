from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import time

# docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable         ## for linux and docker toolbox
# docker run -ti -v "%cd%":/home/fenics/shared quay.io/fenicsproject/stable         ## for windows / docker full install

# Scaled variables

Len = 1                                         # length of object (x=x[0])      (mm)
W = 1                                           # width of object (y=x[1])       (mm)
H = 1                                           # height of object (z=x[2])      (mm)
num = 10                                        # number of elements in each direction
num_nodes = (num+1)**3                          # number of nodes in the whole system
E = 210000                                     # Young's Modulus                (N/mm²)
nu = 0.3                                        # Poisson's ratio                (no dimension)
lmbda = Constant((E*nu)/((1+nu)*(1-2*nu)))    # 1. Lame's parameter            (N/mm²)
mu = Constant(E/(2*(1+nu)))                     # 2. Lame's parameter            (N/mm)
#rho = 7850                                     # density                        (kg/m³)
#g = 9.81                                       # gravitational acceleration     (m/s²)
timesteps = 10                                  # timesteps
modes = 2                                       # number of modes used
tol = 1E-14                                     # tolerance


# Some useful Methods

# Method to convert vector from dof map order x[0] -> x[1] -> x[2] to logical order x[0] x[1] x[2] -> x[0] x[1] x[2]

def conversion(displacement_vector):
    displacement_vector_ph = []
    size = int((displacement_vector.shape[0])/3)
    try:
        displacement_vector_ph2 = []
        for col in range(0, displacement_vector.shape[1]):
            for row in range(0, size):
                for dim in range(0, 3):
                    displacement_vector_ph2.append(displacement_vector[row + dim * size][col])
            displacement_vector_ph.append(displacement_vector_ph2)
            displacement_vector_ph2 = []
    except:
        for row in range(0, size):
                for dim in range(0, 3):
                    displacement_vector_ph.append(displacement_vector[row + dim * size])
    return np.array(displacement_vector_ph).T


# Method for Writing modes to displacement doc

def docit(**kwargs):
    file = open("/home/fenics/shared/elasticity/displacementmodes000000.vtu", "r")
    lines = file.readlines()
    file.close()
    file = open("/home/fenics/shared/elasticity/displacementmodes000000.vtu", "w")
    for line in lines:
        file.write(line)
        if line[0:19] == "<PointData  Vectors":
            string = {}
            for i in range(1, modes+1):
                string[i] = ""
                for j in conversion(phi[:, i-1]):
                    string[i] = string[i] + " " + str(j)
                file.write("<DataArray  type=\"Float64\"  Name=\"mode"+str(i)+"\"  NumberOfComponents=\"3\" format=\"ascii\">" + string[i] + "</DataArray>\n")
            for key in kwargs:
                value = np.array(list(kwargs.values()))[0, :]
                string["final"] = ""
                for j in value:
                    string["final"] = string["final"] + " " + str(j)
                file.write("<DataArray  type=\"Float64\"  Name=\"" + key + "\"  NumberOfComponents=\"3\" format=\"ascii\">" + string["final"] + "</DataArray>\n")

    file.close()


# Method to save a matrix to .txt

def snaptxt(name, matrix_u, steps = (0, 0)):
    matrix = []
    matrix_u = np.array(matrix_u)
    if(matrix_u.shape[0] < matrix_u.shape[1]):
        matrix_u = matrix_u.T
    if steps == (0, 0):
        steps = (0, matrix_u.shape[1])
    for step in range(steps[0], steps[1]):
        matrix.append(matrix_u[:, step])
    matrix = np.array(matrix).T
    np.savetxt("%s.txt" % str(name), conversion(matrix))


# Create mesh and define function space
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, "eliminate_zeros": False, "precompute_basis_const": True, "precompute_ip_const": True}
parameters["reorder_dofs_serial"] = False
mesh = BoxMesh(Point(0, 0, 0), Point(Len, W, H), num, num, num)
V = VectorFunctionSpace(mesh, 'Lagrange', 1)


# Define boundary conditions

# Object can't move at outer Boundary in direction of x

def cb1(x, on_boundary):
    return on_boundary and near(x[0], Len, tol)


bc1 = DirichletBC(V.sub(0), Constant(0), cb1)


# Object can't move at outer Boundary in direction of y

def cb2(x, on_boundary):
    return on_boundary and near(x[1], W, tol)


bc2 = DirichletBC(V.sub(1), Constant(0), cb2)


# Object can't move at bottom Boundary in direction of z


def cb3(x, on_boundary):
    return on_boundary and near(x[2], 0, tol)


bc3 = DirichletBC(V.sub(2), Constant(0), cb3)


# Object can't move at upper Boundary in direction of x and y

def cb4(x, on_boundary):
    return on_boundary and near(x[2], H, tol)


bc4 = DirichletBC(V.sub(0), Constant(0), cb4)
bc5 = DirichletBC(V.sub(1), Constant(0), cb4)


bcs = [bc1, bc2, bc3, bc4, bc5]                 # combining boundary conditions


# creating and applying Neumann Condition

class ForceArea(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > Len/2 and x[1] > W/2 and near(x[2], H, tol)


area = ForceArea()                              # initializing area
Neumann = MeshFunction("size_t", mesh, 2)       # creating function fenics can work with
area.mark(Neumann, 1)                           # marking subdomain to the function
ds = Measure("ds", subdomain_data=Neumann)      # measuring ds with reference to subdomain function
dx = Measure("dx")                              # measuring dx with reference to the whole domain


# Define strain and stress

def epsilon(z):
    return 0.5 * (nabla_grad(z) + nabla_grad(z).T)


def sigma(u):
    return lmbda * tr(epsilon(u)) * I + 2 * mu * epsilon(u)


# Define variational problem


du = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
f = Constant((0, 0, -25000))                    # defining force
d = u.geometric_dimension()                     # space dimension
I = Identity(d)
F = I + grad(u)
C = F.T*F
Ic = tr(C)
J = det(F)
psi = (mu/2)*(Ic -3) - mu*ln(J) + (lmbda/2)*ln(J)**2

u_array_dof = []

# Compute solutions

def compute_unreduced():
    for i in range(1, timesteps+1):
        Pi = psi*dx - dot(f*i, u)*ds(1)
        F = derivative(Pi, u, v)
        J = derivative(F, u, du)
#        starttime = time.time()
#        problem = NonlinearVariationalProblem (F, u, bcs=bcs)
#        solver = NonlinearVariationalSolver(problem)
#        prm = solver.parameters
#        prm["newton_solver"]["linear_solver"] = "gmres"
#        prm['newton_solver']['absolute_tolerance'] = 1E2
#        prm['newton_solver']['relative_tolerance'] = 0.99
#        solver.solve
        solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)
#        print("Full: ", time.time()-starttime)


# Compute stress

    #s = sigma(u) - (1. / 3) * tr(sigma(u)) * Identity(d)  # deviatoric stress
    #von_Mises = sqrt(3. / 2 * inner(s, s))
    #V = FunctionSpace(mesh, 'P', 1)
    #von_Mises = project(von_Mises, V)
    #plot(von_Mises, title='Stress intensity')


# Compute displacement

    #    V = FunctionSpace(mesh, 'P', 1)
    #    u_displacement = np.array([u(Point(x, y, z)) for x, y, z in mesh_points])
    #    u_displacement_ro = []
    #    for j in range(0, num_nodes):
    #        for k in range(0, 3):
    #            u_displacement_ro.append(u_displacement[j][k])
    #    u_array.append(u_displacement_ro)
        File('elasticity/displacement' + str(i) + '.pvd') << u
        File('elasticity/displacementmodes.pvd') << u
        u_dis = np.array(u.vector())
        u_array_dof.append(u_dis)
# Compute displacement magnitude

    #    u_magnitude = sqrt(dot(u, u))
    #    u_magnitude = project(u_magnitude, V)
    #    plot(u_magnitude, title='Displacement magnitude')


# Integrating timesteps in original file

    file = open("/home/fenics/shared/elasticity/displacement1.pvd", "r")
    lines = file.readlines()
    file.close()
    file = open("/home/fenics/shared/elasticity/displacement1.pvd", "w")
    for line in lines:
        if line[0:21] == "    <DataSet timestep":
            for step in range(1, timesteps+1):
                file.write("    <DataSet timestep=\"" + str(step) + "\" part=\"0\" file=\"displacement" + str(step) + "000000.vtu\" />\n")
        else:
            file.write(line)
    file.close()


# Singular Value Decomposition

    u_svd, s_svd, v_svd = np.linalg.svd(np.array(u_array_dof).T)

    phi = []

    for mode in range(modes):
        phi.append(u_svd[:, mode])
    phi = np.array(phi).T                                   # matrix of modes
    return phi

phi = compute_unreduced()


# POD stuff
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), epsilon(v)) * dx                    # defining a, L used to solve the PDE
L = dot(f*timesteps, v)*ds(1)
u = Function(V)
K, f = assemble_system(a, L, bcs)                       # assembling stiffness matrix and force vector
K = np.array(K.array())                                 # stiffness matrix
f = np.array(f)                                         # force vector

K_red = np.matmul(np.matmul(phi.T, K), phi)             # reduced stiffness matrix
f_red = np.matmul(phi.T, f)                             # reduced force vector
u_red = np.matmul(f_red, np.linalg.inv(K_red))          # reduced displacement
final = np.matmul(phi, u_red)                           # projected displacement
endtime = time.time()

snaptxt("phi", phi)
snaptxt("snapshots", u_array_dof)
docit(final=conversion(final))
