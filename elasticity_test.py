from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import time
import scipy as sp
import scipy.linalg as la

# docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable         ## for linux and docker toolbox
# docker run -ti -v "%cd%":/home/fenics/shared quay.io/fenicsproject/stable         ## for windows / docker full install


# Some useful Methods

    # Method to convert vector from dof map order x[0] -> x[1] -> x[2] to logical order x[0] x[1] x[2] -> x[0] x[1] x[2]

def conversion(displacement_vector):
    displacement_vector_ph = []
    displacement_vector = np.array(displacement_vector)
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


# Method for Writing modes or other displacements to displacement file

def docit(**kwargs):
    file = open("/home/fenics/shared/elasticity/displacementmodesstandard000000.vtu", "r")
    lines = file.readlines()
    file.close()
    file = open("/home/fenics/shared/elasticity/displacementmodes000000.vtu", "w")
    try:
        phi[0]
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
    except:
        for line in lines:
            file.write(line)
            if line[0:19] == "<PointData  Vectors":
                string = {}
                for key in kwargs:
                    value = np.array(list(kwargs.values()))[0, :]
                    string["final"] = ""
                    for j in value:
                        string["final"] = string["final"] + " " + str(j)
                    file.write("<DataArray  type=\"Float64\"  Name=\"" + key + "\"  NumberOfComponents=\"3\" format=\"ascii\">" + string["final"] + "</DataArray>\n")
    file.close()


# Method to save a matrix to .txt

def snaptxt(name, matrix_u, conv = False, steps = (0, 0)):
    matrix = []
    matrix_u = np.array(matrix_u)
    if(matrix_u.shape[0] < matrix_u.shape[1]):
        matrix_u = matrix_u.T
    if steps == (0, 0):
        steps = (0, matrix_u.shape[1])
    for step in range(steps[0], steps[1]):
        matrix.append(matrix_u[:, step])
    matrix = np.array(matrix).T
    if not conv:
        np.savetxt("%s.txt" % str(name), matrix)
    elif conv:
        np.savetxt("%s.txt" % str(name), conversion(matrix))


# Scaled variables

error_matrix = []
for trials in range(1, 2):
    print(trials, " ______________________________________________________________________________________________________________")
#    Len = 0.90+np.random.random()*0.2                                        # length of object (x=x[0])      (mm)
    Len = 1
    W = 1                                           # width of object (y=x[1])       (mm)
    H = 1                                           # height of object (z=x[2])      (mm)
    num = 6
#    num = trials*2                                        # number of elements in each direction
    num_nodes = (num+1)**3                          # number of nodes in the whole system
    E = 210000                                          # Young's Modulus                (N/mm²)
    nu = 0.3                                     # Poisson's ratio                (no dimension)
    lmbda = Constant((E*nu)/((1+nu)*(1-2*nu)))      # 1. Lame's parameter            (N/mm²)
    mu = Constant(E/(2*(1+nu)))                     # 2. Lame's parameter            (N/mm)
    force = -25000
    #lmbda = 400000
    #mu = 80
    #rho = 7850                                     # density                        (kg/m³)
    #g = 9.81                                       # gravitational acceleration     (m/s²)
    timesteps = 3                                  # timesteps
    modes = 3                                       # number of modes used
    tol = 1E-14                                     # tolerance

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
    f = Constant((0, 0, force))                     # defining force
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
            starttime = time.time()
            Pi = psi*dx - dot(f*i, u)*ds(1)
            F = derivative(Pi, u, v)
            J = derivative(F, u, du)

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
            solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)
            print("step %s:" % i, time.time()-starttime)
            File('elasticity/displacement' + str(i) + '.pvd') << u
            File('elasticity/displacementmodesstandard.pvd') << u
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

        return u_array_dof
    

# Singular Value Decomposition

    def sing_val_dec(u_unsorted):
            starttime = time.time()
            u_svd, s_svd, v_svd = la.svd(np.array(u_unsorted).T)
            phi = []
            for mode in range(modes):
                phi.append(u_svd[:, mode])
            phi = np.array(phi).T                                           # matrix of modes
            print("SVD:", time.time()-starttime)
            snaptxt("phi_unsorted", phi)
            return phi


# Load mode matrix from .txt file

    def load_phi_from_txt():
        phi_loaded = []
        line_ph = []
        file = open("/home/fenics/shared/phi_unsorted.txt", "r")
        lines =  file.readlines()
        file.close()
        for line in lines:
            line_ph.append(line.split())
        line_ph = np.array(line_ph)
        for mod in range(0, modes):
            phi_loaded_ph = []
            for i in range(0, len(line_ph)):
                phi_loaded_ph.append(float(line_ph[i][mod]))
            phi_loaded.append(phi_loaded_ph)
        phi_loaded = np.array(phi_loaded).T
        return phi_loaded


# Proper Orthogonal Decomposition

    def POD(phi):
        for i in range(1, timesteps+1):
            du = TrialFunction(V)
            u = Function(V)
            v = TestFunction(V)
            f = Constant((0, 0, force))                     # defining force
            d = u.geometric_dimension()                     # space dimension
            I = Identity(d)
            
            a = inner(sigma(du), epsilon(v)) * dx
            L = dot(f*i, v)*ds(1)
            K, f_vec = assemble_system(a, L, bcs)
            #K = np.array(K.array())                                                 # stiffness matrix
            K_red = np.matmul(np.matmul(phi.T, K), phi)                             # reduced stiffness matrix
            #f_vec = np.array(f_vec)  
            solve(a == L, u, bcs)
            ep = np.array([1])
            j = 1
            # f_red = np.matmul(phi.T, f_vec)                                             # reduced force vector
            # u_red = np.matmul(f_red, np.linalg.inv(K_red))                          # reduced displacement
            # u_POD = np.matmul(phi, u_red)                                           # projected displacement
            # G_red = np.matmul(phi.T, np.array(ep))
            while np.amax(ep) > 10e-7:
                print("Newton iteration:", j)
                F = I + grad(u)
                C = F.T*F
                Ic = tr(C)
                J = det(F)
                psi = (mu/2)*(Ic -3) - mu*ln(J) + (lmbda/2)*ln(J)**2
                Pi = psi*dx - dot(f*i, u)*ds(1)
                F = derivative(Pi, u, v)
                J = derivative(F, u, du)
                A, ep = assemble_system(J, F, bcs=bcs)
                delta_u = np.matmul(-np.array(ep), np.linalg.inv(K.array()))
                u = np.array(u.vector()) + delta_u
                u_ph = Function(V)
                u_ph.vector().set_local(u)
                u = u_ph

                # f_red = np.matmul(phi.T, f_vec)                                             # reduced force vector
                # u_red = np.matmul(f_red, np.linalg.inv(K_red))                          # reduced displacement
                # u_POD = np.matmul(phi, u_red)                                           # projected displacement
                ep = np.array(ep)
                print(np.mean(ep))
                if j == 40:
                    break
                j += 1
            u_POD = np.array(u.vector())
            np.savetxt("ep", ep)
            endtime = time.time()
            print("POD step %s:" % i)
            u_max_unr = np.amax(u_array_dof[:][i-1])
            u_max_red = np.amax(u_POD)
            error = abs((u_max_unr-u_max_red)/u_max_unr)
            print("Lenght:", Len, " ; u_max_unr:", u_max_unr, " ; u_max_red:", u_max_red, " ; Difference in Deformation:", error)
        # error_matrix.append((Len, error))
        return u_POD

    u_array_dof = compute_unreduced()

    phi = sing_val_dec(u_array_dof)

    phi_loaded = load_phi_from_txt()

    u_POD = POD(phi_loaded)

#np.savetxt("error_matrix.txt", error_matrix)
#snaptxt("snapshots", u_array_dof, conv=True)
#snaptxt("phi_loaded_unsorted", phi_loaded)
docit(u_POD=conversion(u_POD))