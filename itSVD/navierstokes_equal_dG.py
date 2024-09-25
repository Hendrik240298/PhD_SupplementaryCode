import dolfin
import numpy as np
import matplotlib.pyplot as plt
from sympy import Mul, Id, symbols, init_printing, expand, compose, diff, lambdify, Piecewise, N, Rational
from sympy import sqrt as Sqrt
from fenics import Mesh, VectorElement, Function, TrialFunction, TestFunction, TestFunctions, FunctionSpace, dx, inner, grad, FiniteElement, MixedElement, Constant, assemble, Expression, interpolate, solve, DirichletBC, plot, errornorm, set_log_active, derivative, parameters, split, dot, div, CompiledSubDomain, MeshFunction, sqrt, Measure, FacetNormal, Identity, XDMFFile
from ufl import replace
import time
import argparse
import logging
import os

set_log_active(False) # turn off FEniCS logging
parameters["reorder_dofs_serial"] = False

t = symbols("t")
init_printing()

# parse command line arguments
parser = argparse.ArgumentParser(description="Run a dG(r,r) simulation of Navier-Stokes.")
parser.add_argument("--r", type=int, default=1, help="temporal polynomial degree for velocity and pressure")
parser.add_argument("--dt", type=float, default=0.03125, help="time step size")
parser.add_argument("--problem", type=str, default="2D-2", help="problem to solve")
parser.add_argument("--time_dg_quadrature", type=str, default="Gauss-Lobatto", help="quadrature rule for temporal integrals")
parser.add_argument("--mesh", type=str, default="schaefer_turek_2D_pygmsh.xml", help="mesh file for Schaefer-Turek 2D benchmarks")


# parse args
args = parser.parse_args()
r = args.r
slab_size = args.dt
PROBLEM = args.problem
assert PROBLEM in ["2D-2", "2D-3"], f"Problem {PROBLEM} has not been implemented yet."
TIME_DG_QUADRATURE = args.time_dg_quadrature
assert TIME_DG_QUADRATURE in ["Gauss-Lobatto", "Gauss-Legendre"], f"Time quadrature rule {TIME_DG_QUADRATURE} has not been implemented yet."
MESH_NAME = args.mesh
assert os.path.exists(MESH_NAME), f"Mesh file {MESH_NAME} does not exist."

# create output directory, if it does not exist yet
if not os.path.exists(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}"):
    os.makedirs(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}")

# configure logging
logging.basicConfig(filename=f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/output.log", level=logging.DEBUG)

# HELPER FUNCTIONS
# compute temporal basis functions from the roots
def compute_basis_functions(roots):
    basis = []
    for i in range(len(roots)):
        f = 1 + 0*Id(t)
        for j in range(len(roots)):
            if i != j:
                f = Mul(f,(t - roots[j]) / (roots[i] - roots[j]))
        #basis.append(f)
        basis.append(expand(f))
    return basis

# transform roots from [-1, 1] to [0, 1]
def transform_roots(roots):
    new_roots = []
    for root in roots:
        new_roots.append((root + 1) / 2)
    return new_roots

# dictionary of temporal bases depending on the polynomial degree
# FE basis on reference element (0,1)
φ = {}
roots = {}

if TIME_DG_QUADRATURE == "Gauss-Lobatto":
    roots[0] = [1.]
    φ[0] = [1 + 0*Id(t)]
    roots[1] = [0., 1.]
    φ[1] = [1-t, t]
    roots[2] = transform_roots([-1, 0, 1])
    φ[2] = compute_basis_functions(roots[2])
    #roots[3] = transform_roots([-1, -sqrt(Rational(1,5)), sqrt(Rational(1,5)), 1])
    #φ[3] = compute_basis_functions(roots[3])
elif TIME_DG_QUADRATURE == "Gauss-Legendre":
    roots[0] = [1.]
    φ[0] = [1 + 0*Id(t)]
    roots[1] = transform_roots([-N(Sqrt(Rational(1,3))), N(Sqrt(Rational(1,3)))])
    φ[1] = compute_basis_functions(roots[1])
    roots[2] = transform_roots([-N(Sqrt(Rational(3,5))), 0, N(Sqrt(Rational(3,5)))])
    φ[2] = compute_basis_functions(roots[2])
    # roots_Legendre[3] = transform_roots([
    #     -N(Sqrt(Rational(3,7) + Rational(2,7)*Sqrt(Rational(6,5)))), 
    #     -N(Sqrt(Rational(3,7) - Rational(2,7)*Sqrt(Rational(6,5)))),
    #     N(Sqrt(Rational(3,7) - Rational(2,7)*Sqrt(Rational(6,5)))),
    #     N(Sqrt(Rational(3,7) + Rational(2,7)*Sqrt(Rational(6,5)))),
    # ])
    # φ_Legendre[3] = compute_basis_functions(roots_Legendre[3])
else:
    raise NotImplementedError(f"Time quadrature rule {TIME_DG_QUADRATURE} has not been implemented yet.")

class TimeFE:
    def __init__(self, r=1, a=0., b=1., n_time=1, n_q_points=3):
        self.r = r # polynomial degree in time
        self.n_time = n_time # number of temporal elements
        self.n_dofs = (self.r+1) * self.n_time # total number of temporal dofs on slab
        self.dof_locations = []
        self.a = a # start time of slab
        self.b = b # end time of slab
        self.k = (self.b - self.a) / self.n_time
        self.epsilon = self.k * 1e-10

        self.generate_mesh()
        self.get_full_basis()
        self.lambdify_basis()
        self.compute_quadrature(n_q_points)

    def generate_mesh(self):
        # create an uniform temporal mesh with mesh size self.k
        self.mesh = [(self.a, self.a+self.k)]
        while len(self.mesh) < self.n_time:
            self.mesh.append((self.mesh[-1][1], self.mesh[-1][1]+self.k))

    # transform basis functions from [0,1] to [a,b]
    def transform_function(self, f, a, b):
        return compose(f, (t-a)/(b-a)) # = f((t-a)/(b-a))
    
    def transform_derivative(self, a, b):
        return 1 / (b-a)

    # get full FE basis and its derivative on temporal mesh
    def get_full_basis(self):
        self._basis = []
        self._basis_derivative = []
        self.local_dofs = {}
        i = 0
        for (a,b) in self.mesh:
            self.local_dofs[(a,b)] = []
            for f, t_q in zip(φ[self.r], roots[self.r]):
                self._basis.append(self.transform_function(f, a, b))
                self._basis_derivative.append(diff(self._basis[-1],t))
                #print(diff(self._basis[-1],t))
                #print(self.transform_function(diff(f,t), a, b) * self.transform_derivative(a, b))
                self.local_dofs[(a,b)].append(i)
                self.dof_locations.append(t_q*(b-a)+a)
                i += 1

    # convert SymPy functions to Python functions and ensure that they are 0 outside the element that they are defined on
    def lambdify_basis(self):
        self.phi = []
        self.dt_phi = []

        for (a,b) in self.mesh:
            for i in self.local_dofs[(a,b)]:
                self.phi.append(
                    lambdify(
                        t,
                        Piecewise(
                            (0, t < a),
                            (0, t > b),
                            (self._basis[i], True)
                        )
                    )
                )

                self.dt_phi.append(
                    lambdify(
                        t,
                        Piecewise(
                            (0, t < a),
                            (0, t > b),
                            (self._basis_derivative[i], True)
                        )
                    )
                )

    def compute_quadrature(self, n_q_points):
        # Gauss-Legendre quadrature points and weights on [-1,1]
        quad_points, quad_weights = np.polynomial.legendre.leggauss(n_q_points)

        # transform quadrature points and weights from [-1,1] to [a,b] for each temporal element [a,b]
        self.quadrature = {}
        for (a, b) in self.mesh:
            t_q = 0.5 * (b-a) * quad_points + 0.5 * (a+b)
            w_q = 0.5 * (b-a) * quad_weights
            self.quadrature[(a,b)] = [(t_q[i], w_q[i]) for i in range(t_q.shape[0])]
            
        # Gauss-Legendre quadrature points and weights on [-1,1]
        quad_points, quad_weights = np.polynomial.legendre.leggauss(n_q_points+2)

        # transform quadrature points and weights from [-1,1] to [a,b] for each temporal element [a,b]
        self.quadrature_fine = {}
        for (a, b) in self.mesh:
            t_q = 0.5 * (b-a) * quad_points + 0.5 * (a+b)
            w_q = 0.5 * (b-a) * quad_weights
            self.quadrature_fine[(a,b)] = [(t_q[i], w_q[i]) for i in range(t_q.shape[0])]

    def plot_basis(self, basis_type="function", title=None):
        assert basis_type in ["function", "derivative"], f"basis_type='{basis_type}' has not been implemented."

        _t = np.linspace(self.a, self.b, 100)
        for i in range(len(self.phi)):
            if basis_type == "function":
                plt.plot(_t, [self.phi[i](time_point) for time_point in _t], label=rf"$\varphi_{{{i}}}$")
            elif basis_type == "derivative":
                plt.plot(_t, [self.dt_phi[i](time_point) for time_point in _t], label=rf"$\partial_t \varphi_{{{i}}}$")

        plt.xlabel("t")
        plt.ylabel("f(t)")
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()
        
    def get_solution_at_time(self, time_point, U):
        tmp = np.zeros_like(U[0])
        for i in range(self.n_dofs):
            tmp += self.phi[i](time_point) * U[i]
        return tmp

##############################################
# Start a time marching / time slabbing loop #
##############################################
start_time = 0.
end_time = 0.25

s_v = 2
s_p = 1
nu = 0.001

# start simulation
cpu_start_time = time.time()
logging.info(f"CONFIG: problem = {PROBLEM}, s = ({s_v}/{s_p}), r = {r} ({TIME_DG_QUADRATURE}), slab_size = {slab_size}, mesh = {MESH_NAME.strip('.xml')}")

slabs = [(start_time, start_time+slab_size)]
while slabs[-1][1] < end_time - 1e-8:
    slabs.append((slabs[-1][1], slabs[-1][1]+slab_size))

# get spatial function space
space_mesh = Mesh(MESH_NAME)
element = {
    "v": VectorElement("Lagrange", space_mesh.ufl_cell(), s_v),
    "p": FiniteElement("Lagrange", space_mesh.ufl_cell(), s_p),
}
Vh = FunctionSpace(space_mesh, MixedElement(*element.values())) # spatial function space for a single time point
logging.info(f"Number of spatial DoFs: {Vh.dim()} ({Vh.sub(0).dim()} + {Vh.sub(1).dim()})")
Uh = Function(Vh)
Phih = TestFunctions(Vh)

# boundaries
inflow = CompiledSubDomain("near(x[0], 0) && on_boundary")
outflow = CompiledSubDomain("near(x[0], 2.2) && on_boundary")
walls = CompiledSubDomain("near(x[1], 0) || near(x[1], 0.41) && on_boundary")
cylinder = CompiledSubDomain("x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3 && on_boundary")

facet_marker = MeshFunction("size_t", space_mesh, 1)
facet_marker.set_all(0)
inflow.mark(facet_marker, 1)
outflow.mark(facet_marker, 2)
walls.mark(facet_marker, 3)
cylinder.mark(facet_marker, 4)

ds_cylinder = Measure("ds", subdomain_data=facet_marker, subdomain_id=4)

# initial condition on slab
U0 = Function(Vh)
v0, p0 = split(U0)
U0 = interpolate(Constant((0.,0.,0.)), Vh)

# split functions into velocity and pressure components
v, p = split(Uh)
phi_v, phi_p = Phih

# pre-assemble drag and lift
dU = TrialFunction(Vh) 
dv, dp = split(dU)
D = 0.1
v_bar = 2/3 * 4.0*1.5*0.205*(0.41 - 0.205) / pow(0.41, 2)
n = FacetNormal(space_mesh)
drag_vector = assemble(
    2/(v_bar**2*D)*
    (
    - dot(dp * Identity(len(dv)) , -n)[0]
    + nu * dot(grad(dv), -n)[0]
    ) * ds_cylinder
).get_local()

lift_vector = assemble(
    2/(v_bar**2*D)*
    (
    - dot(dp * Identity(len(dv)) , -n)[1]
    + nu * dot(grad(dv), -n)[1]
    ) * ds_cylinder
).get_local()

drag_values = []
lift_values = []
times_draglift = []
total_n_dofs = 0
total_preprocessing_time = 0.
total_solve_time = 0.
total_postprocessing_time = 0.

xdmffile_v = XDMFFile(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/velocity.xdmf")
xdmffile_p = XDMFFile(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/pressure.xdmf")

#####################
# Time slabbing loop:
for k, slab in enumerate(slabs):
    logging.info(f"Solving on slab_{k} = Ω x ({round(slab[0],5)}, {round(slab[1],5)}) ...")
    print(f"Solving on slab_{k} = Ω x ({round(slab[0],5)}, {round(slab[1],5)}) ...")
    start_preprocessing_time = time.time()

    #########################################
    # Create temporal finite element object #
    #########################################
    Time = {
        "v" : TimeFE(r=r, a=slab[0], b=slab[1], n_time=1, n_q_points=r+1),
        "p": TimeFE(r=r, a=slab[0], b=slab[1], n_time=1, n_q_points=r+1)
    }
    
    V = FunctionSpace(space_mesh, MixedElement(*[element["v"] for _ in range(Time["v"].n_dofs)], *[element["p"] for _ in range(Time["p"].n_dofs)]))
    n_subspaces = V.num_sub_spaces()
    space_string = "Spaces = ["
    for i in range(n_subspaces):
        if V.sub(i).num_sub_spaces() == 2:
            space_string += "v,"
        else:
            space_string += "p,"
    logging.info(space_string + "]")
    
    U_kh = Function(V)
    Phi_kh = TestFunctions(V)

    # split U_kh and Phi_kh into velocity and pressure parts
    U = {"v": split(U_kh)[:Time["v"].n_dofs], "p": split(U_kh)[Time["v"].n_dofs:]}
    Phi = {"v": Phi_kh[:Time["v"].n_dofs], "p": Phi_kh[Time["v"].n_dofs:]}
    
    # start with "empty" space-time variational form
    F = Constant(0.)*U["p"][0]*Phi["p"][0]*dx
    
    # ================= #
    #   (v,v) - Block   #
    # ================= #
    
    # volume integrals
    for time_element in Time["v"].mesh:
        # assemble linear terms
        for i in Time["v"].local_dofs[time_element]:
            for j in Time["v"].local_dofs[time_element]:
                for (t_q, w_q) in Time["v"].quadrature[time_element]:
                    # TODO: to reduce the number of terms in the sum, the sum over the temporal quadrature can be evaluated prior to adding to the form F
                    F += Constant(w_q * Time["v"].dt_phi[j](t_q) * Time["v"].phi[i](t_q)) \
                        * dot(U["v"][j], Phi["v"][i]) * dx
                    F += Constant(w_q * Time["v"].phi[j](t_q) * Time["v"].phi[i](t_q)) \
                        * Constant(nu) * inner(grad(U["v"][j]), grad(Phi["v"][i])) * dx
        # assemble nonlinearity
        for i in Time["v"].local_dofs[time_element]:
            for j in Time["v"].local_dofs[time_element]:
                for l in Time["v"].local_dofs[time_element]:
                    # NOTE: For nonlinearities make sure that the temporal quadrature is fine enough
                    # E.g. for the nonlinearity u^2, we need to be able to integrate polynomials of degree 3r exactly in time
                    #      for this we need Gauss-Legendre quadrature of degree >= (3r+1)/2
                    for (t_q, w_q) in Time["v"].quadrature_fine[time_element]:
                        F += Constant(w_q * Time["v"].phi[j](t_q) * Time["v"].phi[l](t_q) * Time["v"].phi[i](t_q)) \
                            * dot(dot(grad(U["v"][j]), U["v"][l]), Phi["v"][i])* dx   
                        
    # RHS integral
    for n, time_element in enumerate(Time["v"].mesh):
        for i in Time["v"].local_dofs[time_element]:
            # initial condition
            if n == 0:
                F -=  Constant(Time["v"].phi[i](time_element[0]+Time["v"].epsilon)) * inner(v0, Phi["v"][i]) * dx

    # jump terms (NOTE: For Gauss-Lobatto dG basis, we could hard code the values for the jump term)
    for n, time_element in enumerate(Time["v"].mesh):
        # a) v_m^+ * φ_m^{v,+}
        for i in Time["v"].local_dofs[time_element]:
            for j in Time["v"].local_dofs[time_element]:
                F += Constant(Time["v"].phi[j](time_element[0]+Time["v"].epsilon) * Time["v"].phi[i](time_element[0]+Time["v"].epsilon)) * inner(U["v"][j], Phi["v"][i]) * dx

        # b) v_{m-1}^- * φ_m^{v,+}
        if n > 0:
            prev_time_element = Time["v"].mesh[n-1]
            for i in Time["v"].local_dofs[time_element]:
                for j in Time["v"].local_dofs[prev_time_element]:
                    F += Constant((-1.) * Time["v"].phi[j](prev_time_element[1]-Time["v"].epsilon) * Time["v"].phi[i](time_element[0]+Time["v"].epsilon)) * inner(U["v"][j], Phi["v"][i]) * dx       

    # ================= #
    #   (v,p) - Block   #
    # ================= #
    
    # volume integral
    for time_element in Time["v"].mesh: # ASSUMPTION: same mesh for "v" and "p"
        for i in Time["v"].local_dofs[time_element]:
            for j in Time["p"].local_dofs[time_element]:
                for (t_q, w_q) in Time["v"].quadrature[time_element]:
                    F += Constant(w_q * Time["p"].phi[j](t_q) * Time["v"].phi[i](t_q)) \
                        * Constant(-1.) * U["p"][j] * div(Phi["v"][i]) * dx
    
    # ================= #
    #   (p,v) - Block   #
    # ================= #
        
    # volume integral
    for time_element in Time["v"].mesh: # ASSUMPTION: same mesh for "v" and "p"
        for i in Time["p"].local_dofs[time_element]:
            for j in Time["v"].local_dofs[time_element]:
                for (t_q, w_q) in Time["v"].quadrature[time_element]:
                    F += Constant(w_q * Time["v"].phi[j](t_q) * Time["p"].phi[i](t_q)) \
                        * Constant(-1.) * div(U["v"][j]) * Phi["p"][i] * dx

    # define time dependent Dirichlet boundary conditions
    bcs = []
    for i, t_q in enumerate(Time["v"].dof_locations):
        inflow_parabola = ("0","0")
        if PROBLEM == "2D-2":
            inflow_parabola = ("4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)", "0")
        elif PROBLEM == "2D-3":
            inflow_parabola = ("4.0*1.5*sin(0.125*pi*t)*x[1]*(0.41 - x[1]) / pow(0.41, 2)", "0")
        else:
            raise NotImplementedError(f"Problem {PROBLEM} has not been implemented yet.")
        
        bcs.append(DirichletBC(V.sub(i), Expression(inflow_parabola, degree=2, pi=np.pi, t=t_q), inflow))
        bcs.append(DirichletBC(V.sub(i), Constant((0, 0)), walls))
        bcs.append(DirichletBC(V.sub(i), Constant((0, 0)), cylinder))

    total_preprocessing_time += time.time() - start_preprocessing_time

    # solve problem
    start_solve_time = time.time()
    solve(F == 0, U_kh, bcs)
    total_solve_time += time.time() - start_solve_time

    # postprocess solution
    start_postprocessing_time = time.time()
    solutions_v = [U_kh.sub(i, deepcopy=True).vector() for i in range(Time["v"].n_dofs)]
    offset_v = Time["v"].n_dofs
    solutions_p = [U_kh.sub(i + offset_v, deepcopy=True).vector() for i in range(Time["p"].n_dofs)]
    
    # get v0 for next slab
    U0.vector().set_local(np.concatenate((
        Time["v"].get_solution_at_time(slab[1]-Time["v"].epsilon, solutions_v),
        Time["p"].get_solution_at_time(slab[1]-Time["p"].epsilon, solutions_p) #np.zeros((Vh.sub(1).dim(),))
    )))
    v0, p0 = U0.split(deepcopy=True)

    v0.rename("velocity", "solution")
    p0.rename("pressure", "solution")
    
    # plot final solution on slab
    # print(f"t = {slab[1]}:")
    # c = plot(sqrt(dot(v0, v0)), title="Velocity")
    # plt.colorbar(c, orientation="horizontal")
    # plt.show()
    # c = plot(p0, title="Pressure")
    # plt.colorbar(c, orientation="horizontal")
    # plt.show()

    # Save solution to file (XDMF/HDF5)
    xdmffile_v.write(v0, slab[1])
    xdmffile_p.write(p0, slab[1])

    # compute functional values
    total_n_dofs += Vh.sub(0).dim() * Time["v"].n_dofs + Vh.sub(1).dim() * Time["p"].n_dofs
    for time_element in Time["v"].mesh:
        for t_q in [time_element[0]] + [tq for (tq, _) in Time["v"].quadrature_fine[time_element]] + [time_element[1]]:
            Uq = np.concatenate((
                Time["v"].get_solution_at_time(t_q, solutions_v),
                Time["p"].get_solution_at_time(t_q, solutions_p)
            ))
            drag_values.append(drag_vector.dot(Uq))
            lift_values.append(lift_vector.dot(Uq))
            times_draglift.append(t_q)
        drag_values.append(np.inf)
        lift_values.append(np.inf)
        times_draglift.append(time_element[1])
    total_postprocessing_time += time.time() - start_postprocessing_time
    logging.info("Done.\n")
    
logging.info("------------")
logging.info("| RESULTS: |")
logging.info("------------")
logging.info(f"Space-time Dofs: {total_n_dofs:,}")
cpu_time = round(time.time() - cpu_start_time, 5)
logging.info(f"CPU Time:            {cpu_time} s")
logging.info(f"Preprocessing Time:  {total_preprocessing_time:.4f} s ({100. * total_preprocessing_time / cpu_time:.2f} %) ")
logging.info(f"Solve Time:          {total_solve_time:.4f} s ({100. * total_solve_time / cpu_time:.2f} %) ")
logging.info(f"Postprocessing Time: {total_postprocessing_time:.4f} s ({100. * total_postprocessing_time / cpu_time:.2f} %)  \n\n")

# save times and drag/lift values to numpy array
np.save(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/times_draglift.npy", times_draglift)
np.save(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/drag_values.npy", drag_values)
np.save(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/lift_values.npy", lift_values)

#save plots as png
plt.title("Drag")
plt.plot(times_draglift, drag_values)
plt.savefig(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/drag.png")
# clear plot
plt.clf()

plt.title("Lift")
plt.plot(times_draglift, lift_values)
plt.savefig(f"results_equal_dG/{PROBLEM}/{TIME_DG_QUADRATURE}_r_{r}_dt_{slab_size}_mesh_{MESH_NAME.strip('.xml')}/lift.png")

