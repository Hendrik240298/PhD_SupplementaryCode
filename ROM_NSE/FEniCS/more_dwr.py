import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
from dolfin import *

# Problem data
inflow_parabola = ("4.0*1.5*x[1]*(0.41 ^- x[1]) / pow(0.41, 2)", "0")
nu = Constant(0.001)
theta = 0.5
T = 5.0
n_timesteps = 100
dt = T / n_timesteps
PLOT = False

# Create mesh
# Link to mesh generation:
# https://colab.research.google.com/drive/1CkyFWD_FCVOhey_ydJ9PtWBJ9ii6hX76?usp=sharing
"""
channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)
"""
mesh = Mesh("schaefer_turek_2D.xml")
if PLOT:
    plt.figure(figsize=(10, 7))
    plot(mesh)
    plt.show()

element = {
    "v": VectorElement("Lagrange", mesh.ufl_cell(), 2),
    "p": FiniteElement("Lagrange", mesh.ufl_cell(), 1),
}
V = FunctionSpace(mesh, MixedElement(*element.values()))
_V = V.sub(0)
_P = V.sub(1)

# Define boundaries
inflow = "near(x[0], 0)"
outflow = "near(x[0], 2.2)"
walls = "near(x[1], 0) || near(x[1], 0.41)"
cylinder = "on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3"

# Define boundary conditions
bc_v_inflow = DirichletBC(_V, Expression(inflow_parabola, degree=2), inflow)
bc_v_inflow_homogen = DirichletBC(_V, Constant((0, 0)), inflow)
bc_v_walls = DirichletBC(_V, Constant((0, 0)), walls)
bc_v_cylinder = DirichletBC(_V, Constant((0, 0)), cylinder)
bc_v = [bc_v_inflow, bc_v_walls, bc_v_cylinder]
bc_p = []
bc = bc_v + bc_p
bc_v_homogen = [bc_v_inflow_homogen, bc_v_walls, bc_v_cylinder]
bc_homogen = bc_v_homogen + bc_p

# Define trial and test functions and function at old time step
U = Function(V)  # current solution
dU = TrialFunction(V)  # Newton update
Phi_U = TestFunctions(V)
U_n = Function(V)  # old timestep solution

# Split functions into velocity and pressure components
v, p = split(U)
dv, dp = split(dU)
phi_v, phi_p = Phi_U
v_n, p_n = split(U_n)

# initial guess needs to satisfy (time-independent) inhomogeneous BC
U.vector()[:] = 0.0
for _bc in bc:
    _bc.apply(U.vector())

# Define variational forms
F = (
    Constant(1 / dt) * dot(v - v_n, phi_v)
    + Constant(theta) * nu * inner(grad(v), grad(phi_v))
    + Constant(theta) * dot(dot(grad(v), v), phi_v)
    + Constant(1 - theta) * nu * inner(grad(v_n), grad(phi_v))
    + Constant(1 - theta) * dot(dot(grad(v_n), v_n), phi_v)
    - p * div(phi_v)
    - div(v) * phi_p
) * dx

# Gateaux derivative of F w.r.t. to U in direction of dU
J = derivative(F, U, dU)

# Handcoded derivative
A = (
    Constant(1 / dt) * dot(dv, phi_v)
    + Constant(theta) * nu * inner(grad(dv), grad(phi_v))
    + Constant(theta) * dot(dot(grad(dv), v), phi_v)
    + Constant(theta) * dot(dot(grad(v), dv), phi_v)
    - dp * div(phi_v)
    - div(dv) * phi_p
) * dx

# For debugging the derivative of the variational formulation:
# assert np.max(np.abs((assemble(A)-assemble(J)).array())) == 0., f"Handcoded derivative and auto-diff generated derivative should be the same. Difference is {np.max(np.abs((assemble(A)-assemble(J)).array()))}."

# Newton update
dU = Function(V)

# Newton parameters
NEWTON_TOL = 1e-8
MAX_N_NEWTON_STEPS = 60
# decide whether system matrix should be build in Newton step
NONLINEAR_RHO = 0.1

# Line search parameters
MAX_N_LINE_SEARCH_STEPS = 10
LINE_SEARCH_DAMPING = 0.6


# Perform time-stepping
t = 0
while t < T:
    print(f"\n\nt = {round(t,5)}:\n==========")
    U_n.vector()[:] = U.vector()

    # Newton table
    newton_table = rich.table.Table(title="Newton solver")
    newton_table.add_column("Step", justify="right")
    newton_table.add_column("Residuum", justify="right")
    newton_table.add_column("Residuum fraction", justify="right")
    newton_table.add_column("Assembled matrix", justify="center")
    newton_table.add_column("Linesearch steps", justify="right")

    # Newton iteration
    system_matrix = None
    system_rhs = assemble(-F)
    for _bc in bc_homogen:
        _bc.apply(system_rhs)
    newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)
    # print(f"Initial Newton residuum: {newton_residuum}")
    newton_step = 1

    if newton_residuum < NEWTON_TOL:
        print(f"Newton residuum: {newton_residuum}")

    # Newton loop
    while newton_residuum > NEWTON_TOL and newton_step < MAX_N_NEWTON_STEPS:
        old_newton_residuum = newton_residuum

        system_rhs = assemble(-F)
        for _bc in bc_homogen:
            _bc.apply(system_rhs)
        newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

        if newton_residuum < NEWTON_TOL:
            # print(f"Newton residuum: {newton_residuum}")
            newton_table.add_row("-", f"{newton_residuum:.4e}", "-", "-", "-")

            console = rich.console.Console()
            console.print(newton_table)
            break

        if newton_residuum / old_newton_residuum > NONLINEAR_RHO:
            # For debugging the derivative of the variational formulation:
            # assert np.max(np.abs((assemble(A)-assemble(J)).array())) == 0., f"Handcoded derivative and auto-diff generated derivative should be the same. Difference is {np.max(np.abs((assemble(A)-assemble(J)).array()))}."
            system_matrix = assemble(A)
            for _bc in bc_homogen:
                _bc.apply(system_matrix, system_rhs)

        solve(system_matrix, dU.vector(), system_rhs)

        for linesearch_step in range(MAX_N_LINE_SEARCH_STEPS):
            U.vector()[:] += dU.vector()[:]

            system_rhs = assemble(-F)
            for _bc in bc_homogen:
                _bc.apply(system_rhs)
            new_newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

            if new_newton_residuum < newton_residuum:
                break
            else:
                U.vector()[:] -= dU.vector()[:]

            dU.vector()[:] *= LINE_SEARCH_DAMPING

        assembled_matrix = newton_residuum / old_newton_residuum > NONLINEAR_RHO
        # print(f"Newton step: {newton_step} | Newton residuum: {newton_residuum} | Residuum fraction: {newton_residuum/old_newton_residuum } | Assembled matrix: {assembled_matrix} | Linesearch steps: {linesearch_step}")
        newton_table.add_row(
            str(newton_step),
            f"{newton_residuum:.4e}",
            f"{round(newton_residuum/old_newton_residuum, 4):#.4f}",
            str(assembled_matrix),
            str(linesearch_step),
        )
        newton_step += 1

    t += dt
    # f.write(u, t)

    if PLOT:
        c = plot(sqrt(dot(v, v)), title="Velocity")
        plt.colorbar(c, orientation="horizontal")
        plt.show()
