import math
import os
import random
import re
import time
from multiprocessing import Pool

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
import scipy
from dolfin import *

import logging
# configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


class FOM:
    # constructor
    def __init__(self, t, T, dt, theta, nu, SAVE_DIR="results/"):
        self.t = t
        self.T = T
        self.dt = dt
        self.theta = theta
        self.nu = nu  # \nu=\mu and \rho=1

        # IO data
        self.SAVE_DIR = SAVE_DIR

        # Newton parameters
        self.NEWTON_TOL = 1e-8
        self.MAX_N_NEWTON_STEPS = 2000
        # decide whether system matrix should be build in Newton step
        self.NONLINEAR_RHO = 0.1

        # Line search parameters
        self.MAX_N_LINE_SEARCH_STEPS = 10
        self.LINE_SEARCH_DAMPING = 0.6

        self.time_points = np.arange(self.t, self.T + self.dt, self.dt)
        print(f"FIRST/LATEST TIME POINT:    {self.time_points[0]}/{self.time_points[-1]}")
        print(f"NUMBER OF TIME POINTS:      {self.time_points.shape[0]}")

        # L = 2*radius_cylinder = 0.1
        # V = 1
        # rho = 1
        # mu = nu
        logging.info(f"Reynolds number: {0.1/float(self.nu)}")

        # self.mesh = Mesh("schaefer_turek_2D_pygmsh_hendrik.xml")
        
        
        # self.mesh = Mesh("schaefer_turek_2D.xml")
        
        self.mesh_name = "mesh_new"
        self.mesh = Mesh("meshes/" + self.mesh_name + ".xml")

        # check dimension of the mesh 
        print("mesh dim = ", self.mesh.geometric_dimension())

        # plot the mesh 
        plot(self.mesh, linestyle='-', linewidth=0.5)
        plt.xlim(-0.02, 2.22)
        plt.ylim(-0.02, 0.43)
        
        # Adjust layout to remove excess borders
        plt.tight_layout()
        
        plt.savefig("plots/mesh_plot.pdf", bbox_inches='tight')
        # save mesh plot 

        #close fig 
        plt.close()
        
        # zoom in to mesh plot x:[0.075, 0.325] y:[0.075, 0.325]
        plot(self.mesh, linestyle='-', linewidth=0.85)
        plt.xlim(0.075, 0.325)
        plt.ylim(0.075, 0.325)
        # Adjust layout to remove excess borders
        plt.tight_layout()
        plt.savefig("plots/mesh_plot_zoomed.pdf", bbox_inches='tight')
        
        
        plt.rcParams['text.usetex'] = True
        # Plot the full mesh
        fig, ax = plt.subplots()
        
        plot(self.mesh, linestyle='-', linewidth=0.5)
        plt.tick_params(axis="both", which="major", labelsize=13)

        # for cell in cells:
        #     polygon = plt.Polygon(coordinates[cell], edgecolor='black', facecolor='none', linewidth=0.5)
        #     ax.add_patch(polygon)

        ax.set_xlim(-0.02, 2.22)
        ax.set_ylim(-0.02, 0.43)

        # Define the zoomed-in area
        zoom_xlim = (0.075, 0.325)  # Example x limits for zoomed in plot
        zoom_ylim = (0.075, 0.325)  # Example y limits for zoomed in plot

        # Create an inset axis for the zoomed-in plot
        # ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right')
        ax_inset = fig.add_axes([0.4, 0.65, 0.4, 0.3])

        # Plot the zoomed-in mesh on the inset
        # for cell in cells:
        #     polygon = plt.Polygon(coordinates[cell], edgecolor='black', facecolor='none', linewidth=0.5)
        #     ax_inset.add_patch(polygon)

        plot(self.mesh, linestyle='-', linewidth=0.3)
        plt.xlim(0.075, 0.325)
        plt.ylim(0.075, 0.325)

        # remove axis labels
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xlabel(None)
        
        ax_inset.set_xlim(zoom_xlim)
        ax_inset.set_ylim(zoom_ylim)

        # Mark the zoomed area on the main plot
        mark_inset_result = mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", lw=1)

        # If you're not sure how many values mark_inset returns, you can do:
        # rect_patch, connectors, *other_values = mark_inset_result

        # Save the combined plot
        plt.savefig("plots/mesh_plot_with_zoom.pdf", bbox_inches='tight')
        
        # self.mesh = Mesh("schaefer_turek_2D_pygmsh_hendrik.xml")
        
        # self.mesh = Mesh("schaefer_turek_2D_pygmsh_no_circle_fine.xml")
        
        # plot(self.mesh, title="spatial mesh")
        # plt.xlim(-0.001, 0.1)
        # plt.ylim(-0.001, 0.1)
        # plt.show()  

        element = {
            "v": VectorElement("Lagrange", self.mesh.ufl_cell(), 2),
            "p": FiniteElement("Lagrange", self.mesh.ufl_cell(), 1),
        }

        # Define variational problem
        self.V = FunctionSpace(self.mesh, MixedElement(*element.values()))
        self._V = self.V.sub(0)
        self._P = self.V.sub(1)

        self.dofs = {
            "velocity": self._V.dim(),
            "pressure": self._P.dim(),
            "time": self.time_points.shape[0],
        }

        logging.info(f"DOFs: {self.dofs}")  

        # class inflow(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return near(x[0], 0) and on_boundary

        # class outflow(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return near(x[0], 2.2) and on_boundary
            
        # class walls(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return (near(x[1], 0) or near(x[1], 0.41)) and on_boundary
            
        # class cylinder(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return x[0]>0.1 and x[0]<0.3 and x[1]>0.1 and x[1]<0.3 and on_boundary

        inflow_parabola = ("4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)", "0")
        inflow = CompiledSubDomain("near(x[0], 0) && on_boundary")
        outflow = CompiledSubDomain("near(x[0], 2.2) && on_boundary")
        walls = CompiledSubDomain("near(x[1], 0) || near(x[1], 0.41) && on_boundary")
        cylinder = CompiledSubDomain("x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3 && on_boundary")

        facet_marker = MeshFunction("size_t", self.mesh, 1)
        facet_marker.set_all(0)
        inflow.mark(facet_marker, 1)
        outflow.mark(facet_marker, 2)
        walls.mark(facet_marker, 3)
        cylinder.mark(facet_marker, 4)
        
        
        # # plot only the boundary of mesh not the grid itself
        # plot(facet_marker)
        # plt.show()
        

        # print the coordinates of the first velocity dof
        print(self.V.tabulate_dof_coordinates()[0:5])

        # find all duplicates in self.V.tabulate_dof_coordinates()
        # Get the coordinates
        coords = self.V.tabulate_dof_coordinates()

        # Find unique coordinates and their counts
        unique_coords, counts = np.unique(coords, axis=0, return_counts=True)

        # Find the duplicates
        duplicates = unique_coords[counts > 3]

        print("Duplicate coordinates: ", duplicates)
        print("count of duplicates: ", len(duplicates))
        
        # from fenics import plot
        # plot(self.mesh)
        # # colorcode the first dof locattion 
        # plt.scatter(self.V.tabulate_dof_coordinates()[0][0], self.V.tabulate_dof_coordinates()[0][1], color='red')
        # # second 
        # plt.scatter(self.V.tabulate_dof_coordinates()[1][0], self.V.tabulate_dof_coordinates()[1][1], color='blue')
        # # third
        # plt.scatter(self.V.tabulate_dof_coordinates()[2][0], self.V.tabulate_dof_coordinates()[2][1], color='green')
        # # fourth
        # plt.scatter(self.V.tabulate_dof_coordinates()[3][0], self.V.tabulate_dof_coordinates()[3][1], color='yellow')
        # # fifth
        # plt.scatter(self.V.tabulate_dof_coordinates()[4][0], self.V.tabulate_dof_coordinates()[4][1], color='black', marker='x')
        # # sixth
        # plt.scatter(self.V.tabulate_dof_coordinates()[5][0], self.V.tabulate_dof_coordinates()[5][1], color='pink')
        # plt.show()
        # # exit()
        
        

        self.ds_cylinder = Measure("ds", subdomain_data=facet_marker, subdomain_id=4)

     
        # Define boundary conditions
        bc_v_inflow = DirichletBC(self._V, Expression(inflow_parabola, degree=2), inflow)
        bc_v_walls = DirichletBC(self._V, Constant((0, 0)), walls)
        bc_v_cylinder = DirichletBC(self._V, Constant((0, 0)), cylinder)
        
        bc_v_inflow_homogen = DirichletBC(self._V, Constant((0, 0)), inflow)

        
        bc_v = [bc_v_inflow, bc_v_walls, bc_v_cylinder]
        bc_p = []
        bc_v_homogen = [bc_v_inflow_homogen, bc_v_walls, bc_v_cylinder]

        # BC needed for solve
        self.bc = bc_v + bc_p
        self.bc_homogen = bc_v_homogen + bc_p

        # Define trial and test functions and function at old time step
        self.U = Function(self.V)  # current solution
        dU = TrialFunction(self.V)  # Newton update
        Phi_U = TestFunctions(self.V)
        self.U_n = Function(self.V)  # old timestep solution

        # Split functions into velocity and pressure components
        v, p = split(self.U)
        dv, dp = split(dU)
        phi_v, phi_p = Phi_U
        v_n, p_n = split(self.U_n)

        # initial guess needs to satisfy (time-independent) inhomogeneous BC
        self.U.vector()[:] = 0.0
        for _bc in self.bc:
            _bc.apply(self.U.vector())

        self.matrix = {
            "primal": {},
        }

        self.matrix["primal"]["mass"] = scipy.sparse.csr_matrix(
            as_backend_type(assemble(dot(dv, phi_v) * dx)).mat().getValuesCSR()[::-1],
            shape=(
                self.dofs["velocity"] + self.dofs["pressure"],
                self.dofs["velocity"] + self.dofs["pressure"],
            ),
        )[: self.dofs["velocity"], : self.dofs["velocity"]]

        self.matrix["primal"]["laplace"] = scipy.sparse.csr_matrix(
            as_backend_type(assemble(inner(grad(dv), grad(phi_v)) * dx)).mat().getValuesCSR()[::-1],
            shape=(
                self.dofs["velocity"] + self.dofs["pressure"],
                self.dofs["velocity"] + self.dofs["pressure"],
            ),
        )[: self.dofs["velocity"], : self.dofs["velocity"]]

        self.matrix["primal"]["pressure"] = scipy.sparse.csr_matrix(
            as_backend_type(assemble(dp * div(phi_v) * dx)).mat().getValuesCSR()[::-1],
            shape=(
                self.dofs["velocity"] + self.dofs["pressure"],
                self.dofs["velocity"] + self.dofs["pressure"],
            ),
        )[: self.dofs["velocity"], self.dofs["velocity"] :]

        self.matrix["primal"]["div"] = scipy.sparse.csr_matrix(
            as_backend_type(assemble(div(dv) * phi_p * dx)).mat().getValuesCSR()[::-1],
            shape=(
                self.dofs["velocity"] + self.dofs["pressure"],
                self.dofs["velocity"] + self.dofs["pressure"],
            ),
        )[self.dofs["velocity"] :, : self.dofs["velocity"]]

        self.matrix["primal"]["supremizer_system"] = assemble(dot(dv, phi_v) * dx)
        self.matrix["primal"]["supremizer_rhs"] = assemble(dp * div(phi_v) * dx)

        # Define variational forms
        self.F = (
            Constant(1 / dt) * dot(v - v_n, phi_v)
            + Constant(theta) * self.nu * inner(grad(v), grad(phi_v))
            + Constant(theta) * dot(dot(grad(v), v), phi_v)
            + Constant(1 - theta) * self.nu * inner(grad(v_n), grad(phi_v))
            + Constant(1 - theta) * dot(dot(grad(v_n), v_n), phi_v)
            - p * div(phi_v)
            - div(v) * phi_p
        ) * dx

        # Handcoded derivative
        self.A = (
            Constant(1 / dt) * dot(dv, phi_v)
            + Constant(theta) * self.nu * inner(grad(dv), grad(phi_v))
            + Constant(theta) * dot(dot(grad(dv), v), phi_v)
            + Constant(theta) * dot(dot(grad(v), dv), phi_v)
            - dp * div(phi_v)
            - div(dv) * phi_p
        ) * dx


        # define snapshot matrix
        self.Y = {
            "velocity": np.zeros((self.dofs["velocity"], self.dofs["time"])),
            "pressure": np.zeros((self.dofs["pressure"], self.dofs["time"])),
        }

        # define functional values
        self.functional_values = np.zeros((self.dofs["time"] - 1,))


        # c = 2/(rho V^2 D) J
        # rho = 1
        # D = 0.1 (diameter cylinder)
        # V = 2/3 v_x(H/2) (mean inflow velocity)

        # TODO: THINK about if -n is BULLSHIT or not

        D = 0.1
        V = 2/3 * 4.0*1.5*0.205*(0.41 - 0.205) / pow(0.41, 2)
        print("V = ", V)
        n = FacetNormal(self.mesh)
        self.drag_vector = assemble(
            2/(V**2*D)*
            (
            - dot(dp * Identity(len(dv)) , -n)[0]
            + self.nu * dot(grad(dv), -n)[0]
            ) * self.ds_cylinder
        ).get_local()

        self.lift_vector = assemble(
            2/(V**2*D)*
            (
            - dot(dp * Identity(len(dv)) , -n)[1]
            + self.nu * dot(grad(dv), -n)[1]
            ) * self.ds_cylinder
        ).get_local()


        # # define matrices # TODO: need for ROM later
        # self.matrix = {
        #     "primal": {"system": PETScMatrix(), "rhs": PETScMatrix()},
        #     "dual": {"system": PETScMatrix(), "rhs": PETScMatrix()},
        # }


    def compute_drag_lift_time_step(self, velocity, pressure):
        solution = np.concatenate(
            (
                velocity,
                pressure,
            )
        )
        drag_force = self.drag_vector.dot(solution)
        lift_force = self.lift_vector.dot(solution)

        return drag_force, lift_force

    def compute_drag_lift(self):
        offset = 0 #100 #why? :D

        self.drag_force = np.zeros((self.dofs["time"],))
        self.lift_force = np.zeros((self.dofs["time"],))
        self.press_diff = np.zeros((self.dofs["time"],))

        v, p = self.U_n.split()

        for i, t in list(enumerate(self.time_points)):
            # test the snapshots
            # print("velo[i] =     ", self.Y["velocity"][0:10, i])
            # print("pressure[i] = ", self.Y["pressure"][0:10, i])
            
            self.drag_force[i], self.lift_force[i] = self.compute_drag_lift_time_step(self.Y["velocity"][:, i], self.Y["pressure"][:, i])
            # DEBUG 
            # print("drag force[i] = ", self.drag_force[i])
            # print("lift force[i] = ", self.lift_force[i])
            self.U_n.vector().set_local(
                np.concatenate(
                    (
                        self.Y["velocity"][:, i], 
                        self.Y["pressure"][:, i],
                    )
                )
            )
            
            self.press_diff[i] =p(0.15,0.2)-p(0.25,0.2)


        # plot results in subplots 
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(self.time_points[offset:], self.drag_force[offset:], label="drag")
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("drag")
        ax[0].grid()
        # set y axis limits to 3.1 and 3.19
        ax[0].set_ylim(3.14, 3.23)
        
        
        ax[1].plot(self.time_points[offset:], self.lift_force[offset:], label="lift")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("lift")
        ax[1].grid()
        ax[2].plot(self.time_points[offset:], self.press_diff[offset:], label="pressure diff")
        ax[2].set_xlabel("time")
        ax[2].set_ylabel("pressure diff")
        ax[2].grid()
        # set y axis limits to 2.4 and 2.6
        ax[2].set_ylim(2.4, 2.5)
        
        plt.savefig("plots/drag_lift.png")
        
        logging.info("FOM drag and lift computed and saved")
        
        # plt.show()


    def save_solution(self):
        pattern = f"solution_{self.mesh_name}_" + r"\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.dt, self.T, self.theta, float(self.nu)])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    velocity=self.Y["velocity"],
                    pressure=self.Y["pressure"],
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return

        file_name = f"results/solution_{self.mesh_name}_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            velocity=self.Y["velocity"],
            pressure=self.Y["pressure"],
            parameters=parameters,
            compression=True,
        )
        print(f"Saved as {file_name}")



    def save_solution_parallel(self):
        '''
        For compute_FOM_parameter_space.py
        '''
        pattern = f"solution_{self.mesh_name}_{int(0.1/float(self.nu))}_" + r"_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.dt, self.T, self.theta, float(self.nu)])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    velocity=self.Y["velocity"],
                    pressure=self.Y["pressure"],
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return

        file_name = f"results/solution_{self.mesh_name}_{int(0.1/float(self.nu))}_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            velocity=self.Y["velocity"],
            pressure=self.Y["pressure"],
            parameters=parameters,
            compression=True,
        )
        print(f"Saved as {file_name}")

    def load_solution_parallel(self):
        '''
        For load compute_FOM_parameter_space.py results
        '''
        
        files = [self.SAVE_DIR + "solution_" + self.mesh_name + "_" + str(int(0.1/float(self.nu))) + "_000000.npz"]

        parameters = np.array([self.dt, self.T, self.theta, float(self.nu)])

        for file in files:
            # print("file = ", file)
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.Y["velocity"] = tmp["velocity"]
                self.Y["pressure"] = tmp["pressure"]
                print(f"Loaded {file}")

                return True
        print(f"No solution found for {int(0.1/float(self.nu))}")
        return False

    def load_solution(self):
        pattern = f"solution_{self.mesh_name}_" + r"_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.dt, self.T, self.theta, float(self.nu)])

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.Y["velocity"] = tmp["velocity"]
                self.Y["pressure"] = tmp["pressure"]
                print(f"Loaded {file}")

                # for i, _ in enumerate(self.time_points):
                #     v, p = self.U.split()
                #     v.vector().set_local(self.Y["velocity"][:, i])
                #     c = plot(sqrt(dot(v, v)), title="Velocity")
                #     plt.colorbar(c, orientation="horizontal")
                #     plt.show()

                return True
        return False

    def solve_functional_trajectory(self):
        pass  # TODO

    # assemble system matrices and rhs
    def assemble_system(self, force_recompute=False):
        pass  # TODO

    def assemble_lifting_matrices(self, lifting):
        solution = np.concatenate(
            (
                lifting["velocity"],
                np.zeros((self.dofs["pressure"],)),
            )
        )
        U_lifting = Function(self.V)  # lifting function
        U_lifting.vector().set_local(solution)
        dv, _ = split(U_lifting)

        U = TrialFunction(self.V)
        Phi_U = TestFunctions(self.V)

        # Split functions into velocity and pressure components
        v, _ = split(U)
        phi_v, _ = Phi_U

        # assemble Frechet derivative C^l := C(l)
        self.lifting_matrix = scipy.sparse.csr_matrix(
            as_backend_type(
                assemble(dot(dot(grad(v), dv) + dot(grad(dv), v), phi_v) * dx)
            )
            .mat()
            .getValuesCSR()[::-1],
            shape=(
                self.dofs["velocity"] + self.dofs["pressure"],
                self.dofs["velocity"] + self.dofs["pressure"],
            ),
        )[: self.dofs["velocity"], : self.dofs["velocity"]]

        # assemble convection term evaluated at lifting
        self.lifting_rhs = (
            # DEBUG HF: remove duplicate
            # -np.array(
            #     self.dt * self.nu * self.matrix["primal"]["laplace"].dot(lifting["velocity"]),
            #     dtype=np.float64,
            # )
            -self.dt
            * np.array(
                assemble(
                    Constant(self.nu) * inner(grad(dv), grad(phi_v)) * dx
                    + dot(dot(grad(dv), dv), phi_v) * dx
                )
            )[: self.dofs["velocity"]]
        )
        print("Assembled lifting matrices + rhs")

    def assemble_linear_operators(self):
        self.velocity_lin_operator_theta = (
            self.matrix["primal"]["mass"]
            + float(self.dt * self.theta * self.nu) * self.matrix["primal"]["laplace"]
            + float(self.dt * self.theta) * self.lifting_matrix
        )
        self.velocity_lin_operator_one_minus_theta = (
            self.matrix["primal"]["mass"]
            - float(self.dt * (1.0 - self.theta) * self.nu) * self.matrix["primal"]["laplace"]
            - float(self.dt * (1.0 - self.theta)) * self.lifting_matrix
        )
        self.pressure_lin_operator = -self.dt * self.matrix["primal"]["pressure"]

        logging.info("Assembled linear operators")

    # Solve one time_step
    def solve_primal_time_step(self, v_n_vector, p_n_vector):
        # Newton update
        dU = Function(self.V)
        # self.U_n.vector()[:] = u_n_vector.
        old_solution = np.concatenate((v_n_vector, p_n_vector))
        self.U_n.vector().set_local(old_solution)
        # print("old_solution = ", old_solution)
        # Newton table
        newton_table = rich.table.Table(title="Newton solver")
        newton_table.add_column("Step", justify="right")
        newton_table.add_column("Residuum", justify="right")
        newton_table.add_column("Residuum fraction", justify="right")
        newton_table.add_column("Assembled matrix", justify="center")
        newton_table.add_column("Linesearch steps", justify="right")

        # Newton iteration
        system_matrix = None
        system_rhs = assemble(-self.F)
        for _bc in self.bc_homogen:
            _bc.apply(system_rhs)
        newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

        newton_step = 1

        if newton_residuum < self.NEWTON_TOL:
            print(f"Newton residuum: {newton_residuum}")

        # Newton loop
        while newton_residuum > self.NEWTON_TOL and newton_step < self.MAX_N_NEWTON_STEPS:
            old_newton_residuum = newton_residuum

            system_rhs = assemble(-self.F)

            for _bc in self.bc_homogen:
                _bc.apply(system_rhs)
            newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

            if newton_residuum < self.NEWTON_TOL:
                # print(f"Newton residuum: {newton_residuum}")
                newton_table.add_row("-", f"{newton_residuum:.4e}", "-", "-", "-")

                console = rich.console.Console()
                console.print(newton_table)
                break

            if newton_residuum / old_newton_residuum > self.NONLINEAR_RHO:
                # For debugging the derivative of the variational formulation:
                # assert np.max(np.abs((assemble(A)-assemble(J)).array())) == 0., f"Handcoded derivative and auto-diff generated derivative should be the same. Difference is {np.max(np.abs((assemble(A)-assemble(J)).array()))}."
                system_matrix = assemble(self.A)
                for _bc in self.bc_homogen:
                    _bc.apply(system_matrix, system_rhs)

            solve(system_matrix, dU.vector(), system_rhs)
            # print("system_matrix = ", system_matrix.array()[:5][:5])
            # print("rhs = ", system_rhs.get_local()[:10])
            # print("dU =  ", dU.vector().get_local()[:10])

            for linesearch_step in range(self.MAX_N_LINE_SEARCH_STEPS):
                self.U.vector()[:] += dU.vector()[:]

                system_rhs = assemble(-self.F)
                for _bc in self.bc_homogen:
                    _bc.apply(system_rhs)
                new_newton_residuum = np.linalg.norm(np.array(system_rhs), ord=np.Inf)

                if new_newton_residuum < newton_residuum:
                    break
                else:
                    self.U.vector()[:] -= dU.vector()[:]

                dU.vector()[:] *= self.LINE_SEARCH_DAMPING

            assembled_matrix = newton_residuum / old_newton_residuum > self.NONLINEAR_RHO
            # print(f"Newton step: {newton_step} | Newton residuum: {newton_residuum} | Residuum fraction: {newton_residuum/old_newton_residuum } | Assembled matrix: {assembled_matrix} | Linesearch steps: {linesearch_step}")
            newton_table.add_row(
                str(newton_step),
                f"{newton_residuum:.4e}",
                f"{round(newton_residuum/old_newton_residuum, 4):#.4f}",
                str(assembled_matrix),
                str(linesearch_step),
            )
            newton_step += 1

        v, p = self.U.split()

        # c = plot(sqrt(dot(v, v)), title="Velocity")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        # print("v shape = ", v.vector().get_local()[: self._V.dim()].shape)
        # print("p shape = ", p.vector().get_local()[self._V.dim() :].shape)
        # print("v[0:10] = ", v.vector().get_local()[: self._V.dim()][0:10])
        # print("p[0:10] = ", p.vector().get_local()[self._V.dim() :][0:10])

        return v.vector().get_local()[: self._V.dim()], p.vector().get_local()[self._V.dim() :]

    def solve_dual_time_step(self, u_n_vector, z_n_vector):
        pass  # todo

    # Solve time trajectory
    def solve_primal(self, force_recompute=False):
        if not force_recompute:
            if self.load_solution():
                return

        # reset solution
        self.U.vector()[:] = 0.0
        # apply boundary conditions to initial condition
        for _bc in self.bc:
            _bc.apply(self.U.vector())
        print("U init = ", self.U.vector().get_local()[:10])
        v, p = self.U.split()
        self.Y["velocity"][:, 0] = np.zeros_like(v.vector().get_local()[: self.dofs["velocity"]])
        self.Y["pressure"][:, 0] = np.zeros_like(p.vector().get_local()[self.dofs["velocity"] :])

        print(f"MAX PRESSURE: {np.max(self.Y['pressure'][:, 0])}")

        # create solve_time_{nt}_{T}.txt file
        #if exists, delete it
        
        file_name = f"results/solve_time_{self.T}_{self.dt}_{self.mesh_name}_{int(0.1/float(self.nu))}.txt"
        
        if os.path.exists(file_name):
            os.remove(file_name)
        else: 
            # create file
            os.mknod(file_name)
        time_file = open(file_name, "w")
        time_file.write("nt, t, solve_time\n")
        for i, t in enumerate(self.time_points[1:]):
            tic = time.time()
            n = i + 1
            print(f"\n\nt = {round(t,5)}, Re = {round(0.1/float(self.nu),5)}:\n==========")
            self.Y["velocity"][:, n], self.Y["pressure"][:, n] = self.solve_primal_time_step(
                self.Y["velocity"][:, n - 1], self.Y["pressure"][:, n - 1]
            )
            solve_time = time.time() - tic
            # write solve_time to file by extending 
            time_file = open(file_name, "a")
            time_file.write(f"{n}, {t:.5f}, {solve_time:.5f}\n")
            
        self.save_solution()
        self.save_vtk()

    def save_vtk(self):
        folder = f"paraview/{self.dt}_{self.T}_{self.theta}_{float(self.nu)}/FOM"

        if not os.path.exists(folder):
            os.makedirs(folder)

        xdmffile_u = XDMFFile(f"{folder}/velocity.xdmf")


        for i, t in list(enumerate(self.time_points))[::5]:
            # DEBUG HF: remove lifting to see resiudal
            sol_velocity = self.Y["velocity"][:, i]
            sol_pressure = self.Y["pressure"][:, i]

            v, p = self.U_n.split()

            self.U_n.vector().set_local(
                np.concatenate(
                    (
                        sol_velocity,
                        sol_pressure,
                    )
                )
            )

            # subplot for velocuty and pressure
            plt.figure(figsize=(8, 5.5))
            plt.subplot(2, 1, 1)
            c = plot(sqrt(dot(v, v)), title=f"Velocity @ t={t:.2}")
            plt.colorbar(c, orientation="horizontal")
            plt.subplot(2, 1, 2)
            c = plot(p, title=f"Pressure @ t={t:.2}")
            plt.colorbar(c, orientation="horizontal")

            plt.savefig(f"plots/solution_{str(i)}.png")

            # plt.show()


        # for i, t in enumerate(self.time_points):
            # vtk_velocity = File(f"{folder}/velocity_{str(i)}.pvd")
            # vtk_pressure = File(f"{folder}/pressure_{str(i)}.pvd")

            # # self.U_n.vector().set_local(solution)
            # v, p = self.U_n.split()
            # # v.vector().vec()[:self.dofs["velocity"]] = self.Y["velocity"][:, i]

            # v.vector().set_local(self.Y["velocity"][:, i])
            # p.vector().set_local(self.Y["pressure"][:, i])

            # #
            # xdmffile_u.write(v, t)

            # # c = plot(sqrt(dot(v, v)), title="Velocity")
            # # plt.colorbar(c, orientation="horizontal")
            # # plt.show()

            # v.rename("velocity", "solution")
            # p.rename("pressure", "solution")
            # vtk_velocity.write(v)

            # # v.rename('velocity', 'solution')
            # # p.rename('pressure', 'solution')

            # # vtk_velocity << v
            # # vtk_pressure << p

            # # vtkfile.write(v, "velocity")
            # # # vtkfile.write(p, "pressure")
