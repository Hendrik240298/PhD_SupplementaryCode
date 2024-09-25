import math
import os
import random
import re
import time
from multiprocessing import Pool

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fenics import *
from petsc4py import PETSc


class RHSExpression(UserExpression):
    _t = 0.0

    def parameters(self, radius_source=0.125, radius_trajectory=0.25):
        self.radius_source = radius_source
        self.radius_trajectory = radius_trajectory
        if self.radius_trajectory > 0.5:
            raise ValueError("Trajectory radius must be smaller than 0.5")

    def set_time(self, t):
        self._t = t

    def eval_cell(self, value, x, ufc_cell):
        if (x[0] - 0.5 - self.radius_trajectory * np.cos(2.0 * np.pi * self._t)) ** 2 + (
            x[1] - 0.5 - self.radius_trajectory * np.sin(2.0 * np.pi * self._t)
        ) ** 2 < self.radius_source**2:
            # np.sin(4.0 * np.pi * self._t)
            value[0] = 0.1 + np.abs(np.sin(4.0 * np.pi * self._t))
        else:
            value[0] = 0.0

    def value_shape(self):
        return ()  # scalar function


class HeatCoefficientExpression(UserExpression):
    def eval(self, value, x):
        if (x[0] >= self.square_x[0] and x[0] <= self.square_x[1]) and (
            x[1] >= self.square_y[0] and x[1] <= self.square_y[1]
        ):
            value[0] = self.value
        else:
            value[0] = 0.0

    def value_shape(self):
        return ()

    def set_parameter(self, square_x=[0, 1], square_y=[0, 1], value=1.0):
        self.square_x = square_x
        self.square_y = square_y
        self.value = value

    def print_parameter(self):
        print(f"Square x: {self.square_x}")
        print(f"Square y: {self.square_y}")
        print(f"Value: {self.value}")


class FOM:
    # constructor
    def __init__(self, nx, ny, t, T, dt, parameter=None, save_directory="results/"):
        self.nx = nx
        self.ny = ny
        self.t = t
        self.T = T
        self.dt = dt

        if parameter is None:
            # value error
            raise ValueError("No parameter given")

        self.parameter = parameter

        self.time_points = np.arange(self.t, self.T + self.dt, self.dt)
        print(f"FIRST/LATEST TIME POINT:    {self.time_points[0]}/{self.time_points[-1]}")
        print(f"NUMBER OF TIME POINTS:      {self.time_points.shape[0]}")
        mesh = UnitSquareMesh(nx, ny)

        # Define variational problem
        self.V = FunctionSpace(mesh, "P", 1)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.dofs = {
            "space": self.V.dim(),
            "time": self.time_points.shape[0],
        }

        # define initial conditions
        u_0 = Constant(0.0)
        # u_n: solution from last time step
        self.u_0 = interpolate(u_0, self.V)

        # define snapshot matrix
        self.Y = np.zeros((self.dofs["space"], self.dofs["time"]))

        # define functional values
        self.functional_values = np.zeros((self.dofs["time"] - 1,))

        # define matrices
        self.matrix = {
            "primal": {"system": PETScMatrix(), "rhs": PETScMatrix()},
            "dual": {"system": PETScMatrix(), "rhs": PETScMatrix()},
        }

        self.RHS = np.zeros((self.dofs["space"], self.dofs["time"]))

        # IO data
        self.SAVE_DIR = save_directory
        # create dir if not exist (even when subfolders need to be created)
        os.makedirs(self.SAVE_DIR, exist_ok=True)

    def save_solution(self):
        pattern = r"solution_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.nx, self.ny, self.dt, self.T])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(file, Y=self.Y, parameters=parameters, compression=True)
                print(f"Overwrite {file}")
                return

        file_name = self.SAVE_DIR + "/solution_" + str(len(files)).zfill(6) + ".npz"
        np.savez(file_name, Y=self.Y, parameters=parameters, compression=True)
        print(f"Saved as {file_name}")

    def load_solution(self):
        pattern = r"solution_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.nx, self.ny, self.dt, self.T])

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.Y = tmp["Y"]
                print(f"Loaded {file}")
                return True
        return False
    
    def save_RHS(self):
        print("Save RHS...")
        pattern = r"RHS_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.nx, self.ny, self.dt, self.T])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    RHS=self.RHS,
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return
        file_name = self.SAVE_DIR + "RHS_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            RHS=self.RHS,
            parameters=parameters,
            compression=True,
        )

        print(f"Saved as {file_name}")

    def load_RHS(self):
        pattern = r"RHS_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.nx, self.ny, self.dt, self.T])

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.RHS = tmp["RHS"]
                print(f"Loaded {file}")
                return True
        return False

    # assemble system matrices and rhs

    def assemble_system(self, force_recompute=False):
        # boundary condition
        self.bc = DirichletBC(self.V, Constant(0.0), lambda _, on_boundary: on_boundary)

        # mass matrix
        M = assemble(self.u * self.v * dx)

        # squares are defined as [x_left_down, y_left_down, x_right_up, y_right_up]
        # ATTENTION: squares must not overlap!!
        EPS = 1.0e-10
        self.squares = [
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.5 + 1.0e-10, 1.0, 1.0],
        ]
        # 4 squares
        """
        | 4 | 3 |
        ---------
        | 1 | 2 |
        """
        self.squares = [
            [0.0, 0.0, 0.5, 0.5],
            [0.5 + EPS, 0.0, 1.0, 0.5],
            [0.5 + EPS, 0.5 + EPS, 1.0, 1.0],
            [0.0, 0.5 + EPS, 0.5, 1.0],
        ]

        # 16 squares
        '''
        reorder to:

        """
        | 1  | 2  | 5  | 6  |
        ---------------------
        | 4  | 3  | 8  | 7  |
        ---------------------
        | 9  | 10 | 13 | 14  |
        ---------------------
        | 12 | 11 | 16  | 15 |
        """
        '''
        print("---------------------")
        print("| 1  | 2  | 5  | 6  |")
        print("| 4  | 3  | 8  | 7  |")
        print("| 9  | 10 | 13 | 14 |")
        print("| 12 | 11 | 16 | 15 |")

        self.squares = [
            [0.0, 0.75 + EPS, 0.25, 1.0],  # 1
            [0.25 + EPS, 0.75 + EPS, 0.5, 1.0],  # 2
            [0.25 + EPS, 0.5 + EPS, 0.5, 0.75],  # 3
            [0.0, 0.5 + EPS, 0.25, 0.75],  # 4
            [0.5 + EPS, 0.75 + EPS, 0.75, 1.0],  # 5
            [0.75 + EPS, 0.75 + EPS, 1.0, 1.0],  # 6
            [0.75 + EPS, 0.5 + EPS, 1.0, 0.75],  # 7
            [0.5 + EPS, 0.5 + EPS, 0.75, 0.75],  # 8
            [0.0, 0.25 + EPS, 0.25, 0.5],  # 9
            [0.25 + EPS, 0.25 + EPS, 0.5, 0.5],  # 10
            [0.25 + EPS, 0.0, 0.5, 0.25],  # 11
            [0.0, 0.0, 0.25, 0.25],  # 12
            [0.5 + EPS, 0.25 + EPS, 0.75, 0.5],  # 13
            [0.75 + EPS, 0.25 + EPS, 1.0, 0.5],  # 14
            [0.75 + EPS, 0.0, 1.0, 0.25],  # 15
            [0.5 + EPS, 0.0, 0.75, 0.25],  # 16
        ]

        # mus = np.random.uniform(0.01, 100.0, len(self.squares))

        print(f"mus: {self.parameter}")

        print(f"len square: {len(self.squares)}")

        # init empty Laplace matrix list
        K_list = []
        # fill Laplace matrices
        for i, square in enumerate(self.squares):
            indicator_fct = HeatCoefficientExpression(degree=1)
            indicator_fct.set_parameter(
                square_x=[square[0], square[2]], square_y=[square[1], square[3]], value=1.0
            )

            # indicator_fct.print_parameter()

            # # test by plotting
            # indicator_fct_interpolated = interpolate(indicator_fct, self.V)
            # plot(indicator_fct_interpolated, title="Parameter")
            # plt.show()

            # Laplace matrix
            K_list.append(assemble(indicator_fct * dot(grad(self.u), grad(self.v)) * dx))

        # sum up Laplace matrices
        K = self.parameter[0] * K_list[0]
        for i, mu in enumerate(self.parameter[1:]):
            K += mu * K_list[i + 1]

        # check if squares are disjunct and span the whole domain

        K_sum_uniform = 1.0 * K_list[0]
        for i, mu in enumerate(self.parameter[1:]):
            K_sum_uniform += 1.0 * K_list[i + 1]
            # check if K_list[i+1] is symmetric
            if not np.allclose(K_list[i + 1].array(), K_list[i + 1].array().T):
                print(f"K_list[{i+1}] is not symmetric.")

        K_uniform = assemble(dot(grad(self.u), grad(self.v)) * dx)
        # check if K == K_uniform
        if not np.allclose(K_sum_uniform.array(), K_uniform.array()):
            # throw assert
            raise AssertionError(f"Squares are not disjunct or span the whole domain.")

        # Derivative of the Cost functional J= \int\int u dx dt --> J' = \int \int \psi dx dt
        # Dual Problem: (M + k*K) z^n = M z^{n-1} + dt * J'
        self.J_prime = assemble(self.v * dx)
        self.J_prime_vec = np.array(self.J_prime)

        # system matrix
        self.matrix["primal"]["system"] = M + self.dt * K
        # right hand side matrix
        self.matrix["primal"]["rhs"] = M

        # dual sytsem matrix
        self.matrix["dual"]["system"] = self.matrix["primal"]["system"].copy()
        # dual right hand side matrix
        self.matrix["dual"]["rhs"] = self.matrix["primal"]["rhs"].copy()

        # matrices for affine decomposition for iROM
        # primal
        self.matrix["primal"]["mass"] = M
        self.matrix["primal"]["stiffness"] = K_list
        # dual
        self.matrix["dual"]["mass"] = M
        self.matrix["dual"]["stiffness"] = K_list

        if (not self.load_RHS()) or force_recompute:
            rhs_func = RHSExpression()
            t = 0.0

            rhs_func.set_time(t)
            rhs_func.parameters(radius_source=0.125, radius_trajectory=0.25)
            self.RHS[:, 0] = np.array(assemble(rhs_func * self.v * dx))

            print("Precompute RHS...")
            start_execution = time.time()
            # precompute the right hand side

            for i, t in enumerate(self.time_points[1:]):
                n = i + 1
                rhs_func.set_time(t)
                self.RHS[:, n] = np.array(assemble(rhs_func * self.v * dx))
            print(f"Done [{time.time()-start_execution:.2f} s]")
            self.save_RHS()

    # Solve one time_step

    def solve_primal_time_step(self, u_n_vector, rhs_vector):
        A = self.matrix["primal"]["system"]

        # reinit solution vectors
        u = Function(self.V)
        u_n = Function(self.V)
        u_n.vector().set_local(u_n_vector[:])

        # Compute RHS
        b = u_n.vector().copy()
        self.matrix["primal"]["rhs"].mult(u_n.vector(), b)

        # Add force terms
        b[:] += self.dt * rhs_vector  # RHS[:, n]

        # Apply boundary conditions
        self.bc.apply(A, b)

        # Solve linear system
        solve(A, u.vector(), b)
        return u.vector().get_local()

    def solve_dual_time_step(self, z_n_vector):
        A = self.matrix["dual"]["system"]

        # reinit solution vectors
        z_n = Function(self.V)  # dual solution from last (later) time step
        z = Function(self.V)  # new solution

        # write data to functions
        z_n.vector().set_local(z_n_vector[:])

        # compute dual RHS from primal solution and previous dual solution
        b = z_n.vector().copy()
        self.matrix["dual"]["rhs"].mult(z_n.vector(), b)
        b[:] += self.dt * self.J_prime

        # Apply boundary conditions
        self.bc.apply(A, b)

        # Solve linear system
        solve(A, z.vector(), b)
        return z.vector().get_local()

    def get_functional(self, u_n_vector):
        return self.dt * np.dot(self.J_prime_vec, u_n_vector)

    # Solve time trajectory

    def solve_primal(self, force_recompute=False):
        # u_n: solution from last time step
        
        if self.load_functional_values() and (not force_recompute):
            return
        
        u_n = self.u_0

        u = Function(self.V)

        # initial condition
        self.Y[:, 0] = u_n.vector().get_local()

        print("Solving FOM...")
        start_execution = time.time()

        # start time

        for i, t in enumerate(self.time_points[1:]):
            n = i + 1
            # Store solution in snapshot matrix
            self.Y[:, n] = self.solve_primal_time_step(self.Y[:, n - 1], self.RHS[:, n])

            # Compute functional value
            self.functional_values[n - 1] = self.get_functional(self.Y[:, n])

        execution_time_FOM = time.time() - start_execution
        print(f"Done [{execution_time_FOM:.2f} s]")
        print(f"J(u_h):            {np.sum(self.functional_values)}")

        # save self.functional_values in npz
        pattern = r"fom_functional_values_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        discretization_parameters = np.array([self.nx, self.ny, self.dt, self.T])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(discretization_parameters, tmp["discretization_parameters"]):
                if np.array_equal(self.parameter, tmp["parameter"]):
                    np.savez(
                        file,
                        functional_values=self.functional_values,
                        discretization_parameters=discretization_parameters,
                        parameter=self.parameter,
                        compression=True,
                    )
                    print(f"Overwrite {file}")
                    return

        file_name = self.SAVE_DIR + "/fom_functional_values_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            functional_values=self.functional_values,
            discretization_parameters=discretization_parameters,
            parameter=self.parameter,
            compression=True,
        )
        print(f"Saved functional_values as {file_name}")


    def load_functional_values(self):
        pattern = r"fom_functional_values_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]
        
        discretization_parameters = np.array([self.nx, self.ny, self.dt, self.T])
        parameter = self.parameter
        
        for file in files:
            tmp = np.load(file)
            if np.array_equal(discretization_parameters, tmp["discretization_parameters"]):
                if np.array_equal(parameter, tmp["parameter"]):
                    self.functional_values = tmp["functional_values"]
                    print(f"Loaded {file}")
                    return True
        return False
