""" ------------ IMPLEMENTATION of iROM ------------
NOTE: DO NOT FORGET THIS FCKING .copy() IF COPYING A MATRIX. ELSE EVERYTHING IS BULLSHIT!!!
"""
import math
import os
import time

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fenics import *
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from petsc4py import PETSc
from scipy.stats import norm


class iROMGreedy:
    # constructor
    def __init__(
        self,
        fom,
        parameter=None,
        time_steps=1,
        REL_ERROR_TOL=1e-2,
        MAX_ITERATIONS=100,
        dt=0.001,
        time_points=None,
        COST_FCT_TRESHOLD=0.0,
        MEYER_MATTHIES_EXEC=False,
        MEYER_MATTHIES_ITERATIONS = 0,
    ):
        self.REL_ERROR_TOL = REL_ERROR_TOL
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.time_steps = time_steps
        self.COST_FCT_TRESHOLD = COST_FCT_TRESHOLD
        self.MEYER_MATTHIES_EXEC = MEYER_MATTHIES_EXEC  
        self.MEYER_MATTHIES_ITERATIONS = MEYER_MATTHIES_ITERATIONS

        self.fom = fom
        self.dt = dt

        if time_points is None:
            raise ValueError("No time_points given")

        self.time_points = time_points

        if parameter is None:
            # value error
            raise ValueError("No parameter given")

        self.parameter = parameter

        # print(f"Parameter: {self.parameter}")

        self.matrix = {
            "primal": {"system": None, "rhs": None},
            "dual": {"system": None, "rhs": None},
            "estimate": {"system": None, "rhs": None},
        }

        self.LU = {
            "primal": {
                "lu": None,
                "pivot": None,
            },
            "dual": {
                "lu": None,
                "pivot": None,
            },
        }

        self.solution = {"primal": None, "dual": None}
        self.functional_values = np.zeros((self.time_steps - 1,))

        self.timings = {
            "iPOD": 0.0,
            "primal_ROM": 0.0,
            "dual_ROM": 0.0,
            "error_estimate": 0.0,
            "enrichment": 0.0,
            "run": 0.0,
        }

        iterations_infos = {
            "error": [],
            "POD_size": {
                "primal": [],
                "dual": [],
            },
            "functional": [],
        }

        self.iterations_infos = []

    def greedy_estimate(self):
        self.affine_decomposition(type="all")
        self.solve_primal()
        self.solve_dual()
        # self.error_estimate()
        max_relative_error = self.error_estimate()
        return max_relative_error


    def update_ROM(self, matrix, RHS, POD):
        self.matrix = matrix
        self.RHS = RHS
        self.POD = POD

    def iPOD(self, snapshot, type):
        execution_time = time.time()
        # type is either "primal" or "dual"

        self.POD[type]["bunch"] = np.hstack((self.POD[type]["bunch"], snapshot.reshape(-1, 1)))

        # add energy of new snapshot to total energy
        self.POD[type]["energy"] += np.dot((snapshot), (snapshot))

        # check bunch_matrix size to decide if to update POD
        if self.POD[type]["bunch"].shape[1] == self.POD[type]["bunch_size"]:
            # initialize POD with first bunch matrix
            if self.POD[type]["basis"].shape[1] == 0:
                self.POD[type]["basis"], self.POD[type]["sigs"], _ = scipy.linalg.svd(
                    self.POD[type]["bunch"], full_matrices=False
                )

                # compute the number of POD modes to be kept
                r = 0
                while (
                    np.dot(self.POD[type]["sigs"][0:r], self.POD[type]["sigs"][0:r])
                    <= self.POD[type]["energy"] * self.POD[type]["TOL_ENERGY"]
                ) and (r <= np.shape(self.POD[type]["sigs"])[0]):
                    r += 1

                self.POD[type]["sigs"] = self.POD[type]["sigs"][0:r]
                self.POD[type]["basis"] = self.POD[type]["basis"][:, 0:r]
            # update POD with  bunch matrix
            else:
                M = np.dot(self.POD[type]["basis"].T, self.POD[type]["bunch"])
                P = self.POD[type]["bunch"] - np.dot(self.POD[type]["basis"], M)

                Q_p, R_p = scipy.linalg.qr(P, mode="economic")
                Q_q = np.hstack((self.POD[type]["basis"], Q_p))

                S0 = np.vstack(
                    (
                        np.diag(self.POD[type]["sigs"]),
                        np.zeros((np.shape(R_p)[0], np.shape(self.POD[type]["sigs"])[0])),
                    )
                )
                MR_p = np.vstack((M, R_p))
                K = np.hstack((S0, MR_p))

                # check the orthogonality of Q_q heuristically
                if np.abs(np.inner(Q_q[:, 0], Q_q[:, -1])) >= 1e-12:
                    print(f"Reorthogonalization of {type} iPOD")
                    Q_q, R_q = scipy.linalg.qr(Q_q, mode="economic")
                    K = np.matmul(R_q, K)

                # inner SVD of K
                U_k, S_k, _ = scipy.linalg.svd(K, full_matrices=False)

                # compute the number of POD modes to be kept
                r = self.POD[type]["basis"].shape[1]

                while (
                    np.dot(S_k[0:r], S_k[0:r])
                    <= self.POD[type]["energy"] * self.POD[type]["TOL_ENERGY"]
                ) and (r < np.shape(S_k)[0]):
                    r += 1

                self.POD[type]["sigs"] = S_k[0:r]
                self.POD[type]["basis"] = np.matmul(Q_q, U_k[:, 0:r])

            # empty bunch matrix after update
            self.POD[type]["bunch"] = np.empty([self.fom.dofs["space"], 0])
        self.timings["iPOD"] += time.time() - execution_time

    def reduce_vector(self, vector, type):
        return np.dot(self.POD[type]["basis"].T, vector)

    def project_vector(self, vector, type):
        return np.dot(self.POD[type]["basis"], vector)

    def reduce_matrix(self, matrix, type):
        if type == "primal":
            pod_basis_left = self.POD["primal"]["basis"]
            pod_basis_right = self.POD["primal"]["basis"]
        elif type == "dual":
            pod_basis_left = self.POD["dual"]["basis"]
            pod_basis_right = self.POD["dual"]["basis"]
        elif type == "estimate":
            pod_basis_left = self.POD["dual"]["basis"]
            pod_basis_right = self.POD["primal"]["basis"]
        else:
            exit("reduce_matrix: type not recognized")

        # use PETSc for matrix multiplication, since it uses the sparse matrix and
        # the numpy alternative works with the dense matrix which is more
        # expensive
        basis_left = PETSc.Mat().createDense(pod_basis_left.shape, array=pod_basis_left)
        basis_left.transpose()
        basis_right = PETSc.Mat().createDense(pod_basis_right.shape, array=pod_basis_right)
        _matrix = as_backend_type(matrix).mat()
        return Matrix(PETScMatrix(basis_left.matMult(_matrix.matMult(basis_right)))).array()

    def update_rhs(self, type):
        self.RHS[type] = np.dot(self.POD[type]["basis"].T, self.fom.RHS)
        self.RHS["J_prime"][type] = np.dot(self.POD[type]["basis"].T, self.fom.J_prime_vec)

    def update_LES(self, type):
        self.LU[type]["lu"], self.LU[type]["pivot"] = scipy.linalg.lu_factor(
            self.matrix[type]["system"]
        )

    def init_POD(self):
        # primal POD
        time_points = self.time_points[:2]
        old_solution = self.fom.u_0.vector().get_local()
        for i, t in enumerate(time_points[1:]):
            n = i + 1
            print(f"Enrich with primal: {n} at time: {t}")
            solution = self.fom.solve_primal_time_step(old_solution, self.fom.RHS[:, n])
            self.iPOD(solution, type="primal")
            old_solution = solution

        # dual POD
        time_points = self.time_points[-1:]
        old_solution = np.zeros((self.fom.dofs["space"],))
        for i, t in reversed(list(enumerate(time_points[:]))):
            n = i
            print(f"Enrich with dual: {n} at time: {t}")
            solution = self.fom.solve_dual_time_step(old_solution)
            self.iPOD(solution, type="dual")
            old_solution = solution

    def affine_decomposition(self, type="all", recompute_reduced_matrices=False):
        # TL,DR; get affine decomposition as input and build the matrices

        if recompute_reduced_matrices:
            if type == "all":
                self.matrix["primal"]["mass"] = self.reduce_matrix(
                    self.fom.matrix["primal"]["mass"], type="primal"
                )
                self.matrix["primal"]["stiffness"] = [None] * self.parameter.shape[0]
                for i, parameter in enumerate(self.parameter):
                    self.matrix["primal"]["stiffness"][i] = self.reduce_matrix(
                        self.fom.matrix["primal"]["stiffness"][i], type="primal"
                    )

                # dual
                self.matrix["dual"]["mass"] = self.reduce_matrix(
                    self.fom.matrix["dual"]["mass"], type="dual"
                )
                self.matrix["dual"]["stiffness"] = [None] * self.parameter.shape[0]
                for i, parameter in enumerate(self.parameter):
                    self.matrix["dual"]["stiffness"][i] = self.reduce_matrix(
                        self.fom.matrix["dual"]["stiffness"][i], type="dual"
                    )

                # estimator
                self.matrix["estimate"]["mass"] = self.reduce_matrix(
                    self.fom.matrix["primal"]["mass"], type="estimate"
                )
                self.matrix["estimate"]["stiffness"] = [None] * self.parameter.shape[0]
                for i, parameter in enumerate(self.parameter):
                    self.matrix["estimate"]["stiffness"][i] = self.reduce_matrix(
                        self.fom.matrix["primal"]["stiffness"][i], type="estimate"
                    )

                # rhs
                self.update_rhs(type="primal")  # also dual RHS included
                self.update_rhs(type="dual")  # primal reduced with dual basis

            elif type == "primal":
                self.matrix["primal"]["mass"] = self.reduce_matrix(
                    self.fom.matrix["primal"]["mass"], type="primal"
                )
                self.matrix["primal"]["stiffness"] = [None] * self.parameter.shape[0]
                for i, parameter in enumerate(self.parameter):
                    self.matrix["primal"]["stiffness"][i] = self.reduce_matrix(
                        self.fom.matrix["primal"]["stiffness"][i], type="primal"
                    )

                # rhs
                self.update_rhs(type="primal")  # also dual RHS included

            elif type == "dual":
                # dual
                self.matrix["dual"]["mass"] = self.reduce_matrix(
                    self.fom.matrix["dual"]["mass"], type="dual"
                )
                self.matrix["dual"]["stiffness"] = [None] * self.parameter.shape[0]
                for i, parameter in enumerate(self.parameter):
                    self.matrix["dual"]["stiffness"][i] = self.reduce_matrix(
                        self.fom.matrix["dual"]["stiffness"][i], type="dual"
                    )
                # rhs
                self.update_rhs(type="dual")  # primal reduced with dual basis

            elif type == "estimate":
                # estimator
                self.matrix["estimate"]["mass"] = self.reduce_matrix(
                    self.fom.matrix["primal"]["mass"], type="estimate"
                )
                self.matrix["estimate"]["stiffness"] = [None] * self.parameter.shape[0]
                for i, parameter in enumerate(self.parameter):
                    self.matrix["estimate"]["stiffness"][i] = self.reduce_matrix(
                        self.fom.matrix["primal"]["stiffness"][i], type="estimate"
                    )
            else:
                raise ValueError("type not recognized")

        # 2. stuff that will be done in the iROM class
        if type == "all":
            # primal
            # matrices
            self.matrix["primal"]["system"] = self.matrix["primal"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["primal"]["system"] += (
                    parameter * self.dt * self.matrix["primal"]["stiffness"][i]
                )
            self.matrix["primal"]["rhs"] = self.matrix["primal"]["mass"].copy()
            # LES
            self.update_LES("primal")

            # dual
            # matrices
            self.matrix["dual"]["system"] = self.matrix["dual"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["dual"]["system"] += (
                    parameter * self.dt * self.matrix["dual"]["stiffness"][i]
                )
            self.matrix["dual"]["rhs"] = self.matrix["dual"]["mass"].copy()
            # LES
            self.update_LES("dual")

            # estimator
            self.matrix["estimate"]["system"] = self.matrix["estimate"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["estimate"]["system"] += (
                    parameter * self.dt * self.matrix["estimate"]["stiffness"][i]
                )
            self.matrix["estimate"]["rhs"] = self.matrix["estimate"]["mass"].copy()

        elif type == "primal":
            # primal
            # matrices
            self.matrix["primal"]["system"] = self.matrix["primal"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["primal"]["system"] += (
                    parameter * self.dt * self.matrix["primal"]["stiffness"][i]
                )
            self.matrix["primal"]["rhs"] = self.matrix["primal"]["mass"].copy()
            # LES
            self.update_LES("primal")

        elif type == "dual":
            # dual
            # matrices
            self.matrix["dual"]["system"] = self.matrix["dual"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["dual"]["system"] += (
                    parameter * self.dt * self.matrix["dual"]["stiffness"][i]
                )
            self.matrix["dual"]["rhs"] = self.matrix["dual"]["mass"].copy()
            # LES
            self.update_LES("dual")

        elif type == "estimate":
            # estimator
            self.matrix["estimate"]["system"] = self.matrix["estimate"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["estimate"]["system"] += (
                    parameter * self.dt * self.matrix["estimate"]["stiffness"][i]
                )
            self.matrix["estimate"]["rhs"] = self.matrix["estimate"]["mass"].copy()
        else:
            raise ValueError("type not recognized")

    def get_functional(self, solution):
        return self.dt * np.dot(self.RHS["J_prime"]["primal"], solution)

    def error_estimate(self):
        # this method is only for the validation loop
        # else look in corresponding parent_slab version
        self.errors = np.zeros((self.time_steps - 1,))
        relative_errors = np.zeros((self.time_steps - 1,))
        self.functional_values = np.zeros((self.time_steps - 1,))

        start = 1
        # calculate cost functional
        for i in range(0, self.time_steps - 1):  # 0 instead of start here for plotting the CF
            n = i + 1
            self.functional_values[i] = self.get_functional(self.solution["primal"][:, n])

        mean_cost_functional = np.mean(np.abs(self.functional_values))

        for i in range(start, self.time_steps - 1):
            n = i + 1  # to skip primal IC
            self.errors[i] = np.dot(
                self.solution["dual"][:, n],
                -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, n])
                + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, n - 1])
                + self.dt * self.RHS["dual"][:, n],
            )

            # check if cost functional is too small to use relative error
            # estimate -->  set error to zero
            if np.abs(self.functional_values[i]) < self.COST_FCT_TRESHOLD * mean_cost_functional:
                # if np.abs(self.functional_values[i]) < quantile:
                self.errors[i] = 0.0
                # print(f"Cost functional too small at time step {n} --> set error to zero")

            relative_errors[i] = self.errors[i] / (self.errors[i] + self.functional_values[i])

        max_relative_error = np.max(np.abs(relative_errors))
        # since we do not need error estimate for IC
        max_arg_relative_error = np.argmax(np.abs(relative_errors)) + 1

        return max_relative_error, max_arg_relative_error

    def solve_primal_time_step(self, time_step):
        b = np.dot(self.matrix["primal"]["rhs"], self.solution["primal"][:, time_step - 1])
        b += self.dt * self.RHS["primal"][:, time_step]

        self.solution["primal"][:, time_step] = np.linalg.solve(
            self.reduce_matrix(self.fom.matrix["primal"]["system"], type="primal"),
            b,
        )

    def solve_dual_time_step(self, time_step):
        # A = self.matrix["dual"]["system"]
        b = self.matrix["dual"]["rhs"].dot(self.solution["dual"][:, time_step + 1])
        b += self.dt * self.RHS["J_prime"]["dual"]

        self.solution["dual"][:, time_step] = scipy.linalg.lu_solve(
            (self.LU["dual"]["lu"], self.LU["dual"]["pivot"]), b
        )

        # self.solution["dual"][:, time_step] = np.linalg.solve(A,b)

    def solve_primal(self):
        self.solution["primal"] = np.zeros((self.POD["primal"]["basis"].shape[1], self.time_steps))

        # self.matrix["primal"]["mass"] = self.reduce_matrix(self.fom.matrix["primal"]["mass"], type="primal")

        self.solution["primal"][:, 0] = np.zeros((self.POD["primal"]["basis"].shape[1],))
        solution = self.solution["primal"][:, 0]
        for i, t in enumerate(self.time_points[1:]):
            n = i + 1
            b = self.matrix["primal"]["rhs"].dot(solution)
            b += self.dt * self.RHS["primal"][:, n]

            solution = scipy.linalg.lu_solve(
                (self.LU["primal"]["lu"], self.LU["primal"]["pivot"]), b
            )
            self.solution["primal"][:, n] = solution

    def solve_dual(self):
        self.solution["dual"] = np.zeros((self.POD["dual"]["basis"].shape[1], self.time_steps))

        self.solution["dual"][:, -1] = np.zeros((self.POD["dual"]["basis"].shape[1],))
        for i, t in reversed(list(enumerate(self.time_points[:-1]))):
            n = i
            # print(f"dual_step: {n} at time: {t}")
            self.solve_dual_time_step(n)
        # print(self.solution["dual"][:, 0])
        # print(self.solution["dual"][:, -1])

    def validate(self):
        self.affine_decomposition(type="all")
        self.solve_primal()
        self.solve_dual()
        self.error_estimate()

    def enrich_ROM(self, i_max):
        # COMMENT: Right now I have implemented without forward and backward of primal and dual
        #          with new solution to get a better last solution for dual

        execution_time = time.time()

        # NO if for first time_step since zeor IC is stored at solution[0, :]
        last_solution = self.project_vector(
            self.solution["primal"][:, i_max - 1],
            type="primal",
        )
        # enrich primal
        new_solution = self.fom.solve_primal_time_step(last_solution, self.fom.RHS[:, i_max])
        self.iPOD(new_solution, type="primal")
        self.affine_decomposition(type="primal", recompute_reduced_matrices=True)

        # enrich dual
        # check if dual[i_max] is already last solution if
        # true: use zeros
        # false: use dual[i_max+1]
        last_dual_solution = self.project_vector(self.solution["dual"][:, i_max + 1], type="dual")

        new_dual_solution = self.fom.solve_dual_time_step(last_dual_solution)

        self.iPOD(new_dual_solution, type="dual")
        self.affine_decomposition(type="dual", recompute_reduced_matrices=True)

        # update estimate components
        self.affine_decomposition(type="estimate", recompute_reduced_matrices=True)

        self.timings["enrichment"] += time.time() - execution_time

    def run_ROM(self, POD_given=True):
        execution_time = time.time()
        if not POD_given:
            self.init_POD()

        self.affine_decomposition(type="all")

        iteration = 1
        fom_solves = 0
        max_error_iteration = []

        while iteration <= self.MAX_ITERATIONS:
            print(f"====== Iteration: {iteration} ======")
            print(
                f"Bases: {self.POD['primal']['basis'].shape[1]} / {self.POD['dual']['basis'].shape[1]}"
            )
            # 1. Solve primal ROM
            self.solve_primal()

            # 2. Solve dual ROM
            self.solve_dual()

            # 3. Evaluate DWR error estimator (RELATIVE ERROR)
            max_error, max_index = self.error_estimate()
            max_error_iteration.append(max_error)

            # 4. If relative error is too large, then solve primal and dual FOM on
            # time step with largest error
            if max_error <= self.REL_ERROR_TOL:
                print(f" DONE: Largest error @ (i={max_index}): {max_error:.5})")
                break
            else:
                print(f"Enrich for largest error @ (i={max_index}: {max_error:.5})")
                self.enrich_ROM(max_index)
                fom_solves += 2

            iteration += 1


        # =============================================================
        # MEYER MATTHIES ROM BASIS REDUCTION
        # =============================================================
        
        # iteration < self.MAX_ITERATIONS and iteration > 100:
        if iteration > self.MEYER_MATTHIES_ITERATIONS and self.MEYER_MATTHIES_EXEC: # LOWER THAN ON FULL_ENRICHMENT SINCE LESSER ITERATIONS PER ENRICHEMNT            #self.MAX_ITERATIONS:
            if iteration > self.MAX_ITERATIONS:
                print(f"WARNING: Maximum number of iterations reached!")
                self.solve_primal()
                self.solve_dual()

            # TODO: DEBUG: Add code to evaluate impact of the POD modes on cost
            # functional, c.v. [[meyer2003]]
            start = 1
            r_m = np.zeros((self.POD["dual"]["basis"].shape[1],))

            for i in range(start, self.time_steps - 1):
                n = i + 1  # to skip primal IC
                r_m_i = np.dot(
                    np.diag(self.solution["dual"][:, n]),
                    -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, n])
                    + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, n - 1])
                    + self.dt * self.RHS["dual"][:, n],
                )

                r_m += r_m_i

            pod_modes_influence = np.abs(
                self.dt
                * np.dot(self.POD["primal"]["basis"].T, np.dot(self.POD["dual"]["basis"], r_m))
            )

            PLOTTEN = False
            if PLOTTEN:
                # plot pod_modes_influence
                plt.figure()
                plt.plot(pod_modes_influence)
                plt.title("POD modes influence")
                plt.xlabel("POD mode")
                plt.ylabel("Influence")
                plt.yscale("log")
                plt.grid()
                plt.show()

                plt.figure()
                plt.plot(100 * pod_modes_influence / np.max(pod_modes_influence))
                plt.title("POD modes relative influence [%]")
                plt.xlabel("POD mode")
                plt.ylabel("Influence")
                plt.yscale("log")
                plt.grid()
                plt.show()
            # self.validate()

            # neglect all POD modes with less than TOL% influence
            TOL = 0.005  # 0.1%
            relative_influence = pod_modes_influence / np.max(pod_modes_influence)

            # find all indices where relative influence is larger than TOL
            indices = np.where(relative_influence >= TOL)[0]

            # find the order of the POD modes wrt impact (largest to smallest)
            order = np.argsort(relative_influence)[::-1]

            print("order: ", order)
            print("length order: ", len(order))

            # TODO: DEBUG: WRITE A Loop that determines the smallest number of
            # POD modes that are needed to reach the self.REL_ERROR_TOL

            POD_tmp = self.POD["primal"]["basis"].copy()
            sigs_tmp = self.POD["primal"]["sigs"].copy()

            primal_system_tmp = self.matrix["primal"]["system"].copy()
            primal_rhs_tmp = self.matrix["primal"]["rhs"].copy()

            estimate_system_tmp = self.matrix["estimate"]["system"].copy()
            estimate_rhs_tmp = self.matrix["estimate"]["rhs"].copy()

            RHS_primal_tmp = self.RHS["primal"].copy()

            max_error_new = 0
            MEYER_MATTHIES_TOL = self.REL_ERROR_TOL
            MEYER_MATTHIES_USED = False
            print(f"MEYER_MATTHIES_TOL: {MEYER_MATTHIES_TOL}")
            if iteration > self.MAX_ITERATIONS:
                MEYER_MATTHIES_TOL = max_error * 0.01  # 1.05
            for hh in reversed(range(1, len(order))):
                print(f"test hh: {hh}")
                # use the first i-th most impactful POD modes
                self.POD["primal"]["basis"] = POD_tmp[:, order[:hh]].copy()
                self.POD["primal"]["sigs"] = sigs_tmp[order[:hh]].copy()

                # permutate the primal system and rhs according to the order of the POD modes
                # primal
                self.matrix["primal"]["system"] = primal_system_tmp[order, :][:, order][:hh, :][
                    :, :hh
                ].copy()
                self.matrix["primal"]["rhs"] = primal_rhs_tmp[order, :][:, order][:hh, :][
                    :, :hh
                ].copy()
                self.RHS["primal"] = RHS_primal_tmp[order, :][:hh, :].copy()
                self.update_LES("primal")
                # 1. Solve primal
                self.solve_primal()

                # estiamte
                self.matrix["estimate"]["system"] = estimate_system_tmp[:, order][:, :hh].copy()
                self.matrix["estimate"]["rhs"] = estimate_rhs_tmp[:, order][:, :hh].copy()

                # 2. Evaluate DWR error estimator (RELATIVE ERROR)
                # ATTENTION: We use here the best cost functional we have for
                # evaluation. Thus with all POD modes computed instead of inly
                # of these in order
                start = 1
                mean_cost_functional = np.mean(np.abs(self.functional_values))
                relative_errors = np.zeros((self.time_steps - 1,))
                for i in range(start, self.time_steps - 1):
                    n = i + 1  # to skip primal IC
                    self.errors[i] = np.dot(
                        self.solution["dual"][:, n],
                        -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, n])
                        + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, n - 1])
                        + self.dt * self.RHS["dual"][:, n],
                    )

                    # check if cost functional is too small to use relative
                    # error estimate -->  set error to zero
                    if (
                        np.abs(self.functional_values[i])
                        < self.COST_FCT_TRESHOLD * mean_cost_functional
                    ):
                        # if np.abs(self.functional_values[i]) < quantile:
                        self.errors[i] = 0.0
                        # print(f"Cost functional too small at time step {n} --> set error to zero")

                    relative_errors[i] = self.errors[i] / (
                        self.errors[i] + self.functional_values[i]
                    )

                last_max_error_new = max_error_new
                max_error_new = np.max(np.abs(relative_errors))
                max_index = np.argmax(np.abs(relative_errors)) + 1

                print(order[:hh])
                # print("new max error: ", max_error_new)
                # print("old max error: ", max_error)

                if max_error_new >= MEYER_MATTHIES_TOL:
                    MEYER_MATTHIES_USED = True
                    self.POD["primal"]["basis"] = POD_tmp[:, order[: hh + 1]].copy()
                    self.POD["primal"]["sigs"] = sigs_tmp[order[: hh + 1]].copy()
                    print(f"POD modes: {self.POD['primal']['basis'].shape[1]}")
                    # update affine decomposition
                    self.affine_decomposition(type="primal", recompute_reduced_matrices=True)
                    self.affine_decomposition(type="estimate", recompute_reduced_matrices=True)

                    # 1. Solve primal (just needed for plotting)
                    self.solve_primal()
                    print(
                        f"Success: {hh+1} of {len(order)} POD modes are enough to reach the error tolerance: {MEYER_MATTHIES_TOL}"
                    )
                    print(f"Previous max error: {max_error}")
                    print(f"Max error: {last_max_error_new}")
                    break
                
            # if no POD mode can be negelcted
            if not MEYER_MATTHIES_USED:
                print(f"NO POD mode can be neglected to reach the error tolerance: {MEYER_MATTHIES_TOL}")
                # assigne tmp back to original
                self.POD["primal"]["basis"] = POD_tmp.copy()
                self.POD["primal"]["sigs"] = sigs_tmp.copy()

                # permutate the primal system and rhs according to the order of the POD modes
                # primal
                self.matrix["primal"]["system"] = primal_system_tmp.copy()
                self.matrix["primal"]["rhs"] = primal_rhs_tmp.copy()
                self.RHS["primal"] = RHS_primal_tmp.copy()
                self.update_LES("primal")
                # 1. Solve primal
                self.solve_primal()

                # estiamte
                self.matrix["estimate"]["system"] = estimate_system_tmp.copy()
                self.matrix["estimate"]["rhs"] = estimate_rhs_tmp.copy()

            
            new_functional_values = np.zeros((self.time_steps - 1,))
            for i in range(0, self.time_steps - 1):
                n = i + 1  # to skip primal IC
                new_functional_values[i] = self.get_functional(self.solution["primal"][:, n])

            PLOTTEN = False
            if PLOTTEN:
                # plot functional values
                plt.figure()
                plt.plot(self.functional_values, label="old")
                plt.plot(new_functional_values, label="new")
                plt.legend()
                plt.title("Cost functional values")
                plt.xlabel("time step")
                plt.ylabel("Cost functional value")
                plt.grid()
                plt.show()

        self.timings["run"] += time.time() - execution_time
        print(f"Total FOM solves: {fom_solves}")

        return self.POD, fom_solves
