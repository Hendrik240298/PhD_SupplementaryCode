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


class iROM:
    # constructor
    def __init__(
        self,
        fom,
        REL_ERROR_TOL=1e-2,
        MAX_ITERATIONS=100,
        PARENT_SLAB_SIZE=1,
        TOTAL_ENERGY={"primal": 1, "dual": 1},
        parameter=None,
    ):
        self.fom = fom
        self.REL_ERROR_TOL = REL_ERROR_TOL
        self.MAX_ITERATIONS = MAX_ITERATIONS
        PARENT_SLAB_SIZE = np.max([1, np.min([PARENT_SLAB_SIZE, int((self.fom.dofs["time"] - 1))])])

        if parameter is None:
            # value error
            raise ValueError("No parameter given")

        self.parameter = parameter

        # print(f"Parameter: {self.parameter}")

        # legacy non affine code
        # added matrices for estimate
        # self.fom.matrix.update(
        #     {
        #         "estimate": {
        #             "system": self.fom.matrix["primal"]["system"],
        #             "rhs": self.fom.matrix["primal"]["rhs"],
        #         }
        #     }
        # )

        self.POD = {
            "primal": {
                "basis": np.empty((self.fom.dofs["space"], 0)),
                "sigs": None,
                "energy": 0.0,
                "bunch": np.empty((self.fom.dofs["space"], 0)),
                "bunch_size": 1,
                "TOL_ENERGY": TOTAL_ENERGY["primal"],
            },
            "dual": {
                "basis": np.empty((self.fom.dofs["space"], 0)),
                "sigs": None,
                "energy": 0.0,
                "bunch": np.empty((self.fom.dofs["space"], 0)),
                "bunch_size": 1,
                "TOL_ENERGY": TOTAL_ENERGY["dual"],
            },
        }

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

        self.RHS = {
            "primal": None,
            "dual": None,
            "J_prime": {
                "primal": None,
                "dual": None,
            },
        }

        self.solution = {"primal": None, "dual": None}
        self.functional_values = np.zeros((self.fom.dofs["time"] - 1,))

        # initialize parent_slabs
        self.parent_slabs = []

        index_start = 1
        while index_start + PARENT_SLAB_SIZE <= self.fom.dofs["time"]:
            self.parent_slabs.append(
                {
                    "start": index_start,
                    "end": index_start + PARENT_SLAB_SIZE,
                    "steps": PARENT_SLAB_SIZE,
                    "solution": {
                        "primal": None,
                        "dual": None,
                    },
                    "initial": None,
                    "functional": np.zeros((PARENT_SLAB_SIZE,)),
                }
            )
            index_start += PARENT_SLAB_SIZE

        if self.parent_slabs[-1]["end"] < self.fom.dofs["time"]:
            self.parent_slabs.append(
                {
                    "start": self.parent_slabs[-1]["end"],
                    "end": self.fom.dofs["time"] - 1,
                    "steps": self.fom.dofs["time"] - 1 - self.parent_slabs[-1]["end"],
                    "solution": {
                        "primal": None,
                        "dual": None,
                    },
                    "initial": None,
                    "functional": np.zeros((self.fom.dofs["time"] - self.parent_slabs[-1]["end"],)),
                }
            )

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

        self.iterations_infos = [iterations_infos.copy() for _ in range(len(self.parent_slabs))]

        print(f"NUMBER OF PARENT SLABS: {len(self.parent_slabs)}")

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
                if np.abs(np.inner(Q_q[:, 0], Q_q[:, -1])) >= 1e-10:
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

    # NOTE: this is legacy non affine code, thus not used (errors mean there are modifications missing)
    # def update_matrices(self, type):
    #     for key in self.matrix[type].keys():
    #         self.matrix[type][key] = self.reduce_matrix(self.fom.matrix[type][key], type)

    def update_rhs(self, type):
        self.RHS[type] = np.dot(self.POD[type]["basis"].T, self.fom.RHS)
        self.RHS["J_prime"][type] = np.dot(self.POD[type]["basis"].T, self.fom.J_prime_vec)

    def update_LES(self, type):
        self.LU[type]["lu"], self.LU[type]["pivot"] = scipy.linalg.lu_factor(
            self.matrix[type]["system"]
        )

    def init_POD(self):
        # primal POD
        time_points = self.fom.time_points[:2]
        old_solution = self.fom.u_0.vector().get_local()
        for i, t in enumerate(time_points[1:]):
            n = i + 1
            print(f"Enrich with primal: {n} at time: {t}")
            solution = self.fom.solve_primal_time_step(old_solution, self.fom.RHS[:, n])
            self.iPOD(solution, type="primal")
            old_solution = solution

        # dual POD
        time_points = self.fom.time_points[-1:]
        old_solution = np.zeros((self.fom.dofs["space"],))
        for i, t in reversed(list(enumerate(time_points[:]))):
            n = i
            print(f"Enrich with dual: {n} at time: {t}")
            solution = self.fom.solve_dual_time_step(old_solution)
            self.iPOD(solution, type="dual")
            old_solution = solution

    def affine_decomposition(self, type="all"):
        # TL,DR; get affine decomposition as input and build the matrices

        # 1. stuff that will be done later on in the Greedy class
        # matrices
        # primal
        self.matrix["primal"]["mass"] = self.reduce_matrix(
            self.fom.matrix["primal"]["mass"], type="primal"
        )
        self.matrix["primal"]["stiffness"] = [None] * len(self.parameter)
        for i in range(len(self.parameter)):
            self.matrix["primal"]["stiffness"][i] = self.reduce_matrix(
                self.fom.matrix["primal"]["stiffness"][i], type="primal"
            )

        # dual
        self.matrix["dual"]["mass"] = self.reduce_matrix(
            self.fom.matrix["dual"]["mass"], type="dual"
        )
        self.matrix["dual"]["stiffness"] = [None] * len(self.parameter)
        for i in range(len(self.parameter)):
            self.matrix["dual"]["stiffness"][i] = self.reduce_matrix(
                self.fom.matrix["dual"]["stiffness"][i], type="dual"
            )

        # estimator
        self.matrix["estimate"]["mass"] = self.reduce_matrix(
            self.fom.matrix["primal"]["mass"], type="estimate"
        )
        self.matrix["estimate"]["stiffness"] = [None] * len(self.parameter)
        for i in range(len(self.parameter)):
            self.matrix["estimate"]["stiffness"][i] = self.reduce_matrix(
                self.fom.matrix["primal"]["stiffness"][i], type="estimate"
            )

        # rhs
        self.update_rhs(type="primal")  # also dual RHS included
        self.update_rhs(type="dual")  # primal reduced with dual basis

        # 2. stuff that will be done in the iROM class
        if type == "all":
            # primal
            # matrices
            self.matrix["primal"]["system"] = self.matrix["primal"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["primal"]["system"] += (
                    parameter * self.fom.dt * self.matrix["primal"]["stiffness"][i]
                )
            self.matrix["primal"]["rhs"] = self.matrix["primal"]["mass"].copy()
            # LES
            self.update_LES("primal")

            # dual
            # matrices
            self.matrix["dual"]["system"] = self.matrix["dual"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["dual"]["system"] += (
                    parameter * self.fom.dt * self.matrix["dual"]["stiffness"][i]
                )
            self.matrix["dual"]["rhs"] = self.matrix["dual"]["mass"].copy()
            # LES
            self.update_LES("dual")

            # estimator
            self.matrix["estimate"]["system"] = self.matrix["estimate"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["estimate"]["system"] += (
                    parameter * self.fom.dt * self.matrix["estimate"]["stiffness"][i]
                )
            self.matrix["estimate"]["rhs"] = self.matrix["estimate"]["mass"].copy()

            # self.update_LES("primal")
            # self.update_LES("dual")

        elif type == "primal":
            # primal
            # matrices
            self.matrix["primal"]["system"] = self.matrix["primal"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["primal"]["system"] += (
                    parameter * self.fom.dt * self.matrix["primal"]["stiffness"][i]
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
                    parameter * self.fom.dt * self.matrix["dual"]["stiffness"][i]
                )
            self.matrix["dual"]["rhs"] = self.matrix["dual"]["mass"].copy()
            # LES
            self.update_LES("dual")

        elif type == "estimate":
            # estimator
            self.matrix["estimate"]["system"] = self.matrix["estimate"]["mass"].copy()
            for i, parameter in enumerate(self.parameter):
                self.matrix["estimate"]["system"] += (
                    parameter * self.fom.dt * self.matrix["estimate"]["stiffness"][i]
                )
            self.matrix["estimate"]["rhs"] = self.matrix["estimate"]["mass"].copy()

        else:
            raise ValueError("type not recognized")

    def get_functional(self, solution):
        return self.fom.dt * np.dot(self.RHS["J_prime"]["primal"], solution)

    def error_estimate(self):
        # this method is only for the validation loop
        # else look in corresponding parent_slab version
        self.errors = np.zeros((self.fom.dofs["time"] - 1,))
        relative_errors = np.zeros((self.fom.dofs["time"] - 1,))
        self.functional_values = np.zeros((self.fom.dofs["time"] - 1,))
        for i in range(1, self.fom.dofs["time"] - 1):
            self.errors[i - 1] = np.dot(
                self.solution["dual"][:, i - 1],
                -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, i])
                + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, i - 1])
                +
                # self.RHS["primal"][:, i]
                self.fom.dt * self.RHS["dual"][:, i],
            )
            # np.dot(
            # self.solution["dual"][:, i - 1],
            # -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, i])
            # + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, i - 1])
            # + self.RHS["dual"][:, i],
            # )
            self.functional_values[i - 1] = self.get_functional(self.solution["primal"][:, i])
            relative_errors[i - 1] = self.errors[i - 1] / (
                self.errors[i - 1] + self.functional_values[i - 1]
            )
        print(
            f" After validation largest relative error is: {np.max(np.abs(relative_errors))} at index: {np.argmax(np.abs(relative_errors))}"
        )

    def solve_primal_time_step(self, time_step):
        b = np.dot(self.matrix["primal"]["rhs"], self.solution["primal"][:, time_step - 1])
        b += self.fom.dt * self.RHS["primal"][:, time_step]

        self.solution["primal"][:, time_step] = np.linalg.solve(
            self.reduce_matrix(self.fom.matrix["primal"]["system"], type="primal"),
            b,
        )

        # self.solution["primal"][:, time_step] = scipy.linalg.lu_solve((self.LU["primal"]["lu"], self.LU["primal"]["pivot"]), b)

        # self.solution["primal"][:, time_step] = np.linalg.solve(
        #     self.matrix["primal"]["system"],
        #     np.dot(self.matrix["primal"]["rhs"], self.solution["primal"][:, time_step - 1])
        #     + self.RHS["primal"][:, time_step],
        # )

    def solve_dual_time_step(self, time_step):
        # A = self.matrix["dual"]["system"]
        b = self.matrix["dual"]["rhs"].dot(self.solution["dual"][:, time_step + 1])
        b += self.fom.dt * self.RHS["J_prime"]["dual"]

        self.solution["dual"][:, time_step] = scipy.linalg.lu_solve(
            (self.LU["dual"]["lu"], self.LU["dual"]["pivot"]), b
        )

        # self.solution["dual"][:, time_step] = np.linalg.solve(A,b)

    def solve_primal(self):
        self.solution["primal"] = np.zeros(
            (self.POD["primal"]["basis"].shape[1], self.fom.dofs["time"])
        )

        # self.matrix["primal"]["mass"] = self.reduce_matrix(self.fom.matrix["primal"]["mass"], type="primal")

        self.solution["primal"][:, 0] = np.zeros((self.POD["primal"]["basis"].shape[1],))
        solution = self.solution["primal"][:, 0]
        for i, t in enumerate(self.fom.time_points[1:]):
            n = i + 1
            # b = self.reduce_matrix(self.fom.matrix["primal"]["mass"], type="primal").dot(solution)
            b = self.matrix["primal"]["rhs"].dot(solution)
            # b += self.fom.dt * self.reduce_vector(self.fom.RHS[:, n], type="primal")
            b += self.fom.dt * self.RHS["primal"][:, n]
            # solution = scipy.linalg.solve(
            #     self.reduce_matrix(self.fom.matrix["primal"]["system"], type="primal"),
            #     b)

            solution = scipy.linalg.lu_solve(
                (self.LU["primal"]["lu"], self.LU["primal"]["pivot"]), b
            )
            self.solution["primal"][:, n] = solution

            # n = i + 1
            # self.solve_primal_time_step(n)

    def solve_dual(self):
        self.solution["dual"] = np.zeros(
            (self.POD["dual"]["basis"].shape[1], self.fom.dofs["time"])
        )

        self.solution["dual"][:, -1] = np.zeros((self.POD["dual"]["basis"].shape[1],))
        for i, t in reversed(list(enumerate(self.fom.time_points[:-1]))):
            n = i
            # print(f"dual_step: {n} at time: {t}")
            self.solve_dual_time_step(n)

    def validate(self):
        # DEBUG
        # self.solution["primal"] = np.zeros(
        #     (self.POD["primal"]["basis"].shape[1], self.fom.dofs["time"])
        # )

        # time_points = self.fom.time_points[:]
        # old_solution = self.fom.u_0.vector().get_local()
        # for i, t in enumerate(time_points[1:]):
        #     n = i + 1
        #     print(f"Enrich with primal: {n} at time: {t}")
        #     solution = self.fom.solve_primal_time_step(
        #         old_solution, self.fom.RHS[:, n]
        #     )
        #     self.solution["primal"][:, n] = self.reduce_vector(solution, type="primal")
        #     old_solution = solution

        # self.solution["primal"][:, 0] = np.zeros((self.POD["primal"]["basis"].shape[1],))

        # for i in range(self.solution["primal"].shape[1] - 1):
        #     n = i + 1
        #     solution = self.fom.solve_primal_time_step(
        #         self.solution["primal"][:, n-1], self.fom.RHS[:, n]
        #     )
        #     self.solution["primal"][:, n] = self.reduce_vector(solution, type="primal")

        self.affine_decomposition(type="all")
        self.solve_primal()
        self.solve_dual()
        self.error_estimate()

    def solve_primal_time_step_slab(self, time_step, old_solution):
        b = (
            np.dot(self.matrix["primal"]["rhs"], old_solution)
            + self.fom.dt * self.RHS["primal"][:, time_step]
        )
        return scipy.linalg.lu_solve((self.LU["primal"]["lu"], self.LU["primal"]["pivot"]), b)

        # return np.linalg.solve(
        #     self.matrix["primal"]["system"],
        #     np.dot(self.matrix["primal"]["rhs"], old_solution) + self.RHS["primal"][:, time_step],
        # )

    def solve_dual_time_step_slab(self, old_solution):
        # A = self.matrix["dual"]["system"]
        b = self.matrix["dual"]["rhs"].dot(old_solution)
        b += self.fom.dt * self.RHS["J_prime"]["dual"]

        return scipy.linalg.lu_solve((self.LU["dual"]["lu"], self.LU["dual"]["pivot"]), b)

        # return np.linalg.solve(A, b)

    def solve_primal_parent_slab(self, index_parent_slab):
        execution_time = time.time()
        self.parent_slabs[index_parent_slab]["solution"]["primal"] = np.zeros(
            (self.POD["primal"]["basis"].shape[1], self.parent_slabs[index_parent_slab]["steps"])
        )
        initial_condition = self.parent_slabs[index_parent_slab]["initial_condition"]

        for i in range(
            self.parent_slabs[index_parent_slab]["start"],
            self.parent_slabs[index_parent_slab]["end"],
        ):
            n = i - self.parent_slabs[index_parent_slab]["start"]
            self.parent_slabs[index_parent_slab]["solution"]["primal"][
                :, n
            ] = self.solve_primal_time_step_slab(i, initial_condition)
            initial_condition = self.parent_slabs[index_parent_slab]["solution"]["primal"][:, n]
        self.timings["primal_ROM"] += time.time() - execution_time

    def solve_dual_parent_slab(self, index_parent_slab):
        execution_time = time.time()
        self.parent_slabs[index_parent_slab]["solution"]["dual"] = np.zeros(
            (self.POD["dual"]["basis"].shape[1], self.parent_slabs[index_parent_slab]["steps"])
        )
        initial_condition = np.zeros((self.POD["dual"]["basis"].shape[1],))

        for i in reversed(
            range(
                self.parent_slabs[index_parent_slab]["start"],
                self.parent_slabs[index_parent_slab]["end"],  # -1 is DEBUG,
            )
        ):
            n = i - self.parent_slabs[index_parent_slab]["start"]
            self.parent_slabs[index_parent_slab]["solution"]["dual"][
                :, n
            ] = self.solve_dual_time_step_slab(initial_condition)
            initial_condition = self.parent_slabs[index_parent_slab]["solution"]["dual"][:, n]
        self.timings["dual_ROM"] += time.time() - execution_time

    def error_estimate_parent_slab(self, index_parent_slab):
        execution_time = time.time()
        errors = np.zeros((self.parent_slabs[index_parent_slab]["steps"],))
        relative_errors = np.zeros((self.parent_slabs[index_parent_slab]["steps"],))
        self.parent_slabs[index_parent_slab]["functional"] = np.zeros(
            (self.parent_slabs[index_parent_slab]["steps"],)
        )

        i = 0
        errors[i] = np.dot(
            self.parent_slabs[index_parent_slab]["solution"]["dual"][:, i],
            -np.dot(
                self.matrix["estimate"]["system"],
                self.parent_slabs[index_parent_slab]["solution"]["primal"][:, i],
            )
            + np.dot(
                self.matrix["estimate"]["rhs"],
                self.parent_slabs[index_parent_slab]["initial_condition"],
            )
            + self.fom.dt
            * self.RHS["dual"][:, self.parent_slabs[index_parent_slab]["start"] + i + 1],
        )

        for i in range(1, self.parent_slabs[index_parent_slab]["steps"]):
            errors[i] = np.dot(
                self.parent_slabs[index_parent_slab]["solution"]["dual"][:, i - 1],
                -np.dot(
                    self.matrix["estimate"]["system"],
                    self.parent_slabs[index_parent_slab]["solution"]["primal"][:, i],
                )
                + np.dot(
                    self.matrix["estimate"]["rhs"],
                    self.parent_slabs[index_parent_slab]["solution"]["primal"][:, i - 1],
                )
                + self.fom.dt
                * self.RHS["dual"][:, self.parent_slabs[index_parent_slab]["start"] + i],
            )

            self.parent_slabs[index_parent_slab]["functional"][i] = self.get_functional(
                self.parent_slabs[index_parent_slab]["solution"]["primal"][:, i]
            )

            # check if dividing by zero
            if np.abs(errors[i] + self.parent_slabs[index_parent_slab]["functional"][i]) < 1e-12:
                relative_errors[i] = errors[i] / 1e-12
            else:
                relative_errors[i] = errors[i] / (
                    errors[i] + self.parent_slabs[index_parent_slab]["functional"][i]
                )

        self.parent_slabs[index_parent_slab]["functional_total"] = np.sum(
            self.parent_slabs[index_parent_slab]["functional"]
        )
        slab_relative_error = np.abs(
            np.sum(errors)
            / (np.sum(errors) + self.parent_slabs[index_parent_slab]["functional_total"])
        )

        # # ORIGINAL
        i_max = np.argmax(np.abs(relative_errors))
        # print(f"First estim: {relative_errors[0:10]}")
        # print(f"i_max: {i_max}")
        error = {
            "relative": relative_errors,
            "absolute": errors,
            "max": np.abs(relative_errors[i_max]),
            "i_max": i_max,
            "slab_relative_error": slab_relative_error,
        }
        self.timings["error_estimate"] += time.time() - execution_time
        return error

    def enrich_parent_slab(self, index_parent_slab, i_max):
        # COMMENT: Right now I have implemented without forward and backward of primal and dual
        #          with new solution to get a better last solution for dual

        execution_time = time.time()
        projected_last_initial = self.project_vector(
            self.parent_slabs[index_parent_slab]["initial_condition"], type="primal"
        )

        # DEBUG WITH FULL IC
        projected_last_initial = self.DEBUG_FULL_IC

        if i_max == 0:
            # if we need to enrich at first step choose IC as last solution
            last_solution = projected_last_initial
        else:
            last_solution = self.project_vector(
                self.parent_slabs[index_parent_slab]["solution"]["primal"][:, i_max - 1],
                type="primal",
            )
        # enrich primal
        new_solution = self.fom.solve_primal_time_step(
            last_solution, self.fom.RHS[:, i_max + self.parent_slabs[index_parent_slab]["start"]]
        )
        self.iPOD(new_solution, type="primal")
        self.affine_decomposition(type="primal")

        # enrich dual
        # check if dual[i_max] is already last solution if
        # true: use zeros
        # false: use dual[i_max+1]
        if i_max == self.parent_slabs[index_parent_slab]["steps"] - 1:
            last_dual_solution = np.zeros((self.POD["dual"]["basis"].shape[1],))
        else:
            last_dual_solution = self.project_vector(
                self.parent_slabs[index_parent_slab]["solution"]["dual"][:, i_max + 1], type="dual"
            )
        new_dual_solution = self.fom.solve_dual_time_step(last_dual_solution)

        self.iPOD(new_dual_solution, type="dual")
        self.affine_decomposition(type="dual")

        # update estimate components
        self.affine_decomposition(type="estimate")

        # update the initial condition to new basis
        self.parent_slabs[index_parent_slab]["initial_condition"] = self.reduce_vector(
            projected_last_initial, type="primal"
        )
        self.timings["enrichment"] += time.time() - execution_time

    def run_parent_slab(self):
        execution_time = time.time()
        self.init_POD()

        self.affine_decomposition(type="all")

        iteration = 1
        fom_solves = 0
        max_error_iteration = []
        self.parent_slabs[0]["initial_condition"] = np.zeros(
            (self.POD["primal"]["basis"].shape[1],)
        )

        self.DEBUG_FULL_IC = np.zeros((self.POD["primal"]["basis"].shape[0],))

        for index_ps, parent_slab in enumerate(self.parent_slabs):
            print(
                f"====== PARENT SLAB: {index_ps} n=({self.parent_slabs[index_ps]['start']}, {self.parent_slabs[index_ps]['end']}) , t=({self.fom.time_points[self.parent_slabs[index_ps]['start']]:.2}, {self.fom.time_points[np.min([self.parent_slabs[index_ps]['end'],len(self.fom.time_points)-1])]:.2}) ======"
            )
            while iteration <= self.MAX_ITERATIONS:
                print(f"====== Iteration: {iteration} ======")
                print(
                    f"Bases: {self.POD['primal']['basis'].shape[1]} / {self.POD['dual']['basis'].shape[1]}"
                )
                # 1. Solve primal ROM
                self.solve_primal_parent_slab(index_ps)

                # 2. Solve dual ROM
                self.solve_dual_parent_slab(index_ps)

                # 3. Evaluate DWR error estimator
                estimate = self.error_estimate_parent_slab(index_ps)

                max_error_iteration.append(estimate["max"])

                plot_investigation = True

                if plot_investigation:
                    self.iterations_infos[index_ps]["error"].append(estimate["slab_relative_error"])
                    self.iterations_infos[index_ps]["POD_size"]["primal"].append(
                        self.POD["primal"]["basis"].shape[1]
                    )
                    self.iterations_infos[index_ps]["POD_size"]["dual"].append(
                        self.POD["dual"]["basis"].shape[1]
                    )
                    self.iterations_infos[index_ps]["functional"].append(
                        self.parent_slabs[index_ps]["functional_total"]
                    )  # np.sum(self.parent_slabs[index_ps]["functional"]))

                # 4. If relative error is too large, then solve primal and dual FOM on
                # time step with largest error
                if estimate["max"] <= self.REL_ERROR_TOL:
                    print(
                        f" DONE: Largest error @ (i={estimate['i_max']}, t={self.fom.time_points[estimate['i_max'] + self.parent_slabs[index_ps]['start']]:.2}): {estimate['max']:.5}"
                    )
                    break
                else:
                    print(
                        f"Enrich for largest error @ (i={estimate['i_max']}, t={self.fom.time_points[estimate['i_max'] + self.parent_slabs[index_ps]['start']]:.2}): {estimate['max']:.5}"
                    )
                    self.enrich_parent_slab(index_ps, estimate["i_max"])
                    fom_solves += 2

                iteration += 1
            iteration = 1
            print("\n")
            if index_ps < len(self.parent_slabs) - 1:
                self.parent_slabs[index_ps + 1]["initial_condition"] = self.parent_slabs[index_ps][
                    "solution"
                ]["primal"][:, -1]

                # DEBUG by trying full IC
                self.DEBUG_FULL_IC = self.project_vector(
                    self.parent_slabs[index_ps]["solution"]["primal"][:, -1], type="primal"
                )

        self.validate()

        self.timings["run"] += time.time() - execution_time
        print(f"Total FOM solves: {fom_solves}")
