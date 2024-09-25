import math
import os
import time

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from petsc4py import PETSc

"""
This class is needed since parallelization does not work if any Fenics object is passed to a function (aka is used)

THE SOLE PURPOSE OF THIS CLASS IS EVALUTAE THE ERROR ESTIMATE
"""


class iROMGreedyEstimate:
    # constructor
    def __init__(
        self,
        parameter=None,
        time_steps=1,
        REL_ERROR_TOL=1e-2,
        MAX_ITERATIONS=100,
        dt=0.001,
        time_points=None,
        COST_FCT_TRESHOLD=0.0,
    ):
        self.REL_ERROR_TOL = REL_ERROR_TOL
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.time_steps = time_steps
        self.COST_FCT_TRESHOLD = COST_FCT_TRESHOLD

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

    def update_LU(self):
        self.affine_decomposition(type="all", lu=True)

    def greedy_estimate(self):  # , LU_primal, LU_dual):
        self.affine_decomposition(type="all", lu=False)
        self.solve_primal()
        self.solve_dual()

        max_relative_error = self.error_estimate()

        return max_relative_error


    def greedy_estimate_last_time_step(self):
        # just estimate the error at last time step as in original POD-Greedy
        self.affine_decomposition(type="all")
        self.solve_primal()
        
        self.solution["dual"] = np.zeros((self.POD["dual"]["basis"].shape[1], self.time_steps))

        last_time_steps = 100
        # Dual solve
        # #replace loop only over last time steps
        self.solution["dual"][:, -1] = np.zeros((self.POD["dual"]["basis"].shape[1],))
        # for i, t in reversed(list(enumerate(self.time_points[:-1])[-last_time_steps:])):
            
        # length of self.time_points[:-1]
        length = len(self.time_points[:-1])
        for i in reversed(list(range(length - last_time_steps, length))):
            n = i
            # print(f"dual_step: {n} at time: {t}")
            self.solve_dual_time_step(n)


        # this method is only for the validation loop
        # else look in corresponding parent_slab version
        self.errors = np.zeros((self.time_steps - 1,))
        relative_errors = np.zeros((self.time_steps - 1,))
        self.functional_values = np.zeros((self.time_steps - 1,))

        start = 1
        for i in range(self.time_steps - 1 - last_time_steps, self.time_steps - 1):
            n = i + 1  # to skip primal IC
            self.functional_values[i] = self.get_functional(self.solution["primal"][:, n])

        mean_cost_functional = np.mean(np.abs(self.functional_values))

        for i in range(self.time_steps - 1 - last_time_steps, self.time_steps - 1):
            n = i + 1  # to skip primal IC
            self.errors[i] = np.dot(
                self.solution["dual"][:, n - 1],
                -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, n])
                + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, n - 1])
                + self.dt * self.RHS["dual"][:, n],
            )

            # check if cost functional is too small to use relative error
            # estimate -->  set error to zero
            if np.abs(self.functional_values[i]) < self.COST_FCT_TRESHOLD * mean_cost_functional:
                self.errors[i] = 0.0
                # print(f"Cost functional too small at time step {n} --> set error to zero")

            relative_errors[i] = self.errors[i] / (self.errors[i] + self.functional_values[i])

        max_relative_error = np.max(np.abs(relative_errors))
        return max_relative_error













    # def greedy_estimate_last_time_step(self):
    #     # just estimate the error at last time step as in original POD-Greedy
    #     self.affine_decomposition(type="all")
    #     self.solve_primal()
        
    #     # Dual solve
    #     # #replace loop only over last time step
    #     self.solution["dual"] = np.zeros((self.POD["dual"]["basis"].shape[1], self.time_steps))

    #     self.solution["dual"][:, -1] = np.zeros((self.POD["dual"]["basis"].shape[1],))
    #     print(self.solution["dual"].shape)
    #     print(self.time_steps-1)
    #     self.solve_dual_time_step(self.time_steps - 2)
    #     print("check dual integer")
        
        
    #     # for i, t in reversed(list(enumerate(self.time_points[:-1]))):
    #     #     n = i
    #     #     # print(f"dual_step: {n} at time: {t}")
    #     #     self.solve_dual_time_step(n)

    #     # Estimate error

    #     # this method is only for the validation loop
    #     # else look in corresponding parent_slab version
    #     relative_errors = 0.0

    #     n = self.time_steps - 2

    #     functional_values = self.get_functional(n)

    #     mean_cost_functional = np.mean(np.abs(functional_values))

    #     errors =  np.dot( 
    #                     self.solution["dual"][:, n],
    #                     -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, n])
    #                     + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, n - 1])
    #                     + self.dt * self.RHS["dual"][:, n],
    #     )

    #     relative_errors = errors / (errors + functional_values)

    #     max_relative_error = np.max(np.abs(relative_errors))

    #     print(f"Max relative error: {max_relative_error}")

    #     return max_relative_error
    
    
    def greedy_absolute_estimate(self):  # , LU_primal, LU_dual):
        self.affine_decomposition(type="all", lu=False)
        self.solve_primal()
        self.solve_dual()

        _ = self.error_estimate()

        return np.sum(self.errors)

    # look at [[meyer2003]], and [[implement meyer,matthies2003 POD mode
    # selection for whole Greedy Error Esrimation Loop]]
    def sort_out_POD_modes(self, max_relative_error):
        """
        OUTPUT: POD Modes that can be neglected
        If tolerance is not fulfilled by ROM, all modes are neggleceted
        """
        # if error tolerance is not fulfilled, assume no mode is really vital
        if max_relative_error > self.REL_ERROR_TOL:
            return np.arange(self.POD["primal"]["basis"].shape[1])

        # else, sort out modes

        # 1. get POD modes influence on cost functional error
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
            self.dt * np.dot(self.POD["primal"]["basis"].T, np.dot(self.POD["dual"]["basis"], r_m))
        )

        # 2. order POD modes by influence on cost functional error
        order = np.argsort(pod_modes_influence)[::-1]

        # copy POD modes
        POD_tmp = self.POD["primal"]["basis"].copy()
        sigs_tmp = self.POD["primal"]["sigs"].copy()

        primal_system_tmp = self.matrix["primal"]["system"].copy()
        primal_rhs_tmp = self.matrix["primal"]["rhs"].copy()
        RHS_primal_tmp = self.RHS["primal"].copy()

        estimate_system_tmp = self.matrix["estimate"]["system"].copy()
        estimate_rhs_tmp = self.matrix["estimate"]["rhs"].copy()

        for hh in reversed(range(1, len(order))):
            # use the first i-th most impactful POD modes
            self.POD["primal"]["basis"] = POD_tmp[:, order[:hh]].copy()
            self.POD["primal"]["sigs"] = sigs_tmp[order[:hh]].copy()


            # reduce to smaller bases without  FOM matrices needed (or affine_decomposition())
            # reduce primal to smaller basis
            self.matrix["primal"]["system"] = primal_system_tmp[order, :][:, order][:hh, :][
                :, :hh
            ].copy()
            self.matrix["primal"]["rhs"] = primal_rhs_tmp[order, :][:, order][:hh, :][:, :hh].copy()
            self.RHS["primal"] = RHS_primal_tmp[order, :][:hh, :].copy()
            self.update_LES("primal")

            # reduce estimate to smaller basis
            self.matrix["estimate"]["system"] = estimate_system_tmp[:, order][:, :hh].copy()
            self.matrix["estimate"]["rhs"] = estimate_rhs_tmp[:, order][:, :hh].copy()

            # 1. Solve primal
            self.solve_primal()

            # 2. Evaluate DWR error estimator (RELATIVE ERROR)
            # ATTENTION: We use here the best cost functional we have for
            # evaluation. Thus with all POD modes computed instead of inly of
            # these in order
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

                # check if cost functional is too small to use relative error
                # estimate -->  set error to zero
                if (
                    np.abs(self.functional_values[i])
                    < self.COST_FCT_TRESHOLD * mean_cost_functional
                ):
                    # if np.abs(self.functional_values[i]) < quantile:
                    self.errors[i] = 0.0
                    # print(f"Cost functional too small at time step {n} --> set error to zero")

                relative_errors[i] = self.errors[i] / (self.errors[i] + self.functional_values[i])

            max_error_new = np.max(np.abs(relative_errors))
            max_index = np.argmax(np.abs(relative_errors)) + 1

            if max_error_new >= self.REL_ERROR_TOL:
                order = order[hh + 1 :]
                return order

        return order

    def update_ROM(self, matrix, RHS, POD):
        self.matrix = matrix
        self.RHS = RHS
        self.POD = POD

    def reduce_vector(self, vector, type):
        return np.dot(self.POD[type]["basis"].T, vector)

    def update_LES(self, type):
        self.LU[type]["lu"], self.LU[type]["pivot"] = scipy.linalg.lu_factor(
            self.matrix[type]["system"],
            check_finite=False,
        )

    def affine_decomposition(self, type="all", lu=True):
        # TL,DR; get affine decomposition as input and build the matrices

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
            if lu:
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
            if lu:
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
            if lu:
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
            if lu:
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
        for i in range(0, self.time_steps - 1):
            n = i + 1  # to skip primal IC
            self.functional_values[i] = self.get_functional(self.solution["primal"][:, n])

        mean_cost_functional = np.mean(np.abs(self.functional_values))

        for i in range(start, self.time_steps - 1):
            n = i + 1  # to skip primal IC
            self.errors[i] = np.dot(
                self.solution["dual"][:, n - 1],
                -np.dot(self.matrix["estimate"]["system"], self.solution["primal"][:, n])
                + np.dot(self.matrix["estimate"]["rhs"], self.solution["primal"][:, n - 1])
                + self.dt * self.RHS["dual"][:, n],
            )

            # check if cost functional is too small to use relative error
            # estimate -->  set error to zero
            if np.abs(self.functional_values[i]) < self.COST_FCT_TRESHOLD * mean_cost_functional:
                self.errors[i] = 0.0
                # print(f"Cost functional too small at time step {n} --> set error to zero")

            relative_errors[i] = self.errors[i] / (self.errors[i] + self.functional_values[i])

        max_relative_error = np.max(np.abs(relative_errors))
        return max_relative_error

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
                (self.LU["primal"]["lu"], self.LU["primal"]["pivot"]), b, check_finite=False
            )

            self.solution["primal"][:, n] = solution

    def solve_dual(self):
        self.solution["dual"] = np.zeros((self.POD["dual"]["basis"].shape[1], self.time_steps))

        self.solution["dual"][:, -1] = np.zeros((self.POD["dual"]["basis"].shape[1],))
        for i, t in reversed(list(enumerate(self.time_points[:-1]))):
            n = i
            # print(f"dual_step: {n} at time: {t}")
            self.solve_dual_time_step(n)

    def solve_dual_time_step(self, time_step):
        # A = self.matrix["dual"]["system"]
        b = self.matrix["dual"]["rhs"].dot(self.solution["dual"][:, time_step + 1])
        b += self.dt * self.RHS["J_prime"]["dual"]

        self.solution["dual"][:, time_step] = scipy.linalg.lu_solve(
            (self.LU["dual"]["lu"], self.LU["dual"]["pivot"]), b, check_finite=False
        )
