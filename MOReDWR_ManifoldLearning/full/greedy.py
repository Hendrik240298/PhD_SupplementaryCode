import math
import os
import pickle
import re
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fenics import *
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from petsc4py import PETSc
from scipy.stats import norm

from iROM_greedy import iROMGreedy
from iROMGreedy_estimate import iROMGreedyEstimate


# def process_ROM_error_estimation(ROM, matrix, RHS, POD):
#     ROM.update_ROM(matrix=matrix, RHS=RHS, POD=POD)
#     return ROM.greedy_estimate()


def process_ROM_error_estimation_solely(ROM, EVAL_ERROR_LAST_TIME_STEP=False):
    if EVAL_ERROR_LAST_TIME_STEP:
        return ROM.greedy_estimate_last_time_step()
    else:
        return ROM.greedy_estimate()

def process_ROM_absolute_error_estimation_solely(ROM):
    return ROM.greedy_absolute_estimate()

def update_LU(ROM):
    ROM.update_LU()
    return ROM


class Greedy:
    def __init__(
        self,
        fom,
        surrogate=1.0,
        TOTAL_ENERGY={"primal": 1, "dual": 1},
        TOLERANCE=1e-2,
        EVAL_ERROR_LAST_TIME_STEP = False,
        MAX_ITERATIONS=100,
        MAX_ITERATIONS_MORe_DWR=100,
        COST_FCT_TRESHOLD=0.0,
        SAVE_DIR="results/",
        PLOT_DATA=False,
        MEYER_MATTHIES_EXEC=False,
        MEYER_MATTHIES_ITERATIONS=0,
    ):
        self.fom = fom
        self.surrogate = surrogate
        self.number_parameter = surrogate[0].shape[0]
        self.TOLERANCE = TOLERANCE
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.MAX_ITERATIONS_MORe_DWR = MAX_ITERATIONS_MORe_DWR
        self.COST_FCT_TRESHOLD = COST_FCT_TRESHOLD
        self.PLOT_DATA = PLOT_DATA
        self.EVAL_ERROR_LAST_TIME_STEP = EVAL_ERROR_LAST_TIME_STEP
        self.MEYER_MATTHIES_EXEC = MEYER_MATTHIES_EXEC
        self.MEYER_MATTHIES_ITERATIONS = MEYER_MATTHIES_ITERATIONS


        print(f"number parameter: {self.number_parameter}")

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

        self.RHS = {
            "primal": None,
            "dual": None,
            "J_prime": {
                "primal": None,
                "dual": None,
            },
        }

        self.matrix = {
            "primal": {
                "mass": None,
                "stiffness": None,
            },
            "dual": {
                "mass": None,
                "stiffness": None,
            },
            "estimate": {
                "mass": None,
                "stiffness": None,
            },
        }

        self.time_steps = self.fom.dofs["time"]
        self.time_points = self.fom.time_points

        # IO data
        self.SAVE_DIR = SAVE_DIR

    def load_FOM_functionals(self, parameter):
        # load self.functional_values of fom
        pattern = r"fom_functional_values_\d{6}\.npz"
        files = os.listdir(self.SAVE_DIR)
        files = [
            self.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        discretization_parameters = np.array([self.fom.nx, self.fom.ny, self.fom.dt, self.fom.T])

        for file in files:
            tmp = np.load(file)
            if np.allclose(discretization_parameters, tmp["discretization_parameters"], atol=1e-10):
                if np.allclose(parameter, tmp["parameter"], atol=1e-10):
                    functional_values = tmp["functional_values"]
                    # print(f"Loaded {file}")
                    return functional_values

        print(f"Could not find functional values for parameter {parameter}")

        return False

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

    def reduce_vector(self, vector, type):
        return np.dot(self.POD[type]["basis"].T, vector)

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

    def reduce_affine_decomposition(self):
        self.matrix["primal"]["mass"] = self.reduce_matrix(
            self.fom.matrix["primal"]["mass"], type="primal"
        )
        self.matrix["primal"]["stiffness"] = [None] * self.number_parameter
        for i in range(self.number_parameter):
            self.matrix["primal"]["stiffness"][i] = self.reduce_matrix(
                self.fom.matrix["primal"]["stiffness"][i], type="primal"
            )

        # dual
        self.matrix["dual"]["mass"] = self.reduce_matrix(
            self.fom.matrix["dual"]["mass"], type="dual"
        )
        self.matrix["dual"]["stiffness"] = [None] * self.number_parameter
        for i in range(self.number_parameter):
            self.matrix["dual"]["stiffness"][i] = self.reduce_matrix(
                self.fom.matrix["dual"]["stiffness"][i], type="dual"
            )

        # estimator
        self.matrix["estimate"]["mass"] = self.reduce_matrix(
            self.fom.matrix["primal"]["mass"], type="estimate"
        )
        self.matrix["estimate"]["stiffness"] = [None] * self.number_parameter
        for i in range(self.number_parameter):
            self.matrix["estimate"]["stiffness"][i] = self.reduce_matrix(
                self.fom.matrix["primal"]["stiffness"][i], type="estimate"
            )

        # rhs
        self.update_rhs(type="primal")  # also dual RHS included
        self.update_rhs(type="dual")  # primal reduced with dual basis

    def update_ROMs(self):
        for i, ROM in enumerate(self.ROM_surrogate):
            ROM.update_ROM(matrix=self.matrix, RHS=self.RHS, POD=self.POD)
        for i, ROM in enumerate(self.ROM_surrogate_estimate):
            ROM.update_ROM(matrix=self.matrix, RHS=self.RHS, POD=self.POD)

    def init_surrogate(self):
        self.ROM_surrogate = []
        self.ROM_surrogate_estimate = []
        for parameter in self.surrogate:
            self.ROM_surrogate.append(
                iROMGreedy(
                    parameter=parameter,
                    time_steps=self.time_steps,
                    dt=self.fom.dt,
                    time_points=self.time_points,
                    fom=self.fom,
                    MAX_ITERATIONS=self.MAX_ITERATIONS_MORe_DWR,
                    COST_FCT_TRESHOLD=self.COST_FCT_TRESHOLD,
                    REL_ERROR_TOL=self.TOLERANCE,
                    MEYER_MATTHIES_EXEC=self.MEYER_MATTHIES_EXEC,
                    MEYER_MATTHIES_ITERATIONS=self.MEYER_MATTHIES_ITERATIONS,
                )
            )
            self.ROM_surrogate_estimate.append(
                iROMGreedyEstimate(
                    parameter=parameter,
                    time_steps=self.time_steps,
                    dt=self.fom.dt,
                    time_points=self.time_points,
                    MAX_ITERATIONS=self.MAX_ITERATIONS_MORe_DWR,
                    COST_FCT_TRESHOLD=self.COST_FCT_TRESHOLD,
                    REL_ERROR_TOL=self.TOLERANCE,
                )
            )

    def init_POD(self, parameter=np.ones(16)):
        # primal POD
        self.fom.parameter = parameter
        self.fom.assemble_system(force_recompute=False)

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

    def estimate_surrogate(self):
        relative_error = np.zeros(len(self.ROM_surrogate))
        for i, ROM in enumerate(self.ROM_surrogate):
            tic = time.time()
            ROM.update_ROM(matrix=self.matrix, RHS=self.RHS, POD=self.POD)
            # just the value not the location
            relative_error[i] = ROM.greedy_estimate()[0]
            toc = time.time()
            # print(f"update_ROM took {toc-tic:2.4f} seconds")

        return relative_error

    def estimate_surrogate_parallel(self):
        for i, ROM in enumerate(self.ROM_surrogate_estimate):
            ROM.update_ROM(matrix=self.matrix, RHS=self.RHS, POD=self.POD)

        relative_errors = np.zeros(len(self.ROM_surrogate_estimate))
        for i, ROM in enumerate(self.ROM_surrogate_estimate):
            tic = time.time()
            ROM = ROM.update_LU()

            toc = time.time()

        with Pool() as pool:
            # Prepare arguments for each process

            args = [(ROM,self.EVAL_ERROR_LAST_TIME_STEP,) for ROM in self.ROM_surrogate_estimate]
            # Parallel execution
            relative_errors = pool.starmap(process_ROM_error_estimation_solely, args)
            
        return relative_errors

    def absolute_error_for_effectivity(self):
        for i, ROM in enumerate(self.ROM_surrogate_estimate):
            ROM.update_ROM(matrix=self.matrix, RHS=self.RHS, POD=self.POD)

        absolute_errors = np.zeros(len(self.ROM_surrogate_estimate))
        for i, ROM in enumerate(self.ROM_surrogate_estimate):
            ROM = ROM.update_LU()

        with Pool() as pool:
            # Prepare arguments for each process

            args = [(ROM,) for ROM in self.ROM_surrogate_estimate]
            # Parallel execution
            absolute_errors = pool.starmap(process_ROM_absolute_error_estimation_solely, args)
            
        return absolute_errors

    def greedy_enirchment(self, initial_parameter_index=-1):
        # performance measurements
        self.iteration = {
            "relative_error": [],
            "arr_rel_error": [],
            "fom_error": [],
            "fom_error_max": [],
            "fom_solves": [],
            "POD_size": 
            {
                "primal": [],
                "dual": [],
            },
        }

        # init surrogate by initializing ROM classes for parameter
        self.init_surrogate()

        # init ROM
        self.init_POD(parameter=self.surrogate[initial_parameter_index])
        self.iteration["fom_solves"].append(2)        

        iter = 0
        parameter_index = initial_parameter_index
        old_parameter_index = initial_parameter_index
        print("Started Greedy Loop")
        while iter < self.MAX_ITERATIONS:
            self.reduce_affine_decomposition()
            self.update_ROMs()

            tic = time.time()
            relative_error = self.estimate_surrogate_parallel()
            toc = time.time()
            print(f"Estimation parallel took {toc-tic} seconds")

            # find parameter with largest error
            parameter_index = np.argmax(relative_error)
            print(
                f"Largest error {relative_error[parameter_index]} at parameter_index {parameter_index}"
            )

            # Compute real error to FOM for plot
            if self.PLOT_DATA:
                self.iteration["fom_error"].append(self.exact_error())
                max_real_error_index = np.argmax(self.iteration["fom_error"][-1])
                self.iteration["fom_error_max"].append(
                    self.iteration["fom_error"][-1][max_real_error_index]
                )
                print(
                    f"Largest real error {self.iteration['fom_error'][-1][max_real_error_index]} at parameter_index {max_real_error_index}"
                )

            self.iteration["relative_error"].append(relative_error[parameter_index])
            self.iteration["arr_rel_error"].append(relative_error)
            self.iteration["POD_size"]["primal"].append(self.POD["primal"]["basis"].shape[1])
            self.iteration["POD_size"]["dual"].append(self.POD["dual"]["basis"].shape[1])
            
            if relative_error[parameter_index] < self.TOLERANCE:
                print(f"Greedy finished after {iter} iterations")
                break

            print(f"Greedy enrichment with parameter: {self.surrogate[parameter_index][0::4]}")
            self.POD, fom_solves = self.ROM_surrogate[parameter_index].run_ROM()
            self.iteration["fom_solves"].append(fom_solves)

            # Other Stuff
            DEBUG = False
            if DEBUG:
                # DEBUG: Plot CF of enriched
                fom_functional = self.load_FOM_functionals(self.surrogate[parameter_index])
                if fom_functional_values[i] is False:
                    raise ValueError("FOM functional values could not be loaded")
                mu, std = norm.fit(self.ROM_surrogate[parameter_index].functional_values)
                quantile = norm.ppf(self.COST_FCT_TRESHOLD, loc=mu, scale=std)

                if fom_functional is False:
                    plt.plot(
                        self.time_points[:-1],
                        self.ROM_surrogate[parameter_index].functional_values,
                        label="estimated",
                    )
                    # plt.yscale("log")

                    plt.axhline(
                        y=self.COST_FCT_TRESHOLD
                        * np.max(self.ROM_surrogate[parameter_index].functional_values),
                        color="g",
                        linestyle="--",
                        label="COST_FCT_TRESHOLD",
                    )
                    # plt.axhline(y = quantile, color="g", linestyle="--", label="COST_FCT_TRESHOLD")
                    plt.grid()
                    plt.legend()
                    plt.show()
                else:
                    # two plots in one figure: left FOM and ROM, Right:
                    # Relative error
                    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                    axs[0].plot(
                        self.time_points[:-1],
                        self.ROM_surrogate[parameter_index].functional_values,
                        label="estimated",
                    )
                    axs[0].plot(self.time_points[:-1], fom_functional[:], label="fom")
 
                    
                    axs[0].axhline(
                        y=self.COST_FCT_TRESHOLD * np.max(fom_functional[:]),
                        color="g",
                        linestyle="--",
                        label="COST_FCT_TRESHOLD",
                    )

                    axs[0].grid()
                    axs[0].legend()
                    axs[1].plot(
                        self.time_points[:-1],
                        np.abs(
                            fom_functional[:]
                            - self.ROM_surrogate[parameter_index].functional_values[:]
                        )
                        / np.abs(fom_functional[:]),
                        label="relative error",
                    )
                    axs[1].axhline(y=self.TOLERANCE, color="g", linestyle="--", label="tolerance")
                    axs[1].grid()
                    axs[1].set_yscale("log")
                    axs[1].legend()
                    plt.show()

            iter += 1

        with open(self.SAVE_DIR + "iteration_data.pkl", "wb") as f:
            pickle.dump(self.iteration, f)
        with open(self.SAVE_DIR + "surrogate.pkl", "wb") as f:
            pickle.dump(self.surrogate, f)
            
        absolute_errors = self.absolute_error_for_effectivity()
        with open(self.SAVE_DIR + "absolute_errors.pkl", "wb") as f:
            pickle.dump(absolute_errors, f)

        # self.validate_estimates()

    def exact_error(self):
        print("Computing exact error")
        # to get the ROM functional values since they get lost in
        # parallelization
        tic = time.time()
        self.estimate_surrogate()

        fom_functional_values = []
        fom_relative_error = []
        for i, ROM in enumerate(self.ROM_surrogate):
            fom_functional_values.append(self.load_FOM_functionals(ROM.parameter))
            if fom_functional_values[i] is False:
                raise ValueError("FOM functional values could not be loaded")
            # np.mean(np.abs(ROM.functional_values[:-2]))
            mean_cost_functional = np.mean(fom_functional_values[i])
            error = np.zeros(self.time_steps - 1)
            for kk in range(0, self.time_steps - 1):
                error[kk] = np.abs(fom_functional_values[i][kk] - ROM.functional_values[kk])
                if (
                    np.abs(fom_functional_values[i][kk])
                    < self.COST_FCT_TRESHOLD * mean_cost_functional
                ):
                    error[kk] = 0.0

            relative_error = np.max(error / np.abs(fom_functional_values[i]))
            fom_relative_error.append(relative_error)
        toc = time.time()
        print(f"Max FOM relative error: {np.max(fom_relative_error):2.4f} in {toc-tic} seconds")

        return fom_relative_error

    def validate_estimates(self):
        # etsimated error
        relative_error = self.estimate_surrogate_parallel()

        print(f"relative error: {relative_error}")

        # to get the ROM functional values since they get lost in
        # parallelization
        self.estimate_surrogate()

        # Safe ROM functional values to file
        for i, ROM in enumerate(self.ROM_surrogate):
            print(f"Saving ROM functional values for parameter: {ROM.parameter} at {self.SAVE_DIR}/ROM_functional_values_{i:06d}.npz")
            np.savez(
                self.SAVE_DIR + f"ROM_functional_values_{i:06d}.npz",
                parameter=ROM.parameter,
                functional_values=ROM.functional_values,
            )

        # real error to fom
        fom_functional_values = []
        fom_relative_error = []
        for i, ROM in enumerate(self.ROM_surrogate):
            fom_functional_values.append(self.load_FOM_functionals(ROM.parameter))
            if fom_functional_values[i] is False:
                raise ValueError("FOM functional values could not be loaded")
            mean_cost_functional = np.mean(np.abs(ROM.functional_values[:-2]))
            error = np.zeros(self.time_steps - 1)
            for kk in range(0, self.time_steps - 1):
                error[kk] = np.abs(fom_functional_values[i][kk] - ROM.functional_values[kk])
                if (
                    np.abs(ROM.functional_values[kk])
                    < self.COST_FCT_TRESHOLD * mean_cost_functional
                ):
                    error[kk] = 0.0

            max_relative_error = 100 * np.max(error / np.abs(fom_functional_values[i]))
            fom_relative_error.append(max_relative_error)
            print(f"max relative error: {max_relative_error}")
            print(
                f"at position: {np.argmax(np.abs(fom_functional_values[i] - ROM.functional_values) / np.abs(fom_functional_values[i]))}"
            )

        # plot relative error
        plot_para = np.zeros(len(self.ROM_surrogate))
        print(f"len plot_para: {len(plot_para)}")
        for i, parameter in enumerate(self.surrogate):
            plot_para[i] = parameter[0]
            print(f"parameter: {parameter[0]}")

        # multiply each relative error with 100 to get percentage (Attention
        # relative_error is list)
        relative_error = np.array(relative_error) * 100

        # # plot fom relative error vs error estimate
        # plt.plot(plot_para, relative_error, label="relative error estimate")
        # plt.plot(plot_para, fom_relative_error, label="fom relative error")
        # plt.yscale("log")
        # plt.legend()
        # plt.grid()
        # plt.show()

        # # plot cost functional for first (left), middle (mid) and last element
        # # in self.ROM_surrogate in subfigures (see brackets)
        # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        # axs[0].plot(
        #     self.time_points[:-3], self.ROM_surrogate[0].functional_values[:-2], label="estimated"
        # )
        # axs[0].plot(self.time_points[:-3], fom_functional_values[0][:-2], label="fom")
        # axs[0].legend()
        # axs[0].grid()
        # axs[0].set_title("left")
        # axs[1].plot(
        #     self.time_points[:-3],
        #     self.ROM_surrogate[len(self.ROM_surrogate) // 2].functional_values[:-2],
        #     label="estimated",
        # )
        # axs[1].plot(
        #     self.time_points[:-3],
        #     fom_functional_values[len(self.ROM_surrogate) // 2][:-2],
        #     label="fom",
        # )
        # axs[1].legend()
        # axs[1].grid()
        # axs[1].set_title("mid")
        # axs[2].plot(
        #     self.time_points[:-3],
        #     self.ROM_surrogate_estimate[-1].functional_values[:-2],
        #     label="estimated",
        # )
        # axs[2].plot(self.time_points[:-3], fom_functional_values[-1][:-2], label="fom")
        # axs[2].legend()
        # axs[2].grid()
        # axs[2].set_title("right")
        # plt.show()