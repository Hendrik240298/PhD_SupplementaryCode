""" ------------ IMPLEMENTATION of ROM ------------
"""
import math
import os
import re
import time
from multiprocessing import Process, Queue

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
import scipy
from dolfin import *
from fenics import *
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from petsc4py import PETSc
from slepc4py import SLEPc

import logging
# configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)

class ROM:
    # constructor
    def __init__(
        self,
        fom,
        TOTAL_ENERGY={
            "primal": {"velocity": 1, "pressure": 1},
        },
    ):
        self.fom = fom

        self.fom.Y.update(
            {
                "supremizer": np.zeros((self.fom.dofs["velocity"], self.fom.dofs["time"])),
            }
        )

        self.fom.dofs.update({"supremizer": self.fom.dofs["velocity"]})

        self.POD = {
            "primal": {
                "velocity": {
                    "basis": np.empty((self.fom.dofs["velocity"], 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.empty((self.fom.dofs["velocity"], 0)),
                    "bunch_size": 1,
                    "TOL_ENERGY": TOTAL_ENERGY["primal"]["velocity"],
                },
                "pressure": {
                    "basis": np.empty((self.fom.dofs["pressure"], 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.empty((self.fom.dofs["pressure"], 0)),
                    "bunch_size": 1,
                    "TOL_ENERGY": TOTAL_ENERGY["primal"]["pressure"],
                },
                "supremizer": {
                    "basis": np.empty((self.fom.dofs["velocity"], 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.empty((self.fom.dofs["velocity"], 0)),
                    "bunch_size": 1,
                    "TOL_ENERGY": TOTAL_ENERGY["primal"]["velocity"],
                },
            },
        }

        # TODO: think about which matrices needed
        self.matrix = {
            "primal": {"jacobi": None, "new_system": None, "old_system": None},
        }

        self.solution = {"primal": {"velocity": None, "pressure": None}}
        self.functional_values = np.zeros((self.fom.dofs["time"] - 1,))

        # lifting function
        self.lifting = {}
        # TODO: do we need separate lifting functions for primal and dual?

    def iPOD(self, snapshot, type, quantity, equal_size=False, empty_bunch_matrix=False):
        # type is either "primal" or "dual"
        # empty_bunch_matrix to empty bunch matrix even if not full
        self.POD[type][quantity]["bunch"] = np.hstack(
            (self.POD[type][quantity]["bunch"], snapshot.reshape(-1, 1))
        )

        # add energy of new snapshot to total energy
        self.POD[type][quantity]["energy"] += np.dot((snapshot), (snapshot))

        # check bunch_matrix size to decide if to update POD
        if self.POD[type][quantity]["bunch"].shape[1] == self.POD[type][quantity]["bunch_size"] or empty_bunch_matrix:
            logging.info(f"Update {quantity} POD with {self.POD[type][quantity]['bunch_size']} snapshots with {self.POD[type][quantity]['basis'].shape[1]} modes")
            # initialize POD with first bunch matrix
            if self.POD[type][quantity]["basis"].shape[1] == 0:
                (
                    self.POD[type][quantity]["basis"],
                    self.POD[type][quantity]["sigs"],
                    _,
                ) = scipy.linalg.svd(self.POD[type][quantity]["bunch"], full_matrices=False)

                # compute the number of POD modes to be kept
                r = 0
                while (
                    np.dot(
                        self.POD[type][quantity]["sigs"][0:r], self.POD[type][quantity]["sigs"][0:r]
                    )
                    <= self.POD[type][quantity]["energy"] * self.POD[type][quantity]["TOL_ENERGY"]
                ) and (r <= np.shape(self.POD[type][quantity]["sigs"])[0]):
                    r += 1

                if quantity == "supremizer":
                    r = self.POD[type]["pressure"]["basis"].shape[1]

                self.POD[type][quantity]["sigs"] = self.POD[type][quantity]["sigs"][0:r]
                self.POD[type][quantity]["basis"] = self.POD[type][quantity]["basis"][:, 0:r]
            # update POD with  bunch matrix
            else:
                M = np.dot(self.POD[type][quantity]["basis"].T, self.POD[type][quantity]["bunch"])
                P = self.POD[type][quantity]["bunch"] - np.dot(self.POD[type][quantity]["basis"], M)

                Q_p, R_p = scipy.linalg.qr(P, mode="economic")
                Q_q = np.hstack((self.POD[type][quantity]["basis"], Q_p))

                S0 = np.vstack(
                    (
                        np.diag(self.POD[type][quantity]["sigs"]),
                        np.zeros((np.shape(R_p)[0], np.shape(self.POD[type][quantity]["sigs"])[0])),
                    )
                )
                MR_p = np.vstack((M, R_p))
                K = np.hstack((S0, MR_p))

                # check the orthogonality of Q_q heuristically
                if np.abs(np.inner(Q_q[:, 0], Q_q[:, -1])) >= 1e-10:
                    Q_q, R_q = scipy.linalg.qr(Q_q, mode="economic")
                    K = np.matmul(R_q, K)

                # inner SVD of K
                U_k, S_k, _ = scipy.linalg.svd(K, full_matrices=False)

                # compute the number of POD modes to be kept
                r = self.POD[type][quantity]["basis"].shape[1]

                while (
                    np.dot(S_k[0:r], S_k[0:r])
                    <= self.POD[type][quantity]["energy"] * self.POD[type][quantity]["TOL_ENERGY"]
                ) and (r < np.shape(S_k)[0]):
                    r += 1

                if equal_size:
                    max = np.max([
                        self.POD[type]["velocity"]["basis"].shape[1],
                        self.POD[type]["pressure"]["basis"].shape[1],
                    ])
                    
                    r = np.max([np.min([np.shape(S_k)[0], max]),r])

                if quantity == "supremizer":
                    r = self.POD[type]["pressure"]["basis"].shape[1]

                self.POD[type][quantity]["sigs"] = S_k[0:r]
                self.POD[type][quantity]["basis"] = np.matmul(Q_q, U_k[:, 0:r])

            # empty bunch matrix after update
            self.POD[type][quantity]["bunch"] = np.empty([self.fom.dofs[quantity], 0])

    def reduce_vector(self, vector, type, quantity):
        return np.dot(self.POD[type][quantity]["basis"].T, vector)

    def project_vector(self, vector, type, quantity):
        return np.dot(self.POD[type][quantity]["basis"], vector)

    # TODO: consider also supremizer here
    def reduce_matrix(self, matrix, type, quantity0, quantity1):
        """
        OLD:
            |   A_uu    |   A_up    |
        A = | --------- | --------- |
            |   A_pu    |   A_pp    |

        NEW: (is always better - B.S.)

        A_N = Z_{q0}^T A Z_{q1}

        REMARK:
        - A_N is reduced submatrix, c.f. OLD A
        """

        reduced_matrix = self.POD[type][quantity0]["basis"].T.dot(
            matrix.dot(self.POD[type][quantity1]["basis"])
        )
        return reduced_matrix

    def update_matrices(self, type):
        # TODO: update when matrix are known

        # self.

        for key in self.matrix[type].keys():
            self.matrix[type][key] = self.reduce_matrix(self.fom.matrix[type][key], type)

    def update_rhs(self, type):
        self.RHS[type] = np.dot(self.POD[type]["basis"].T, self.fom.RHS)

    def init_POD(self):
        # primal POD
        time_points = self.fom.time_points[:]

        logging.info("Start POD")

        for i, t in enumerate(time_points[500:]):
            self.iPOD(self.fom.Y["velocity"][:, i+500], type="primal", quantity="velocity", equal_size=True)
            self.iPOD(self.fom.Y["pressure"][:, i+500], type="primal", quantity="pressure", equal_size=True)
        
        # # because of the size restrition on r in the iPOD
        # for i, t in enumerate(time_points[500:]):
            # logging.info(f"Enrich with time step {i} at time {t}")
            self.iPOD(self.fom.Y["supremizer"][:, i+500], type="primal", quantity="supremizer")

        logging.info(f"VELOCITY POD size:   {self.POD['primal']['velocity']['basis'].shape[1]}")
        logging.info(f"PRESSURE POD size:   {self.POD['primal']['pressure']['basis'].shape[1]}")
        logging.info(f"SUPREMIZER POD size: {self.POD['primal']['supremizer']['basis'].shape[1]}")

        # for i in range(self.POD["primal"]["velocity"]["basis"].shape[1]):
        #     v, p = self.fom.U_n.split()
        #     v.vector().set_local(self.POD["primal"]["velocity"]["basis"][:,i])
        #     c = plot(sqrt(dot(v, v)), title="Velocity")
        #     plt.colorbar(c, orientation="horizontal")
        #     plt.show()

    def compute_lifting_function(self, force_recompute=False):
        if not force_recompute:
            if self.load_lifting_function():
                # solution = np.concatenate((self.lifting["velocity"], self.lifting["pressure"]))
                # U_lifting = Function(self.fom.V)  # lifting function
                # U_lifting.vector().set_local(solution)
                # v, p = split(U_lifting)
                # plt.title("Lifting function")
                # c = plot(sqrt(dot(v, v)), title="Velocity")
                # plt.colorbar(c, orientation="horizontal")
                # plt.show()
                return

        U_lifting = Function(self.fom.V)  # lifting function
        Phi_U = TestFunctions(self.fom.V)

        # Split functions into velocity and pressure components
        v, p = split(U_lifting)
        phi_v, phi_p = Phi_U

        # Define variational forms
        a = (self.fom.nu * inner(grad(v), grad(phi_v)) - p * div(phi_v) - div(v) * phi_p) * dx

        solve(a == 0, U_lifting, self.fom.bc)

        # plt.title("Lifting function")
        # c = plot(sqrt(dot(v, v)), title="Velocity")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        v, p = U_lifting.split()
        self.lifting["velocity"] = v.vector().get_local()[: self.fom._V.dim()]
        self.lifting["pressure"] = p.vector().get_local()[self.fom._V.dim() :]


        # self.lifting["velocity"] = np.mean(self.fom.Y["velocity"][:, 500:], axis=1)

        self.save_lifting_function()

    def subtract_lifting_function(self):
        for i, t in enumerate(self.fom.time_points[:]):
            self.fom.Y["velocity"][:, i] -= self.lifting["velocity"]
            self.fom.Y["supremizer"][:, i] -= self.lifting["velocity"]

    def save_lifting_function(self):
        pattern = "lifting_function.npz"
        files = os.listdir(self.fom.SAVE_DIR)
        files = [
            self.fom.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.fom.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.fom.dt, self.fom.T, self.fom.theta, float(self.fom.nu)])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    velocity=self.lifting["velocity"],
                    pressure=self.lifting["pressure"],
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return

        file_name = "results/lifting_function.npz"
        np.savez(
            file_name,
            velocity=self.lifting["velocity"],
            pressure=self.lifting["pressure"],
            parameters=parameters,
            compression=True,
        )
        print(f"Saved as {file_name}")

    def load_lifting_function(self):
        pattern = "lifting_function.npz"
        files = os.listdir(self.fom.SAVE_DIR)
        files = [
            self.fom.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.fom.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.fom.dt, self.fom.T, self.fom.theta, float(self.fom.nu)])

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.lifting["velocity"] = tmp["velocity"]
                self.lifting["pressure"] = tmp["pressure"]
                print(f"Loaded {file}")
                return True
        return False

    def compute_reduced_matrices(self):
        self.reduce_lifting_matrices()
        self.reduce_linear_operators()
        self.nonlinearity_tensor = {}
        self.nonlinearity_tensor["velocity"] = np.array(
            self.compute_reduced_nonlinearity(type="primal", quantity="velocity")
        )
        self.nonlinearity_tensor["supremizer"] = np.array(
            self.compute_reduced_nonlinearity(type="primal", quantity="supremizer")
        )

    def reduce_lifting_matrices(self):
        self.velocity_lifting_matrix = self.reduce_matrix(
            self.fom.lifting_matrix, type="primal", quantity0="velocity", quantity1="velocity"
        )
        self.supremizer_lifting_matrix = self.reduce_matrix(
            self.fom.lifting_matrix, type="primal", quantity0="supremizer", quantity1="velocity"
        )
        self.velocity_lifting_rhs = self.reduce_vector(
            self.fom.lifting_rhs, type="primal", quantity="velocity"
        )
        self.supremizer_lifting_rhs = self.reduce_vector(
            self.fom.lifting_rhs, type="primal", quantity="supremizer"
        )
        print("Reduced lifting matrices")

    def reduce_linear_operators(self):
        self.velocity_lin_operator_theta = self.reduce_matrix(
            self.fom.velocity_lin_operator_theta,
            type="primal",
            quantity0="velocity",
            quantity1="velocity",
        )
        self.velocity_lin_operator_one_minus_theta = self.reduce_matrix(
            self.fom.velocity_lin_operator_one_minus_theta,
            type="primal",
            quantity0="velocity",
            quantity1="velocity",
        )
        self.supremizer_lin_operator_theta = self.reduce_matrix(
            self.fom.velocity_lin_operator_theta,
            type="primal",
            quantity0="supremizer",
            quantity1="velocity",
        )
        self.supremizer_lin_operator_one_minus_theta = self.reduce_matrix(
            self.fom.velocity_lin_operator_one_minus_theta,
            type="primal",
            quantity0="supremizer",
            quantity1="velocity",
        )
        self.pressure_lin_operator = self.reduce_matrix(
            self.fom.pressure_lin_operator,
            type="primal",
            quantity0="supremizer",
            quantity1="pressure",
        )
        print("Reduced linear operators")

    def assemble_system_matrix(self):
        size_v_N = self.POD["primal"]["velocity"]["basis"].shape[1]
        # == size_p_N !
        size_s_N = self.POD["primal"]["supremizer"]["basis"].shape[1]
        size_p_N = self.POD["primal"]["pressure"]["basis"].shape[1]

        system_matrix = np.zeros((size_v_N + size_s_N, size_v_N + size_p_N))

        # top left
        ## linear part
        system_matrix[:size_v_N, :size_v_N] = (
            self.velocity_lin_operator_theta 
        )
        ## nonlinearities
        system_matrix[:size_v_N, :size_v_N] += (
            self.fom.theta
            * self.fom.dt
            * self.evaluate_reduced_nonlinearity(vector=self.U[:size_v_N], quantity="velocity")
        )

        # bottom left
        ## linear part
        system_matrix[size_v_N:, :size_v_N] = (
            self.supremizer_lin_operator_theta 
        )
        ## nonlinearities
        system_matrix[size_v_N:, :size_v_N] += (
            self.fom.theta
            * self.fom.dt
            * self.evaluate_reduced_nonlinearity(vector=self.U[:size_v_N], quantity="supremizer")
        )      
        # bottom right
        system_matrix[size_v_N:, size_v_N:] = self.pressure_lin_operator
        
        return system_matrix

    def assemble_system_rhs(self):
        size_v_N = self.POD["primal"]["velocity"]["basis"].shape[1]
        # == size_p_N !
        size_s_N = self.POD["primal"]["supremizer"]["basis"].shape[1]
        system_rhs = np.zeros((size_v_N + size_s_N,))


        # current timestep
        # velocity part: -A_v u - theta C_v^l u - theta dt D_v(u)u
        # -A_v u - theta C_v^l u
        system_rhs[:size_v_N] -= np.dot(self.velocity_lin_operator_theta, self.U[:size_v_N])


        # system_rhs[:size_v_N] -= self.fom.theta * np.dot(
        #     self.velocity_lifting_matrix, self.U[:size_v_N]
        # )

        # # DBEUG: INLCUDE PRESSURE FOR VELOCITY
        # velocity_pressure_lin_operator = self.reduce_matrix(
        #     self.fom.pressure_lin_operator,
        #     type="primal",
        #     quantity0="velocity",
        #     quantity1="pressure",
        # )
        # print("SHAPE MAT: ", velocity_pressure_lin_operator.shape)
        # print("SHAPE VEC: ", self.U[:size_v_N].shape)
        # print("SHAPE SOL: ", system_rhs[:size_v_N].shape)
        # print("SOL:       ", np.dot(velocity_pressure_lin_operator, self.U[size_v_N:]))
        # system_rhs[:size_v_N] -= np.dot(velocity_pressure_lin_operator, self.U[size_v_N:])

        # - theta dt D_v(u)u
        system_rhs[:size_v_N] -= (
            self.fom.theta
            * self.fom.dt
            * self.twice_evaluate_reduced_nonlinearity(
                vector=self.U[:size_v_N], quantity="velocity"
            )
        )

        # pressure part: - A_s u - theta C_s^l u - P_p p - theta dt D_s(u)u
        # A_s u - theta C_s^l u
        system_rhs[size_v_N:] -= np.dot(self.supremizer_lin_operator_theta, self.U[:size_v_N])
        # system_rhs[size_v_N:] -= self.fom.theta * np.dot(
        #     self.supremizer_lifting_matrix, self.U[:size_v_N]
        # )
        # - P_p p
        system_rhs[size_v_N:] -= np.dot(self.pressure_lin_operator, self.U[size_v_N:])
        # - theta dt D_s(u)u
        system_rhs[size_v_N:] -= (
            self.fom.theta
            * self.fom.dt
            * self.twice_evaluate_reduced_nonlinearity(
                vector=self.U[:size_v_N], quantity="supremizer"
            )
        )

        # old timestep
        # velocity part: A^{-theta} u_n - (1-theta) dt C_v^l u_n - (1-theta) dt D_v(u_n)u_n
        # A^{-theta} u_n - (1-theta) dt C_v^l u_n
        system_rhs[:size_v_N] += np.dot(
            self.velocity_lin_operator_one_minus_theta, self.U_n[:size_v_N]
        )
        # print(f"MAX LIN: {np.linalg.norm(np.dot(self.velocity_lin_operator_one_minus_theta, self.U_n[:size_v_N]), ord=np.inf)}")
        # DEBUG HF: changed sign
        # system_rhs[:size_v_N] -= (1.0 - self.fom.theta) * np.dot(
        #     self.velocity_lifting_matrix, self.U_n[:size_v_N]
        # )
        # DEBUG HF: changed sign
        # - (1-theta) dt D_v(u_n)u_n
        system_rhs[:size_v_N] -= (
            (1.0 - self.fom.theta)
            * self.fom.dt
            * self.twice_evaluate_reduced_nonlinearity(
                vector=self.U_n[:size_v_N], quantity="velocity"
            )
        )

        # print(f"MAX NONLIN: {np.linalg.norm(self.twice_evaluate_reduced_nonlinearity(vector=self.U_n[:size_v_N], quantity='velocity'), ord=np.inf)}")

        # pressure part: A_s^{-theta} u_n - (1-theta) dt C_s^l u_n - (1-theta) dt D_s(u_n)u_n
        # A_s^{-theta} u_n - (1-theta) dt C_s^l u_n
        system_rhs[size_v_N:] += np.dot(
            self.supremizer_lin_operator_one_minus_theta, self.U_n[:size_v_N]
        )

        # print(f"MAX LIN: {np.linalg.norm(np.dot(self.supremizer_lin_operator_one_minus_theta, self.U_n[:size_v_N]), ord=np.inf)}")

        # 2 DEBUG HF: lifting already in supremizer_lin_operator_one_minus_theta??
        # 1 DEBUG HF: changed sign
        # system_rhs[size_v_N:] -= (1.0 - self.fom.theta) * np.dot(
        #     self.supremizer_lifting_matrix, self.U_n[:size_v_N]
        # )
        # DEBUG HF: changed sign
        # - (1-theta) dt D_s(u_n)u_n
        system_rhs[size_v_N:] -= (
            (1.0 - self.fom.theta)
            * self.fom.dt
            * self.twice_evaluate_reduced_nonlinearity(
                vector=self.U_n[:size_v_N], quantity="supremizer"
            )
        )

        # lifting
        system_rhs[:size_v_N] += self.velocity_lifting_rhs
        # # DEBUG HF
        system_rhs[size_v_N:] += self.supremizer_lifting_rhs

        return system_rhs

    def save_supremizer(self):
        pattern = f"supremizer_{self.fom.mesh_name}_" r"\d{6}\.npz"
        files = os.listdir(self.fom.SAVE_DIR)
        files = [
            self.fom.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.fom.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.fom.dt, self.fom.T, self.fom.theta, float(self.fom.nu)])

        for file in files:
            tmp = np.load(file, allow_pickle=True)
            if np.array_equal(parameters, tmp["parameters"]):
                np.savez(
                    file,
                    supremizer=self.fom.Y["supremizer"],
                    parameters=parameters,
                    compression=True,
                )
                print(f"Overwrite {file}")
                return

        file_name = f"results/supremizer_{self.fom.mesh_name}_" + str(len(files)).zfill(6) + ".npz"
        np.savez(
            file_name,
            supremizer=self.fom.Y["supremizer"],
            parameters=parameters,
            compression=True,
        )
        print(f"Saved as {file_name}")

    def load_supremizer(self):
        pattern = f"supremizer_{self.fom.mesh_name}_" r"\d{6}\.npz"
        files = os.listdir(self.fom.SAVE_DIR)
        files = [
            self.fom.SAVE_DIR + f
            for f in files
            if os.path.isfile(os.path.join(self.fom.SAVE_DIR, f)) and re.match(pattern, f)
        ]

        parameters = np.array([self.fom.dt, self.fom.T, self.fom.theta, float(self.fom.nu)])

        for file in files:
            tmp = np.load(file)
            if np.array_equal(parameters, tmp["parameters"]):
                self.fom.Y["supremizer"] = tmp["supremizer"]
                print(f"Loaded {file}")
                return True
        return False

    def compute_supremizer_time_step(self, pressure_snapshot):
                
        A = self.fom.matrix["primal"]["supremizer_system"]
        rhs_matrix = self.fom.matrix["primal"]["supremizer_rhs"]

        pressure_snapshot = np.concatenate(
            (np.zeros((self.fom.dofs["velocity"],)), pressure_snapshot)
        )

        U_supremizer = Function(self.fom.V)  # supremizer function
        U_supremizer.vector().set_local(pressure_snapshot)

        b = U_supremizer.vector().copy()
        rhs_matrix.mult(U_supremizer.vector(), b)

        # Apply boundary conditions
        for bc in self.fom.bc:
            bc.apply(A, b)

        A_sp = scipy.sparse.csr_matrix(
            as_backend_type(A).mat().getValuesCSR()[::-1],
            shape=(
                self.fom.dofs["velocity"] + self.fom.dofs["pressure"],
                self.fom.dofs["velocity"] + self.fom.dofs["pressure"],
            ),
        )

        A_sp = A_sp[: self.fom.dofs["velocity"], : self.fom.dofs["velocity"]]
        rhs = np.array(b)[: self.fom.dofs["velocity"]]

        
        ## Jacobi preconditioner 
        D_inv = scipy.sparse.diags(
            1.0 / A_sp.diagonal()
        ).tocsr()
    
        # # scipy sparse solver
        # solution_old = scipy.sparse.linalg.spsolve(A_sp, rhs)

        solution, exit_code = scipy.sparse.linalg.gmres(
            A_sp,
            rhs,
            M=D_inv,
            tol = 1e-12,
            maxiter = 500,
            restart = 100,
        )

        # difference direct and iterative solver
        # logging.info(f"Error:     {np.linalg.norm(solution - solution_old)}")
        # logging.info(f"Rel Error: {np.linalg.norm(solution - solution_old) / np.linalg.norm(solution)}")


        # _solution = np.concatenate((
        #     solution,
        #     pressure_snapshot[self.fom.dofs["velocity"]:] #np.zeros((self.fom.dofs["pressure"],))
        # ))
        # U_supremizer = Function(self.fom.V)  # lifting function
        # U_supremizer.vector().set_local(_solution)
        # v, p = split(U_supremizer)
        # plt.title("Supremizer function")
        # c = plot(sqrt(dot(v, v)), title="Supremizer velocity magnitude")
        # #c = plot(p, title="Pressure snapshot for supremizer")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        return solution

    def compute_supremizer(self, force_recompute=False):
        if not force_recompute:
            if self.load_supremizer():
                return
        
        logging.info("Start computing supremizer")

        for i, t in enumerate(self.fom.time_points[1:]):
            n = i + 1
            self.fom.Y["supremizer"][:, n] = self.compute_supremizer_time_step(
                self.fom.Y["pressure"][:, n]
            )
            if n % 50 == 0:
                logging.info(f"Computed supremizer at time step {n}")
        self.save_supremizer()

    def solve_primal_time_step_DEBUG(self, v_n_vector, p_n_vector, n):
        # raise SystemExit("EXIT: Adapt solve_primal_time_step to ROM")
        # TODO: work on Newton method for ROM

        self.U = np.zeros(
            (
                self.solution["primal"]["velocity"][:, 0].shape[0]
                + self.solution["primal"]["pressure"][:, 0].shape[0],
            )
        )
        self.U[: self.POD["primal"]["velocity"]["basis"].shape[1]] = self.reduce_vector(
            self.fom.Y["velocity"][:, n], type="primal", quantity="velocity"
        )
        self.U[self.POD["primal"]["velocity"]["basis"].shape[1] :] = self.reduce_vector(
            self.fom.Y["pressure"][:, n], type="primal", quantity="pressure"
        )

        self.U_n = np.zeros(
            (
                self.solution["primal"]["velocity"][:, 0].shape[0]
                + self.solution["primal"]["pressure"][:, 0].shape[0],
            )
        )
        self.U_n[: self.POD["primal"]["velocity"]["basis"].shape[1]] = self.reduce_vector(
            self.fom.Y["velocity"][:, n - 1], type="primal", quantity="velocity"
        )
        self.U_n[self.POD["primal"]["velocity"]["basis"].shape[1] :] = self.reduce_vector(
            self.fom.Y["pressure"][:, n - 1], type="primal", quantity="pressure"
        )

        # Old solution
        # self.U_n = np.concatenate((v_n_vector, p_n_vector))

        # Current solution (iterate)
        # self.U = np.zeros_like(self.U_n) #TODO: undo comments

        # Newton table
        newton_table = rich.table.Table(title=f"Newton solver")
        newton_table.add_column("Step", justify="right")
        newton_table.add_column("Residuum", justify="right")
        newton_table.add_column("Residuum fraction", justify="right")
        newton_table.add_column("Assembled matrix", justify="center")
        newton_table.add_column("Linesearch steps", justify="right")

        # Newton iteration
        system_matrix = None
        system_rhs = self.assemble_system_rhs()
        # print(f"{self.U=}")
        newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

        # print(f"Newton residuum: {newton_residuum}")

        self.DEBUG_RES.append(newton_residuum)

        # # FOR DEBUGGING:
        # self.U = np.zeros_like(self.U_n)
        # system_rhs = self.assemble_system_rhs()
        # newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)
        # print(f"Newton residuum 2: {newton_residuum}")

        # v = self.U[: self.POD["primal"]["velocity"]["basis"].shape[1]]
        # p = self.U[self.POD["primal"]["velocity"]["basis"].shape[1] :]

        v, p = self.fom.U_n.split()

        v.vector().set_local(self.project_vector(system_rhs[:self.POD["primal"]["velocity"]["basis"].shape[1]], type="primal", quantity="velocity"))
        p.vector().set_local(self.project_vector(system_rhs[self.POD["primal"]["velocity"]["basis"].shape[1]:], type="primal", quantity="pressure"))


        # subplot for velocuty and pressure
        # plt.figure(figsize=(8, 5.5))
        # plt.subplot(2, 1, 1)
        # c = plot(sqrt(dot(v, v)), title=f"Velocity @ t={n}")
        # plt.colorbar(c, orientation="horizontal")
        # plt.subplot(2, 1, 2)
        # c = plot(p, title=f"Pressure @ t={n}")
        # plt.colorbar(c, orientation="horizontal")

        # plt.show()

        v = system_rhs[:self.POD["primal"]["velocity"]["basis"].shape[1]]
        p = system_rhs[self.POD["primal"]["velocity"]["basis"].shape[1]:]

        return v, p

        newton_step = 1

        if newton_residuum < self.fom.NEWTON_TOL:
            print(f"Newton residuum: {newton_residuum}")

        # Newton loop
        while newton_residuum > self.fom.NEWTON_TOL and newton_step < self.fom.MAX_N_NEWTON_STEPS:
            old_newton_residuum = newton_residuum

            system_rhs = self.assemble_system_rhs()  

            newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

            if newton_residuum < self.fom.NEWTON_TOL:
                # print(f"Newton residuum: {newton_residuum}")
                newton_table.add_row("-", f"{newton_residuum:.4e}", "-", "-", "-")

                console = rich.console.Console()
                console.print(newton_table)
                break

            if newton_residuum / old_newton_residuum > self.fom.NONLINEAR_RHO:
                system_matrix = self.assemble_system_matrix()

            dU = np.linalg.solve(system_matrix, system_rhs)

            for linesearch_step in range(self.fom.MAX_N_LINE_SEARCH_STEPS):
                self.U += dU

                system_rhs = self.assemble_system_rhs()
                new_newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

                if new_newton_residuum < newton_residuum:
                    break
                else:
                    self.U -= dU

                dU *= self.fom.LINE_SEARCH_DAMPING

            assembled_matrix = newton_residuum / old_newton_residuum > self.fom.NONLINEAR_RHO
            # print(f"Newton step: {newton_step} | Newton residuum: {newton_residuum} | Residuum fraction: {newton_residuum/old_newton_residuum } | Assembled matrix: {assembled_matrix} | Linesearch steps: {linesearch_step}")
            newton_table.add_row(
                str(newton_step),
                f"{newton_residuum:.4e}",
                f"{round(newton_residuum/old_newton_residuum, 4):#.4f}",
                str(assembled_matrix),
                str(linesearch_step),
            )
            newton_step += 1
        else:
            print(
                f"Newton residuum: {newton_residuum}, Newton step: {newton_step}, FAILED TO CONVERGE!!!"
            )

        v = self.U[: self.POD["primal"]["velocity"]["basis"].shape[1]]
        p = self.U[self.POD["primal"]["velocity"]["basis"].shape[1] :]

        # c = plot(sqrt(dot(v, v)), title="Velocity")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        return v, p

    def assemble_system_matrix_velocity(self):
        size_v_N = self.POD["primal"]["velocity"]["basis"].shape[1]

        system_matrix = np.zeros((size_v_N, size_v_N))

        # top left
        ## linear part
        system_matrix = (
            self.velocity_lin_operator_theta 
        )
        ## nonlinearities
        system_matrix += (
            self.fom.theta
            * self.fom.dt
            * self.evaluate_reduced_nonlinearity(vector=self.U[:size_v_N], quantity="velocity")
        )

        return system_matrix
    

    def assemble_system_rhs_velocity(self):
        size_v_N = self.POD["primal"]["velocity"]["basis"].shape[1]
        system_rhs = np.zeros((size_v_N,))

        # current timestep
        # velocity part: -A_v u - theta C_v^l u - theta dt D_v(u)u
        # -A_v u - theta C_v^l u
        system_rhs -= np.dot(self.velocity_lin_operator_theta, self.U[:size_v_N])

        # - theta dt D_v(u)u
        system_rhs -= (
            self.fom.theta
            * self.fom.dt
            * self.twice_evaluate_reduced_nonlinearity(
                vector=self.U, quantity="velocity"
            )
        )
        print(self.twice_evaluate_reduced_nonlinearity(
                vector=self.U, quantity="velocity"
            ).shape)

        # old timestep
        # velocity part: A^{-theta} u_n - (1-theta) dt C_v^l u_n - (1-theta) dt D_v(u_n)u_n
        # A^{-theta} u_n - (1-theta) dt C_v^l u_n
        system_rhs += np.dot(
            self.velocity_lin_operator_one_minus_theta, self.U_n
        )
        # - (1-theta) dt D_v(u_n)u_n
        system_rhs -= (
            (1.0 - self.fom.theta)
            * self.fom.dt
            * self.twice_evaluate_reduced_nonlinearity(
                vector=self.U_n, quantity="velocity"
            )
        )

        # lifting
        system_rhs += self.velocity_lifting_rhs

        return system_rhs
    
    def solve_primal_time_step_velocity(self, v_n_vector):
        self.U_n = v_n_vector
        self.U = np.zeros_like(self.U_n)
        # Newton table
        newton_table = rich.table.Table(title=f"Newton solver")
        newton_table.add_column("Step", justify="right")
        newton_table.add_column("Residuum", justify="right")
        newton_table.add_column("Residuum fraction", justify="right")
        newton_table.add_column("Assembled matrix", justify="center")
        newton_table.add_column("Linesearch steps", justify="right")

        system_matrix = None
        system_rhs = self.assemble_system_rhs_velocity()
        newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

        print(f"Newton residuum: {newton_residuum}")

        newton_trajectory = []

        newton_step = 1

        if newton_residuum < self.fom.NEWTON_TOL:
            print(f"Newton residuum: {newton_residuum}")

        # Newton loop
        while newton_residuum > self.fom.NEWTON_TOL and newton_step < self.fom.MAX_N_NEWTON_STEPS:
            old_newton_residuum = newton_residuum

            system_rhs = self.assemble_system_rhs_velocity()  

            newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

            if newton_residuum < self.fom.NEWTON_TOL:
                # print(f"Newton residuum: {newton_residuum}")
                newton_table.add_row("-", f"{newton_residuum:.4e}", "-", "-", "-")

                console = rich.console.Console()
                console.print(newton_table)
                break

            if newton_residuum / old_newton_residuum > self.fom.NONLINEAR_RHO:
                system_matrix = self.assemble_system_matrix_velocity()

            dU = np.linalg.solve(system_matrix, system_rhs)

            for linesearch_step in range(self.fom.MAX_N_LINE_SEARCH_STEPS):
                self.U += dU

                system_rhs = self.assemble_system_rhs_velocity()
                new_newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

                if new_newton_residuum < newton_residuum:
                    break
                else:
                    self.U -= dU

                dU *= self.fom.LINE_SEARCH_DAMPING


            print(f"New Newton residuum: {new_newton_residuum}")
            newton_trajectory.append(new_newton_residuum)

            assembled_matrix = newton_residuum / old_newton_residuum > self.fom.NONLINEAR_RHO
            # print(f"Newton step: {newton_step} | Newton residuum: {newton_residuum} | Residuum fraction: {newton_residuum/old_newton_residuum } | Assembled matrix: {assembled_matrix} | Linesearch steps: {linesearch_step}")
            newton_table.add_row(
                str(newton_step),
                f"{newton_residuum:.4e}",
                f"{round(newton_residuum/old_newton_residuum, 4):#.4f}",
                str(assembled_matrix),
                str(linesearch_step),
            )
            newton_step += 1
        else:
            print(
                f"Newton residuum: {newton_residuum}, Newton step: {newton_step}, FAILED TO CONVERGE!!!"
            )

        plt.semilogy(newton_trajectory)
        plt.show()
        return self.U


    def solve_primal_time_step(self, v_n_vector, p_n_vector):
        # raise SystemExit("EXIT: Adapt solve_primal_time_step to ROM")
        # TODO: work on Newton method for ROM

        # self.U = np.zeros(solve_primal
        #     (
        #         self.solution["primal"]["velocity"][:, 0].shape[0]
        #         + self.solution["primal"]["pressure"][:, 0].shape[0],
        #     )
        # )

        # Old solution
        self.U_n = np.concatenate((v_n_vector, p_n_vector))

        # Current solution (iterate)
        self.U = np.zeros_like(self.U_n) #TODO: undo comments

        # Newton table
        newton_table = rich.table.Table(title=f"Newton solver")
        newton_table.add_column("Step", justify="right")
        newton_table.add_column("Residuum", justify="right")
        newton_table.add_column("Residuum fraction", justify="right")
        newton_table.add_column("Assembled matrix", justify="center")
        newton_table.add_column("Linesearch steps", justify="right")

        # Newton iteration
        system_matrix = None
        system_rhs = self.assemble_system_rhs()
        # print(f"{self.U=}")
        newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

        print(f"Newton residuum: {newton_residuum}")

        newton_step = 1

        if newton_residuum < self.fom.NEWTON_TOL:
            print(f"Newton residuum: {newton_residuum}")

        # Newton loop
        while newton_residuum > self.fom.NEWTON_TOL and newton_step < self.fom.MAX_N_NEWTON_STEPS:
            old_newton_residuum = newton_residuum

            system_rhs = self.assemble_system_rhs()  

            newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

            if newton_residuum < self.fom.NEWTON_TOL:
                # print(f"Newton residuum: {newton_residuum}")
                newton_table.add_row("-", f"{newton_residuum:.4e}", "-", "-", "-")

                console = rich.console.Console()
                console.print(newton_table)
                break

            if newton_residuum / old_newton_residuum > self.fom.NONLINEAR_RHO:
                system_matrix = self.assemble_system_matrix()

            # try out normal equation for non-symmetric matrices
            
            # dU = np.linalg.solve(system_matrix.T @ system_matrix, system_matrix.T @ system_rhs)

            dU = np.linalg.solve(system_matrix, system_rhs)

            for linesearch_step in range(self.fom.MAX_N_LINE_SEARCH_STEPS):
                self.U += dU

                system_rhs = self.assemble_system_rhs()
                new_newton_residuum = np.linalg.norm(system_rhs, ord=np.Inf)

                if new_newton_residuum < newton_residuum:
                    break
                else:
                    self.U -= dU

                dU *= self.fom.LINE_SEARCH_DAMPING

            assembled_matrix = newton_residuum / old_newton_residuum > self.fom.NONLINEAR_RHO
            # print(f"Newton step: {newton_step} | Newton residuum: {newton_residuum} | Residuum fraction: {newton_residuum/old_newton_residuum } | Assembled matrix: {assembled_matrix} | Linesearch steps: {linesearch_step}")
            newton_table.add_row(
                str(newton_step),
                f"{newton_residuum:.4e}",
                f"{round(newton_residuum/old_newton_residuum, 4):#.4f}",
                str(assembled_matrix),
                str(linesearch_step),
            )
            newton_step += 1
        else:
            print(
                f"Newton residuum: {newton_residuum}, Newton step: {newton_step}, FAILED TO CONVERGE!!!"
            )

        v = self.U[: self.POD["primal"]["velocity"]["basis"].shape[1]]
        p = self.U[self.POD["primal"]["velocity"]["basis"].shape[1] :]

        # c = plot(sqrt(dot(v, v)), title="Velocity")
        # plt.colorbar(c, orientation="horizontal")
        # plt.show()

        return v, p

    def solve_primal_velocity(self):
        self.solution["primal"]["velocity"] = np.zeros(
            (self.POD["primal"]["velocity"]["basis"].shape[1], self.fom.dofs["time"])
        )

        self.solution["primal"]["velocity"][:, 0] = self.reduce_vector(
            self.fom.Y["velocity"][:, 1], type="primal", quantity="velocity"
        )
        i = 0
        t = 0.
        sol_velocity = self.project_vector(
            self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
        ) + self.lifting["velocity"]
        sol_pressure = self.project_vector(
            np.zeros((self.POD["primal"]["pressure"]["basis"].shape[1],)), type="primal", quantity="pressure"
        )
        
        v, p = self.fom.U_n.split()

        self.fom.U_n.vector().set_local(
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

        plt.show()


        self.DEBUG_RES = []

        for i, t in enumerate(self.fom.time_points[1:]):
            print("#-----------------------------------------------#")
            print(f"t = {t:.4f}")
            n = i + 1
            (
                self.solution["primal"]["velocity"][:, n]
            ) = self.solve_primal_time_step_velocity(
                self.solution["primal"]["velocity"][:, n - 1]
            )

            # DEBUG HF: remove lifting to see resiudal
            sol_velocity = self.project_vector(
                self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
            ) + self.lifting["velocity"]
            sol_pressure = self.project_vector(
                np.zeros((self.POD["primal"]["pressure"]["basis"].shape[1],)), type="primal", quantity="pressure"
            )
            
            # if i % 500 == 0:
            #     v, p = self.fom.U_n.split()

            #     self.fom.U_n.vector().set_local(
            #         np.concatenate(
            #             (
            #                 sol_velocity,
            #                 sol_pressure,
            #             )
            #         )
            #     )


            #     # subplot for velocuty and pressure
            #     plt.figure(figsize=(8, 5.5))
            #     plt.subplot(2, 1, 1)
            #     c = plot(sqrt(dot(v, v)), title=f"Velocity @ t={t:.2}")
            #     plt.colorbar(c, orientation="horizontal")
            #     plt.subplot(2, 1, 2)
            #     c = plot(p, title=f"Pressure @ t={t:.2}")
            #     plt.colorbar(c, orientation="horizontal")

            #     plt.show()


    def solve_primal(self):
        self.solution["primal"]["velocity"] = np.zeros(
            (self.POD["primal"]["velocity"]["basis"].shape[1], self.fom.dofs["time"])
        )

        self.solution["primal"]["pressure"] = np.zeros(
            (self.POD["primal"]["pressure"]["basis"].shape[1], self.fom.dofs["time"])
        )

        self.solution["primal"]["velocity"][:, 0] = self.reduce_vector(
            self.fom.Y["velocity"][:, 500], type="primal", quantity="velocity"
        )
   
        self.solution["primal"]["pressure"][:, 0] = self.reduce_vector(
            self.fom.Y["pressure"][:, 500], type="primal", quantity="pressure"
        )
   
        # self.solution["primal"]["velocity"][:, 0] = self.reduce_vector(
        #     +self.lifting["velocity"], type="primal", quantity="velocity"
        # )     
        
        # self.solution["primal"]["pressure"][:, 1] = np.zeros(
        #     (self.POD["primal"]["pressure"]["basis"].shape[1],)
        # )

    
        self.DEBUG_RES = []

        for i, t in enumerate(self.fom.time_points[1:]):
            print("#-----------------------------------------------#")
            print(f"t = {t:.4f}; n = {i+1}")
            n = i + 1
            (
                self.solution["primal"]["velocity"][:, n],
                self.solution["primal"]["pressure"][:, n],
            ) = self.solve_primal_time_step(
                self.solution["primal"]["velocity"][:, n - 1],
                self.solution["primal"]["pressure"][:, n - 1]
            )

            # print("BREAKING ROM LOOP FOR DEBUGGING")
            # break

            # if i % 500 == 0 or i == 0: 

            #     sol_velocity = self.project_vector(
            #         self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
            #     ) + self.lifting["velocity"]
            #     sol_pressure = self.project_vector(
            #         self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
            #     )
                
            #     v, p = self.fom.U_n.split()

            #     self.fom.U_n.vector().set_local(
            #         np.concatenate(
            #             (
            #                 sol_velocity,
            #                 sol_pressure,
            #             )
            #         )
            #     )


            #     # subplot for velocuty and pressure
            #     plt.figure(figsize=(8, 5.5))
            #     plt.subplot(2, 1, 1)
            #     c = plot(sqrt(dot(v, v)), title=f"Velocity @ t={t:.2}")
            #     plt.colorbar(c, orientation="horizontal")
            #     plt.subplot(2, 1, 2)
            #     c = plot(p, title=f"Pressure @ t={t:.2}")
            #     plt.colorbar(c, orientation="horizontal")

            #     plt.show()

        # plt.semilogy(self.DEBUG_RES)
        # plt.show()

    # def reduce_sub_matrix(self,matrix):
    def compute_reduced_nonlinearity(self, type, quantity):
        size_quantity = self.POD[type][quantity]["basis"].shape[1]

        U = TrialFunction(self.fom.V)
        Phi_U = TestFunctions(self.fom.V)

        # Split functions into velocity and pressure components
        v, _ = split(U)
        phi_v, _ = Phi_U

        # Define a function to assemble the i-th reduced tensor
        def assemble_i(i, queue):
            # Set dU as the i-th POD mode
            solution = np.concatenate(
                (
                    self.POD[type][quantity]["basis"][:, i],
                    np.zeros_like(self.POD[type]["pressure"]["basis"][:, 0]),
                )
            )
            dU = Function(self.fom.V)  # POD vector
            dU.vector().set_local(solution)
            dv, _ = split(dU)

            # Assemble the convection term of Navier-Stokes with the i-th POD
            # mode
            a = dot(dot(grad(v), phi_v), dv) * dx

            # Reduce the matrix for the convective term
            reduced_tensor_i = self.reduce_matrix(
                matrix=scipy.sparse.csr_matrix(
                    as_backend_type(assemble(a)).mat().getValuesCSR()[::-1],
                    shape=(
                        self.fom.dofs["velocity"] + self.fom.dofs["pressure"],
                        self.fom.dofs["velocity"] + self.fom.dofs["pressure"],
                    ),
                )[: self.fom.dofs["velocity"], : self.fom.dofs["velocity"]],
                type=type,
                quantity0="velocity",
                quantity1="velocity",
            )
            print("HARDCODED v,v in nonlinearity reduction")

            # Put the assembled reduced tensor in the queue
            queue.put((i, reduced_tensor_i))

        # Use a process pool to assemble each reduced matrix entry of the
        # tensor on a separate process
        queue = Queue()
        processes = [Process(target=assemble_i, args=(i, queue)) for i in range(size_quantity)]
        for process in processes:
            process.start()

        # Get the assembled reduced tensors from the queue
        reduced_tensor = [None for i in range(size_quantity)]
        for _ in range(size_quantity):
            i, reduced_tensor_i = queue.get()
            reduced_tensor[i] = reduced_tensor_i

        # Join the processes to wait for them to finish
        for process in processes:
            process.join()

        return reduced_tensor

    def evaluate_reduced_nonlinearity(self, vector, quantity):
        # a, b, c = self.nonlinearity_tensor[quantity].shape
        # print(a,b,c)

        # output = np.zeros((a,b))

        # for l in range(b):
        #     for i in range(a):
        #         for j in range(c):
        #             output[i,j] += self.nonlinearity_tensor[quantity][i][l,j] * vector[l]
        #             output[i,j] += self.nonlinearity_tensor[quantity][i][j,l] * vector[l]

        # tmp = np.einsum('ijk,j->ik', self.nonlinearity_tensor[quantity], vector) +\
        #        np.einsum('ijk,k->ij', self.nonlinearity_tensor[quantity], vector)

        # print(f"Difference: {tmp - output}")
        # print(f"Difference max: {np.linalg.norm(tmp - output, ord=np.inf)}")
        # return tmp

        return np.einsum("ijk,j->ik", self.nonlinearity_tensor[quantity], vector) + np.einsum(
            "ijk,k->ij", self.nonlinearity_tensor[quantity], vector
        )

    def twice_evaluate_reduced_nonlinearity(self, vector, quantity):
        return np.einsum("ijk,j,k->i", self.nonlinearity_tensor[quantity], vector, vector)

    def compute_deim_snaphots(self):
        self.deim_snapshots = np.zeros(
            (self.fom.dofs["velocity"] + self.fom.dofs["pressure"], len(self.fom.time_points[1:]))
        )

        U = Function(self.fom.V)
        Phi_U = TestFunctions(self.fom.V)

        # Split functions into velocity and pressure components
        v, _ = split(U)
        phi_v, _ = Phi_U

        for i in range(len(self.fom.time_points[1:])):
            # set U as the i-th FOM snapshot
            solution = np.concatenate(
                (self.fom.Y["velocity"][:, i + 1], self.fom.Y["pressure"][:, i + 1])
            )
            U.vector().set_local(solution)

            # assemble nonlinearity snapshot
            a = dot(dot(grad(v), v), phi_v) * dx
            self.deim_snapshots[:, i] = assemble(a).get_local()

    def deim(self):
        """
        Discrete Empirical Interpolation Method
        URL: http://homepage.tudelft.nl/w4u81/MOR/LectureNotes/DEIM_SISC.pdf
        """

        # Compute DEIM projection basis
        (
            deim_basis,
            deim_sigs,
            _,
        ) = scipy.linalg.svd(self.deim_snapshots, full_matrices=False)

        # Plot decay of DEIM singular values
        plt.title("DEIM Singular Values")
        plt.semilogy(deim_sigs)
        plt.show()

        # compute the number of POD modes to be kept
        deim_energy = np.dot(deim_sigs, deim_sigs)
        DEIM_TOL_ENERGY = 1 - 1e-4
        r = 0
        while (np.dot(deim_sigs[0:r], deim_sigs[0:r]) <= deim_energy * DEIM_TOL_ENERGY) and (
            r <= np.shape(deim_sigs)[0]
        ):
            r += 1

        deim_sigs = deim_sigs[0:r]
        # U = [u_0, ..., u_r-1]
        deim_basis = deim_basis[:, 0:r]

        # Compute DEIM interpolation indices
        deim_indices = np.zeros((r,), dtype=int)
        # p_0 = argmax |u_0|
        deim_indices[0] = np.argmax(np.abs(deim_basis[:, 0]))
        for i in range(1, r):
            # Solve (P^T U) c = P^T u_i
            c = np.linalg.solve(
                deim_basis[:, 0:i][deim_indices[0:i], :],  # P^T U
                deim_basis[:, i][deim_indices[0:i]],  # P^T u_i
            )
            # p_i = argmax |u_i - U c|
            deim_indices[i] = np.argmax(np.abs(deim_basis[:, i] - np.dot(deim_basis[:, 0:i], c)))

        return deim_basis, deim_indices

    def update_matrices(self, type):
        """
        "mass": mass matrix
        "laplace": laplace matrix
        "press": pressure matrix
        "div": divergence matrix

        NONLINEARITY: reduction with compute_reduced_nonlinearity
        """

        G_lin = (
            self.fom.matrix[type]["mass"]
            + self.fom.matrix[type]["laplace"]
            + self.fom.matrix[type]["pressure"]
            + self.fom.matrix[type]["div"]
        )
        G_nonlin = self.compute_reduced_nonlinearity(type)

        for key in self.matrix[type].keys():
            self.matrix[type][key] = self.reduce_matrix(self.fom.matrix[type][key], type)

    def run_parent_slab(self):
        execution_time = time.time()
        self.init_POD()

        # update reduced matrices
        self.update_matrices("primal")
        self.update_matrices("dual")
        self.update_matrices("estimate")

        # update reduced rhs
        self.update_rhs("primal")
        self.update_rhs("dual")

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

                # 4. If relative error is too large, then solve primal and dual FOM on
                # time step with largest error
                if estimate["max"] <= self.REL_ERROR_TOL:
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

    def save_vtk(self):
        folder = f"paraview/{self.fom.dt}_{self.fom.T}_{self.fom.theta}_{float(self.fom.nu)}/ROM"

        if not os.path.exists(folder):
            os.makedirs(folder)

        lifting = self.lifting["velocity"]

        for i, t in list(enumerate(self.fom.time_points))[::10]:
            print(f"PLOT {i}-th solution")
            vtk_velocity = File(f"{folder}/velocity_{str(i)}.pvd")
            vtk_pressure = File(f"{folder}/pressure_{str(i)}.pvd")

            # DEBUG HF: remove lifting to see resiudal
            sol_velocity = self.project_vector(
                self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
            ) + self.lifting["velocity"]
            sol_pressure = self.project_vector(
                self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
            )
            v, p = self.fom.U_n.split()

            self.fom.U_n.vector().set_local(
                np.concatenate(
                    (
                        sol_velocity,
                        sol_pressure,
                    )
                )
            )

            vtk_velocity << v
            vtk_pressure << p

            # subplot for velocuty and pressure
            plt.figure(figsize=(8, 5.5))
            plt.subplot(2, 1, 1)
            c = plot(sqrt(dot(v, v)), title=f"Velocity @ t={t:.2}")
            plt.colorbar(c, orientation="horizontal")
            plt.subplot(2, 1, 2)
            c = plot(p, title=f"Pressure @ t={t:.2}")
            plt.colorbar(c, orientation="horizontal")

            plt.show()


    def compute_drag_lift(self):
        offset = 100

        self.drag_force = np.zeros((self.fom.dofs["time"],))
        self.lift_force = np.zeros((self.fom.dofs["time"],))

        for i, t in list(enumerate(self.fom.time_points)):
            sol_velocity = self.project_vector(
                self.solution["primal"]["velocity"][:, i], type="primal", quantity="velocity"
            ) + self.lifting["velocity"]
            sol_pressure = self.project_vector(
                self.solution["primal"]["pressure"][:, i], type="primal", quantity="pressure"
            )
            self.drag_force[i], self.lift_force[i] = self.fom.compute_drag_lift_time_step(sol_velocity, sol_pressure)


        # plot results in subplots 
        # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        # ax[0].plot(self.fom.time_points[offset:], self.drag_force[offset:], label="drag")
        # ax[0].set_xlabel("time")
        # ax[0].set_ylabel("drag")
        # ax[0].grid()
        # ax[1].plot(self.fom.time_points[offset:], self.lift_force[offset:], label="lift")
        # ax[1].set_xlabel("time")
        # ax[1].set_ylabel("lift")
        # ax[1].grid()
        # plt.show()

        # subplot for velocuty and pressure
        # plt.figure(figsize=(8, 5.5))
        plt.subplot(1, 2, 1)
        plt.plot(self.fom.time_points[offset:], self.drag_force[offset:], label="drag - ROM")
        plt.plot(self.fom.time_points[offset:], self.fom.drag_force[offset:], label="drag - FOM")
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(self.fom.time_points[offset:], self.lift_force[offset:], label="lift - ROM")
        plt.plot(self.fom.time_points[offset:], self.fom.lift_force[offset:], label="lift - FOM")
        plt.legend()
        plt.grid()

        # plt.legend()
        # plt.grid()
        # plt.subplot(2, 2, 3)
        # plt.legend()
        # plt.grid()
        # plt.subplot(2, 2, 4)
        # plt.plot(self.fom.time_points[offset:], self.fom.lift_force[offset:], label="lidt - FOM")
        # plt.legend()
        # plt.grid()

        plt.show()

