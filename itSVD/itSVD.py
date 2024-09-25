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


class itSVD:
    def __init__(
        self,
        fom,
        REL_ERROR_TOL=1e-2,
        TOTAL_ENERGY={
            "primal": {"velocity": 1, "pressure": 1},
        },
        BUNCH_SIZE=1,
    ):
        self.fom = fom
        self.REL_ERROR_TOL = REL_ERROR_TOL

        self.BUNCH_SIZE = BUNCH_SIZE

        self.POD = {
            "primal": {
                "velocity": {
                    "basis": np.empty((self.fom.dofs["velocity"], 0)),
                    "time_basis": np.empty((0, 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.zeros((self.fom.dofs["velocity"], self.BUNCH_SIZE)),
                    "bunch_size": self.BUNCH_SIZE,
                    "bunch_counter": 0,
                    "TOL_ENERGY": TOTAL_ENERGY["primal"]["velocity"],
                },
                "pressure": {
                    "basis": np.empty((self.fom.dofs["pressure"], 0)),
                    "time_basis": np.empty((0, 0)),
                    "sigs": None,
                    "energy": 0.0,
                    "bunch": np.zeros((self.fom.dofs["pressure"], self.BUNCH_SIZE)),
                    "bunch_size": self.BUNCH_SIZE,
                    "bunch_counter": 0,
                    "TOL_ENERGY": TOTAL_ENERGY["primal"]["pressure"],
                },
            },
        }
        
        self.timings = {
            "velocity": {
                "expand": 0.0,
                "SVD": 0.0,
                "QR": 0.0,
                "rank": 0.0,
                "prep": 0.0,
                "build_comps": 0.0,
                "update_U": 0.0, 
                "update_V": 0.0,
                "update_S" : 0.0,
                "orthogonality": 0.0, 
                "total": 0.0,
            },
            "pressure": {
                "expand": 0.0,
                "SVD": 0.0,
                "QR": 0.0,
                "rank": 0.0,
                "prep": 0.0,
                "build_comps": 0.0,
                "update_U": 0.0, 
                "update_V": 0.0,
                "update_S" : 0.0,
                "orthogonality": 0.0, 
                "total": 0.0,
            },
        }
    
    def evaluate(self, type, quantity, index): 
        
        
        SigmaXVT= np.dot(
            np.diag(self.POD[type][quantity]["sigs"]), 
            self.POD[type][quantity]["time_basis"][index, :].T
            )

        UXSigmaXVT = np.dot(
            self.POD[type][quantity]["basis"],
            SigmaXVT
            )
        
        return UXSigmaXVT
    
    
    def compute_full_SVD(self, type, quantity, matrix):
        '''
        for validation and debugging purposes
        '''
        U, S, V = scipy.linalg.svd(matrix, full_matrices=False)
        
        V = V.T
        
        self.POD[type][quantity]["energy"] = np.dot(S, S)
        
        r = 0   
        while (
            np.dot(S[0:r], S[0:r])
            <= self.POD[type][quantity]["energy"] * self.POD[type][quantity]["TOL_ENERGY"]
        ) and (r < np.shape(S)[0]):
            r += 1  
            
        # r = np.shape(S)[0]
            
        self.POD[type][quantity]["sigs"] = S[0:r]
        self.POD[type][quantity]["basis"] = U[:, 0:r]
        self.POD[type][quantity]["time_basis"] = V[:, 0:r]
            
    def compute_iteration(self, snapshot, type, quantity, empty_bunch_matrix=False):
        # type is either "primal" or "dual"
        # empty_bunch_matrix to empty bunch matrix even if not full
        # self.POD[type][quantity]["bunch"] = np.hstack(
        #     (self.POD[type][quantity]["bunch"], snapshot.reshape(-1, 1))
        # )
        
        tic_total = time.time()

        tic = time.time()
        self.POD[type][quantity]["bunch"][:, self.POD[type][quantity]["bunch_counter"]] = snapshot
        self.POD[type][quantity]["bunch_counter"] += 1
        toc = time.time()
        self.timings[quantity]["expand"] += toc - tic


        tic = time.time()
        # add energy of new snapshot to total energy
        self.POD[type][quantity]["energy"] += np.dot((snapshot), (snapshot))
        toc = time.time()
        self.timings[quantity]["rank"] += toc - tic

        # check bunch_matrix size to decide if to update POD
        if self.POD[type][quantity]["bunch_counter"] == self.POD[type][quantity]["bunch_size"] or empty_bunch_matrix:            
            # initialize POD with first bunch matrix
            if self.POD[type][quantity]["basis"].shape[1] == 0:
                tic = time.time()
                (
                    self.POD[type][quantity]["basis"],
                    self.POD[type][quantity]["sigs"],
                    self.POD[type][quantity]["time_basis"],
                ) = scipy.linalg.svd(self.POD[type][quantity]["bunch"], full_matrices=False)
                # retranspose time_basis
                self.POD[type][quantity]["time_basis"] = self.POD[type][quantity]["time_basis"].T
                toc = time.time()
                self.timings[quantity]["SVD"] += toc - tic
                # compute the number of POD modes to be kept
                
                tic = time.time()
                r = 0
                while (
                    np.dot(
                        self.POD[type][quantity]["sigs"][0:r], self.POD[type][quantity]["sigs"][0:r]
                    )
                    <= self.POD[type][quantity]["energy"] * self.POD[type][quantity]["TOL_ENERGY"]
                ) and (r <= np.shape(self.POD[type][quantity]["sigs"])[0]):
                    r += 1
                toc = time.time()
                self.timings[quantity]["rank"] += toc - tic
   
                tic = time.time()
                self.POD[type][quantity]["sigs"] = self.POD[type][quantity]["sigs"][0:r]
                toc = time.time()
                self.timings[quantity]["update_S"] += toc - tic
                
                tic = time.time()
                self.POD[type][quantity]["basis"] = self.POD[type][quantity]["basis"][:, 0:r]
                toc = time.time()
                self.timings[quantity]["update_U"] += toc - tic
                
                tic = time.time()
                self.POD[type][quantity]["time_basis"] = self.POD[type][quantity]["time_basis"][:, 0:r]
                toc = time.time()
                self.timings[quantity]["update_V"] += toc - tic
            # update POD with  bunch matrix
            else:
                tic = time.time()
                M = np.dot(self.POD[type][quantity]["basis"].T, self.POD[type][quantity]["bunch"])
                P = self.POD[type][quantity]["bunch"] - np.dot(self.POD[type][quantity]["basis"], M)
                toc = time.time()
                self.timings[quantity]["prep"] += toc - tic
                
                tic = time.time()
                Q_p, R_p = scipy.linalg.qr(P, mode="economic", check_finite=False, overwrite_a=True)
                toc = time.time()
                self.timings[quantity]["QR"] += toc - tic


                tic = time.time()
                Q_q = np.hstack((self.POD[type][quantity]["basis"], Q_p))

                nb_sigs = self.POD[type][quantity]["sigs"].shape[0]
                bunch_size = self.POD[type][quantity]["bunch_size"]

                K = np.zeros(
                    (
                        nb_sigs + bunch_size,
                        nb_sigs + bunch_size,
                    )
                )

                # put sigs on upper diagonal of K 
                np.fill_diagonal(K[:nb_sigs, :nb_sigs], self.POD[type][quantity]["sigs"])

                # put M on the right of sigs
                K[:nb_sigs, nb_sigs:] = M
                
                # put R_p on the bottom right of K
                K[nb_sigs:, nb_sigs:] = R_p


                # S0 = np.vstack(
                #     (
                #         np.diag(self.POD[type][quantity]["sigs"]),
                #         np.zeros((np.shape(R_p)[0], np.shape(self.POD[type][quantity]["sigs"])[0])),
                #     )
                # )
                # MR_p = np.vstack((M, R_p))
                # K = np.hstack((S0, MR_p))
                toc = time.time()
                self.timings[quantity]["build_comps"] += toc - tic

                tic = time.time()
                # check the orthogonality of Q_q heuristically
                if np.abs(np.inner(Q_q[:, 0], Q_q[:, -1])) >= 1e-10:
                    Q_q, R_q = scipy.linalg.qr(Q_q, mode="economic", check_finite=False, overwrite_a=True)
                    K = np.matmul(R_q, K)
                toc = time.time()
                self.timings[quantity]["orthogonality"] += toc - tic
                
                tic = time.time()
                # inner SVD of K
                U_k, S_k, V_k = scipy.linalg.svd(K, full_matrices=False)
                # retranspose V_k
                V_k = V_k.T
                toc = time.time()   
                self.timings[quantity]["SVD"] += toc - tic
                
                tic = time.time()
                # compute the number of POD modes to be kept
                r = self.POD[type][quantity]["basis"].shape[1]

                while (
                    np.dot(S_k[0:r], S_k[0:r])
                    <= self.POD[type][quantity]["energy"] * self.POD[type][quantity]["TOL_ENERGY"]
                ) and (r < np.shape(S_k)[0]):
                    r += 1

                toc = time.time()
                self.timings[quantity]["rank"] += toc - tic


                tic = time.time()
                self.POD[type][quantity]["sigs"] = S_k[0:r]
                toc = time.time()
                self.timings[quantity]["update_S"] += toc - tic
                tic = time.time()
                self.POD[type][quantity]["basis"] = np.dot(Q_q, U_k[:, 0:r])
                toc = time.time()
                self.timings[quantity]["update_U"] += toc - tic
                
                tic = time.time()
                # extend time basis by ones on the diagonal (as many as the number of new snapshots, aka bunch_size), thus R^nxn -> R^(nxb)x(nxb)
                # 1. enlarge time basis by b zero columns and rows
                self.POD[type][quantity]["time_basis"] = np.hstack((self.POD[type][quantity]["time_basis"], np.zeros((self.POD[type][quantity]["time_basis"].shape[0], self.POD[type][quantity]["bunch_size"]))))
                self.POD[type][quantity]["time_basis"] = np.vstack((self.POD[type][quantity]["time_basis"], np.zeros((self.POD[type][quantity]["bunch_size"], self.POD[type][quantity]["time_basis"].shape[1]))))

                # 2. add ones on the diagonal
                self.POD[type][quantity]["time_basis"][-self.POD[type][quantity]["bunch_size"]:, -self.POD[type][quantity]["bunch_size"]:] = np.eye(self.POD[type][quantity]["bunch_size"])
                
                # matmul with V_k to get the new time_basis
                self.POD[type][quantity]["time_basis"] = np.dot(self.POD[type][quantity]["time_basis"],V_k[:,:r])
                toc = time.time()
                self.timings[quantity]["update_S"] += toc - tic
                
            # empty bunch matrix after update
            self.POD[type][quantity]["bunch"] = np.zeros((self.fom.dofs[quantity], self.BUNCH_SIZE))
            self.POD[type][quantity]["bunch_counter"] = 0
            
        toc_total = time.time()
        self.timings[quantity]["total"] += toc_total - tic_total