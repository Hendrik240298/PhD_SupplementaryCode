import time

import dolfin
import multiprocessing as mp
import copy

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
from dolfin import *

import scipy

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


from FOM import FOM
from ROM import ROM

import pickle

import logging
# configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)

def greedy_enrichment(
                      Re,
                      master_ROM,
                      ):

    # ----------- FOM parameters -----------
    # nu = Constant(0.001)    
    theta = 0.5
    T =  20.0 # 5.  # 10.0
    dt = 0.01
    n_timesteps = int(T / dt)
    # dt = T / n_timesteps

    # ----------- ROM parameters -----------
    REL_ERROR_TOL = 1e-2
    MAX_ITERATIONS = 100
    TOTAL_ENERGY = {
        "primal": {
            "velocity": 1,
            "pressure": 1,
        },
    }

    # find where time points is >= 15
    index = np.where(master_ROM.fom.time_points > 18.)[0][0]
    index_end = np.where(master_ROM.fom.time_points > 20)[0][0]


    # since we start at t = 5 this is shifted by 5s
    index_rom_start = np.where(master_ROM.fom.time_points >= 13.)[0][0]
    index_rom_end = np.where(master_ROM.fom.time_points >= 15)[0][0]    

    FOM_dict = {}
    ROM_dict = {}
    for re in Re:
        fom = FOM(0, T, dt, theta, 0.1/re)
        # fom.load_solution_parallel()
        FOM_dict[re] = fom
        ROM_dict[re] = ROM(
            fom,
            REL_ERROR_TOL=REL_ERROR_TOL,
            MAX_ITERATIONS=MAX_ITERATIONS,
            TOTAL_ENERGY=TOTAL_ENERGY,
        )    
        FOM_dict[re].assemble_lifting_matrices(lifting=master_ROM.lifting)
        # FOM Matrices
        FOM_dict[re].assemble_linear_operators()

    STARTING_TIME = 5.0
    STARTING_POINT = np.where(FOM_dict[Re[0]].time_points >= STARTING_TIME)[0][0]

    for re in Re:
        logging.info(f"Re = {re}")
        # fom = FOM(0, T, dt, theta, 0.1/re)
        FOM_dict[re].load_solution_parallel()
        FOM_dict[re].compute_drag_lift()
        ROM_dict[re].compute_supremizer(force_recompute=False)
    
    Re_greedy = Re[0]
    
    FOM_snapshots = {
        "velocity": np.empty((FOM_dict[Re[0]].dofs["velocity"], 0)),
        "pressure": np.empty((FOM_dict[Re[0]].dofs["pressure"], 0)),
        "supremizer": np.empty((FOM_dict[Re[0]].dofs["velocity"], 0)),
    }
    
    FOM_snapshots["velocity"] = FOM_dict[Re_greedy].Y["velocity"][:, STARTING_POINT:]
    FOM_snapshots["pressure"] = FOM_dict[Re_greedy].Y["pressure"][:, STARTING_POINT:]
    FOM_snapshots["supremizer"] = ROM_dict[Re_greedy].fom.Y["supremizer"][:, STARTING_POINT:]
    
    for i in range(FOM_snapshots["velocity"].shape[1]):
        FOM_snapshots["velocity"][:,i] -= master_ROM.lifting["velocity"]
        FOM_snapshots["supremizer"][:,i] -= master_ROM.lifting["velocity"]
    
    U = {
        "velocity": np.empty((FOM_dict[Re_greedy].dofs["velocity"], 0)),
        "pressure": np.empty((FOM_dict[Re_greedy].dofs["pressure"], 0)),
        "supremizer": np.empty((FOM_dict[Re_greedy].dofs["velocity"], 0)),
    }
    
    U["velocity"], S_v, _ = scipy.linalg.svd(FOM_snapshots["velocity"], full_matrices=False)
    U["pressure"], S_p, _ = scipy.linalg.svd(FOM_snapshots["pressure"], full_matrices=False)
    U["supremizer"], S_s, _ = scipy.linalg.svd(FOM_snapshots["supremizer"], full_matrices=False)
    
    error = {}
    
    for re in Re:
        for quantity in ["pressure", "velocity", "supremizer"]:
            ROM_dict[re].POD["primal"][quantity]["basis"] = U[quantity][:, :5]
            ROM_dict[re].compute_reduced_matrices()
            ROM_dict[re].solve_primal()
            ROM_dict[re].compute_drag_lift()
            error[re] = np.linalg.norm(FOM_dict[re].drag_force[index-2:index_end]-ROM_dict[re].drag_force[index_rom_start-2:index_rom_end])
    
