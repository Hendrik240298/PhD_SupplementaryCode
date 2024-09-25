# to get a reference timing for the SVD algorithm

import time

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
from dolfin import *

import pickle

from FOM import FOM
from ROM import ROM

from itSVD import itSVD

import logging
# configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)

# Turn off interactive mode
plt.ioff()

# from os import environ
# environ['OMP_NUM_THREADS'] = '32'

# ---------- FEniCS parameters ---------
parameters["reorder_dofs_serial"] = False
set_log_active(False) # turn off FEniCS logging
# ----------- FOM parameters -----------
nu = Constant(0.001)
theta = 0.5
T =  20. # 5.  # 10.0
dt = 0.01
n_timesteps = int(T / dt)
# dt = T / n_timesteps

# ----------- ROM parameters -----------
REL_ERROR_TOL = 1e-2
MAX_ITERATIONS = 100
TOTAL_ENERGY = {
    "primal": {
        "velocity": 1 - 1e-6,
        "pressure": 1 - 1e-6,
    },
}

fom = FOM(0, T, dt, theta, nu)
fom.solve_primal(force_recompute=False)

# fom.load_solution_parallel()
# fom.compute_drag_lift()


FOM_snapshots = {
    "velocity": np.empty((fom.dofs["velocity"], 0)),
    "pressure": np.empty((fom.dofs["pressure"], 0)),
}


FOM_snapshots["velocity"] = fom.Y["velocity"] 
FOM_snapshots["pressure"] = fom.Y["pressure"]


iterator = FOM_snapshots["velocity"].shape[1] -2 







timings_velo = []
timings_press = []

for i in range(5):
    logging.info(f"i = {i}")

    # velocity
    tic = time.time()

    U,S,V = np.linalg.svd(fom.Y["velocity"], full_matrices=False) 

    toc = time.time()    

    logging.info(f"Time for SVD of bunch_matrix['velocity'] = {toc - tic}")

    timings_velo.append(toc - tic)

    # pressure
    tic = time.time()

    U,S,V = np.linalg.svd(fom.Y["pressure"], full_matrices=False)
    toc = time.time()

    logging.info(f"Time for SVD of bunch_matrix['pressure'] = {toc - tic}")

    timings_press.append(toc - tic)

timings = {
    "velocity": np.mean(timings_velo),
    "pressure": np.mean(timings_press),
}


# save timings dict with pickle 
with open('results/itSVD/pure_SVD_timings.pkl', 'wb') as f:
    pickle.dump(timings, f)