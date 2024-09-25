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


BUNCH_SIZES = 100, [1, 5, 10, 25, 50, 100]
TEST_ITERATIONS = 5

timings = []

for bunch_size in BUNCH_SIZES:
    logging.info(f"\nStarting itSVD computation with bunch size = {bunch_size}")
    timings_bunch = {
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
    
    for i in range(TEST_ITERATIONS):
        itSVD_instance = itSVD(
            fom,
            TOTAL_ENERGY=TOTAL_ENERGY,
            BUNCH_SIZE=bunch_size
        )

        iterator = FOM_snapshots["velocity"].shape[1] -2 

        # velocity
        for i in range(iterator):
            itSVD_instance.compute_iteration(FOM_snapshots["velocity"][:, i], type="primal", quantity="velocity")

        # pressure
        for i in range(iterator):
            itSVD_instance.compute_iteration(FOM_snapshots["pressure"][:, i], type="primal", quantity="pressure")

        # saving timings
        for key, value in itSVD_instance.timings["velocity"].items():
            timings_bunch["velocity"][key] += value/TEST_ITERATIONS
        
        for key, value in itSVD_instance.timings["pressure"].items():
            timings_bunch["pressure"][key] += value/TEST_ITERATIONS

        logging.info(f"itSVD POD in {itSVD_instance.POD['primal']['velocity']['basis'].shape[1]} (velo) + {itSVD_instance.POD['primal']['pressure']['basis'].shape[1]} (press)")

        logging.info(f"itSVD computation in {itSVD_instance.timings['velocity']['total']} (velo) + {itSVD_instance.timings['pressure']['total']} (press) seconds")

    timings.append(timings_bunch)
    
# print timings
for i, bunch_size in enumerate(BUNCH_SIZES):
    logging.info(f"bunch size = {bunch_size}")
    logging.info("Timings - velo:")
    for key, value in timings[i]["velocity"].items():
        logging.info(f"{key} = {value}")
    logging.info("Timings - press:")
    for key, value in timings[i]["pressure"].items():
        logging.info(f"{key} = {value}")
    logging.info("\n")

# save timings dict with pickle 
with open('results/itSVD/itSVD_timings_bunch_sizes.pkl', 'wb') as f:
    pickle.dump(timings, f)
    
