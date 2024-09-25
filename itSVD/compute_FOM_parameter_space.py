import time

import dolfin
import multiprocessing as mp

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
from dolfin import *

from FOM import FOM
from ROM import ROM

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
# nu = Constant(0.001)    
theta = 0.5
T =  20.0 # 5.  # 10.0
dt = 0.01
n_timesteps = int(T / dt)
# dt = T / n_timesteps


Re = np.arange(50, 200+1, 5)

logging.info(f"Re = {Re}")

def compute_FOM(nu):
    nu = Constant(nu)
    fom = FOM(0, T, dt, theta, nu)
    fom.solve_primal(force_recompute=True)
    fom.save_solution_parallel()
    fom.compute_drag_lift()

logging.info(f"Number cores: {mp.cpu_count()}")

with mp.Pool(mp.cpu_count()) as pool:
    # Convert Reynolds numbers to nu values and compute FOM in parallel
    pool.map(compute_FOM, [(0.1/reynolds) for reynolds in Re])

logging.info("Parallel computation done")

# for fom in FOM:
#     fom.save_solution()
#     fom.save_vtk()
#     fom.compute_drag_lift()

# for reynolds in Re:
#     nu = Constant(0.1/reynolds)
#     compute_FOM(nu)