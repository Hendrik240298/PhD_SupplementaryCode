import time

import dolfin

# from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import rich.console
import rich.table
from dolfin import *

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
        "velocity": 1 - 1e-5,
        "pressure": 1 - 1e-5,
    },
}

fom = FOM(0, T, dt, theta, nu)
fom.solve_primal(force_recompute=False)
fom.compute_drag_lift()

# fom.load_solution_parallel()
# fom.compute_drag_lift()


FOM_snapshots = {
    "velocity": np.empty((fom.dofs["velocity"], 0)),
    "pressure": np.empty((fom.dofs["pressure"], 0)),
}


FOM_snapshots["velocity"] = fom.Y["velocity"] 
FOM_snapshots["pressure"] = fom.Y["pressure"]

itSVD = itSVD(
    fom,
    TOTAL_ENERGY=TOTAL_ENERGY,
    BUNCH_SIZE=10,
)



logging.info("Starting POD (SVD) computation")
# build and init POD bases 

logging.info(f"bunch size = {itSVD.POD['primal']['velocity']['bunch_size']} and #snapshots = {FOM_snapshots['velocity'].shape[1]}")    # loop over columns of FOM_snapshots 

iterator = FOM_snapshots["velocity"].shape[1] -2 #500

tic = time.time()
for i in range(iterator):
    itSVD.compute_iteration(FOM_snapshots["velocity"][:, i], type="primal", quantity="velocity")
toc = time.time()
time_velo = toc - tic

logging.info(f"velocity SVD done")

tic = time.time()
for i in range(iterator):
    itSVD.compute_iteration(FOM_snapshots["pressure"][:, i], type="primal", quantity="pressure")
toc = time.time()
time_press = toc - tic

logging.info(f"POD (SVD) sizes: velocity = {itSVD.POD['primal']['velocity']['basis'].shape[1]}, pressure = {itSVD.POD['primal']['pressure']['basis'].shape[1]} \n")

logging.info(f"POD (SVD) computation took {time_velo} (velo) + {time_press} (press) seconds")

# print timings
sum = 0
logging.info("Timings - velo:")
for key, value in itSVD.timings["velocity"].items():
    logging.info(f"{key} = {value}")
    if key != "total":
        sum += value    
logging.info(f"Total time = {sum} \n")

sum = 0
logging.info("Timings - press:")
for key, value in itSVD.timings["pressure"].items():
    logging.info(f"{key} = {value}")
    if key != "total":
        sum += value            
logging.info(f"Total time = {sum} \n")


# ===========================
# Evaluation
# ===========================

# 1. CF 
itSVD_FOM = FOM(0, T, dt, theta, nu)

# for i in range(itSVD_FOM.dofs["time"] - 2):
for i in range(iterator):
    # i+1 due to zero vector in the beginning
    itSVD_FOM.Y["velocity"][:, i+1] = itSVD.evaluate("primal", "velocity", i)
    itSVD_FOM.Y["pressure"][:, i+1] = itSVD.evaluate("primal", "pressure", i)

itSVD_FOM.compute_drag_lift()

# compare itSVD and FOM drag and lift coefficients in plot 
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(fom.time_points, fom.drag_force, label="FOM")
plt.plot(itSVD_FOM.time_points, itSVD_FOM.drag_force, label="itSVD")
plt.legend()
plt.grid()
plt.ylim([3.14, 3.23])
plt.title("Drag Force")

plt.subplot(1, 3, 2)
plt.plot(fom.time_points, fom.lift_force, label="FOM")
plt.plot(itSVD_FOM.time_points, itSVD_FOM.lift_force, label="itSVD")
plt.legend()
plt.grid()
# plt.ylim([2.4, 2.55])
plt.title("Lift Force")

plt.subplot(1, 3, 3)
plt.plot(fom.time_points, fom.press_diff, label="FOM")
plt.plot(itSVD_FOM.time_points, itSVD_FOM.press_diff, label="itSVD")
plt.legend()
plt.ylim([2.4, 2.55])
plt.grid()
plt.title("Pressure Difference")

plt.show()

plt.savefig(f"plots/compare_FOM_itSVD_CF_fine_mesh.png")