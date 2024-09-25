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
fom.compute_drag_lift()

rom = ROM(
    fom,
    REL_ERROR_TOL=REL_ERROR_TOL,
    MAX_ITERATIONS=MAX_ITERATIONS,
    TOTAL_ENERGY=TOTAL_ENERGY,
)

###########
# OFFLINE #
###########

# SUPREMIZER
rom.compute_supremizer(force_recompute=False)

# LIFTING
rom.compute_lifting_function(force_recompute=False)
# rom.lifting["velocity"] *= 0

# centering with lifting function
rom.subtract_lifting_function()
fom.assemble_lifting_matrices(lifting=rom.lifting)

# exit()

# FOM Matrices
fom.assemble_linear_operators()

# POD
rom.init_POD()

# compute reduced matrices
print("starting matrix reduction")
rom.compute_reduced_matrices()
print("finished matrix reduction")

# rom.reduce_matrix(fom.matrix["primal"]["mass"], type="primal", quantity0="velocity",quantity1="velocity")
rom.solve_primal()
rom.compute_drag_lift()
# rom.save_vtk()
# TODO: Use in the future and compare computational cost with rank 3 tensor
# # DEIM
# rom.compute_deim_snaphots()
# rom.deim()

##########
# ONLINE #
##########

# ROM
# TODO

# start_time = time.time()
# rank3_tensor = rom.compute_reduced_nonlinearity(type="primal")
# print("--- %s seconds for rank 3 ---" % (time.time() - start_time))
# rom.test_reduced_nonlinearity(type="primal")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.semilogy(rom.POD["primal"]["velocity"]["sigs"], label="velocity")
ax1.grid(True)  # Add a grid to the first subplot
# Set the x-axis limits of the first subplot
ax1.set_xlim(0, len(rom.POD["primal"]["velocity"]["sigs"]) - 1)
ax1.legend()  # Activate the legend for the first subplot

ax2.semilogy(rom.POD["primal"]["pressure"]["sigs"], label="pressure")
ax2.grid(True)  # Add a grid to the second subplot
# Set the x-axis limits of the second subplot
ax2.set_xlim(0, len(rom.POD["primal"]["pressure"]["sigs"]) - 1)
ax2.legend()  # Activate the legend for the second subplot

ax3.semilogy(rom.POD["primal"]["supremizer"]["sigs"], label="supremizer")
ax3.grid(True)
ax3.set_xlim(0, len(rom.POD["primal"]["supremizer"]["sigs"]) - 1)
ax3.legend()


plt.show()
