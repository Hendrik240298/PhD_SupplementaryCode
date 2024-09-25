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



# ---------- FEniCS parameters ---------
parameters["reorder_dofs_serial"] = False
set_log_active(False) # turn off FEniCS logging
# ----------- FOM parameters -----------
nu = Constant(0.001)
theta = 0.5
T =  20. # 5.  # 10.0
dt = 0.01
n_timesteps = int(T / dt)

fom = FOM(0, T, dt, theta, nu)
fom.solve_primal(force_recompute=False)
fom.compute_drag_lift()

# load results/itSVD/itSVD_cost_functionals_energy_content.pkl
with open("results/itSVD/itSVD_cost_functionals_energy_content.pkl", "rb") as f:
    cost_functionals = pickle.load(f)
    
# load results/itSVD/itSVD_timings_energy_content.pkl
with open("results/itSVD/itSVD_timings_energy_content.pkl", "rb") as f:
    timings = pickle.load(f)
    
    
    
time_start_SVD = 1
time_end_SVD = -1

plt.rcParams['text.usetex'] = True

FONT_SIZE_AXIS = 15
FONT_LABEL_SIZE = 13
FONT_SIZE_AXIS_NUMBER = 13



# figure with 2x4 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# plot drag functionals
for i in range(2):
    # FOM
    axs[0, i].plot(
        fom.time_points,
        fom.drag_force, 
        label="FOM",
        linewidth=3,
        )
    
    # ROM
    axs[0, i].plot(
        fom.time_points[0:-2], 
        cost_functionals[i+2]["drag"][1:-1], 
        label="itSVD",
        linestyle=':',
        linewidth=2,
        color="red",
        )
    
    axs[0, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
    axs[0, i].set_ylabel("drag", fontsize = FONT_SIZE_AXIS)
    axs[0, i].grid()
    axs[0, i].legend(fontsize = FONT_LABEL_SIZE)
    # set y axis limits to 3.1 and 3.19
    axs[0, i].set_ylim(3.14, 3.23)
    axs[0, i].set_xlim(15, 20)
    axs[0, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)  # Set tick parameters


# plot error to fom.drag_force
for i in range(2):
    axs[1, i].plot(fom.time_points[0:-2], 
                   np.abs(fom.drag_force[0:-2] - cost_functionals[i+2]["drag"][1:-1])/ np.maximum(np.abs(fom.lift_force[0:-2]),1e-12),
                   )
    axs[1, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
    axs[1, i].set_ylabel("relative error", fontsize = FONT_SIZE_AXIS)
    # log scale
    axs[1, i].set_yscale("log")
    axs[1, i].grid()
    axs[1, i].set_xlim(0, 20)
    axs[1, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)


plt.savefig(f"plots/itSVD/itSVD_drag_energy_contents.pdf")

plt.show()


# figure with 2x4 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# plot lift functionals
for i in range(2):
    # FOM
    axs[0, i].plot(
        fom.time_points, 
        fom.lift_force, 
        label="FOM",
        linewidth=3,
        )
    # ROM
    axs[0, i].plot(
        fom.time_points[0:-2], 
        cost_functionals[i+2]["lift"][1:-1],
        label="itSVD",
        linestyle=':',
        linewidth=2,
        color="red",
        )
    axs[0, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
    axs[0, i].set_ylabel("lift", fontsize = FONT_SIZE_AXIS)
    axs[0, i].grid()
    axs[0, i].legend(fontsize = FONT_LABEL_SIZE)
    # set y axis limits to 3.1 and 3.19
    # axs[0, i].set_ylim(3.14, 3.23)
    axs[0, i].set_xlim(15, 20)
    axs[0, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)  # Set tick parameters


# plot error to fom.lift_force
for i in range(2):
    axs[1, i].plot(
        fom.time_points[0:-2],
        np.abs(fom.lift_force[0:-2] - cost_functionals[i+2]["lift"][1:-1])/ np.maximum(np.abs(fom.lift_force[0:-2]),1e-12),
        )
    axs[1, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
    axs[1, i].set_ylabel("relative error", fontsize = FONT_SIZE_AXIS)
    # log scale
    axs[1, i].set_yscale("log")
    axs[1, i].grid()
    axs[1, i].set_xlim(0, 20)
    axs[1, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)



plt.savefig(f"plots/itSVD/itSVD_lift_energy_contents.pdf")

plt.show()


# open results/itSVD/itSVD_timings_energy_content.pkl and print timings
with open("results/itSVD/itSVD_timings_energy_content.pkl", "rb") as f:
    timings = pickle.load(f)
    
# print timings
for i in range(4):
    logging.info(f"Total velo: {timings[i]['velocity']['total'] - timings[i]['velocity']['expand']}")
    logging.info(f"Total press: {timings[i]['pressure']['total'] - timings[i]['pressure']['expand']}")
# # figure with 2x4 subplots
# fig, axs = plt.subplots(2, 4, figsize=(30, 10))

# # plot drag functionals
# for i in range(4):
#     # FOM
#     axs[0, i].plot(
#         fom.time_points,
#         fom.drag_force, 
#         label="FOM",
#         linewidth=3,
#         )
    
#     # ROM
#     axs[0, i].plot(
#         fom.time_points[0:-2], 
#         cost_functionals[i]["drag"][1:-1], 
#         label="itSVD",
#         linestyle=':',
#         linewidth=2,
#         color="red",
#         )
    
#     axs[0, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
#     axs[0, i].set_ylabel("drag", fontsize = FONT_SIZE_AXIS)
#     axs[0, i].grid()
#     axs[0, i].legend(fontsize = FONT_LABEL_SIZE)
#     # set y axis limits to 3.1 and 3.19
#     axs[0, i].set_ylim(3.14, 3.23)
#     axs[0, i].set_xlim(15, 20)
#     axs[0, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)  # Set tick parameters


# # plot error to fom.drag_force
# for i in range(4):
#     axs[1, i].plot(fom.time_points[0:-2], 
#                    np.abs(fom.drag_force[0:-2] - cost_functionals[i]["drag"][1:-1])/ np.maximum(np.abs(fom.lift_force[0:-2]),1e-12),
#                    )
#     axs[1, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
#     axs[1, i].set_ylabel("relative error", fontsize = FONT_SIZE_AXIS)
#     # log scale
#     axs[1, i].set_yscale("log")
#     axs[1, i].grid()
#     axs[1, i].set_xlim(0, 20)
#     axs[1, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)


# plt.savefig(f"plots/itSVD/itSVD_drag_energy_contents.pdf")

# plt.show()


# # figure with 2x4 subplots
# fig, axs = plt.subplots(2, 4, figsize=(30, 10))

# # plot lift functionals
# for i in range(4):
#     # FOM
#     axs[0, i].plot(
#         fom.time_points, 
#         fom.lift_force, 
#         label="FOM",
#         linewidth=3,
#         )
#     # ROM
#     axs[0, i].plot(
#         fom.time_points[0:-2], 
#         cost_functionals[i]["lift"][1:-1],
#         label="itSVD",
#         linestyle=':',
#         linewidth=2,
#         color="red",
#         )
#     axs[0, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
#     axs[0, i].set_ylabel("lift", fontsize = FONT_SIZE_AXIS)
#     axs[0, i].grid()
#     axs[0, i].legend(fontsize = FONT_LABEL_SIZE)
#     # set y axis limits to 3.1 and 3.19
#     # axs[0, i].set_ylim(3.14, 3.23)
#     axs[0, i].set_xlim(15, 20)
#     axs[0, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)  # Set tick parameters


# # plot error to fom.lift_force
# for i in range(4):
#     axs[1, i].plot(
#         fom.time_points[0:-2],
#         np.abs(fom.lift_force[0:-2] - cost_functionals[i]["lift"][1:-1])/ np.maximum(np.abs(fom.lift_force[0:-2]),1e-12),
#         )
#     axs[1, i].set_xlabel("$t \,[s]$", fontsize = FONT_SIZE_AXIS)
#     axs[1, i].set_ylabel("relative error", fontsize = FONT_SIZE_AXIS)
#     # log scale
#     axs[1, i].set_yscale("log")
#     axs[1, i].grid()
#     axs[1, i].set_xlim(0, 20)
#     axs[1, i].tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS_NUMBER)



# plt.savefig(f"plots/itSVD/itSVD_lift_energy_contents.pdf")

# plt.show()