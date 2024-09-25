import argparse
import time
import itertools

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
from fenics import *
from petsc4py import PETSc
from sklearn.metrics import confusion_matrix

from FOM import FOM
from greedy import Greedy
from iROM import iROM

EPS = 1e-14


# ---------- Parameter file ------------
# specify yaml config file ove command line python3 main_greedy.py
# PATH_config_file
parser = argparse.ArgumentParser(description="Input file to specify the problem.")
parser.add_argument("yaml_config", nargs="?", help="Path/Name to the YAML config file")

# parse the arguments
args = parser.parse_args()

# ATTENTION: No sanity check for yaml config exists yet!
if args.yaml_config is None:
    print("No YAML config file was specified. Thus standard config 'standard.yaml' is used.")
    config_file = "config/standard.yaml"
else:
    config_file = args.yaml_config

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# ----------- FOM parameters -----------
# start time
t = config["FOM"]["start_time"]
# end time
T = config["FOM"]["end_time"]
# time step size
dt = config["FOM"]["dt"]
# defining the mesh
nx = ny = config["FOM"]["mesh_size"]

# ----------- ROM parameters -----------
# REL_ERROR_TOL = 1e-2
# MAX_ITERATIONS = 200

TOTAL_ENERGY = {
    "primal": 1 - config["ROM"]["total_energy"]["primal"],
    "dual": 1 - config["ROM"]["total_energy"]["dual"],
}
# we have a parameter space P, its surrogate P_h and an living parameter
# parameter
parameter = np.random.uniform(0.1, 2.0, 16)
# np.concatenate([np.ones(4) * 2., np.ones(4) * 2., np.ones(4) * 2,
# np.ones(4) * 2.]) #np.ones(16) *
# config["Greedy"]["surrogate"]["min_value"]  # np.random.uniform(0.01,
# 2., 1)
initial_parameter = np.ones(16) * 0.5

# %% ---------------- FOM -----------------
fom = FOM(
    nx,
    ny,
    t,
    T,
    dt,
    parameter=initial_parameter,
    save_directory=config["Infrastructure"]["save_directory"],
)
fom.assemble_system(force_recompute=False)
# start_time = time.time()
# fom.solve_primal(force_recompute=False)
# end_time = time.time() - start_time
# print(f"Time for FOM: {end_time:2.4f}")

# %% ---------------- Greedy -----------------
surrogate = []
parameter_per_field = np.linspace(
    config["Greedy"]["surrogate"]["min_value"],
    config["Greedy"]["surrogate"]["max_value"],
    config["Greedy"]["surrogate"]["num_values"],
)


if config["Greedy"]["surrogate"]["dimension"] == 1:
    for i in parameter_per_field:
        surrogate.append(np.ones(16) * i)
elif config["Greedy"]["surrogate"]["dimension"] == 4:
    # loop for 4 fields
    for i in parameter_per_field:
        print(f"i-th loop: {i}")
        for j in parameter_per_field:
            for k in parameter_per_field:
                for l in parameter_per_field:
                    array = np.concatenate(
                        [np.ones(4) * i, np.ones(4) * j, np.ones(4) * k, np.ones(4) * l]
                    )
                    surrogate.append(array)
        # surrogate.append(np.ones(16) * i)
        
elif config["Greedy"]["surrogate"]["dimension"] == 6:
    for i in parameter_per_field:
        for j in parameter_per_field:
            for k in parameter_per_field:
                for l in parameter_per_field:
                    for m in parameter_per_field:
                        for n in parameter_per_field:
                                    array = np.array(
                                        [
                                            i, #1
                                            i, #2
                                            m, #3
                                            i, #4
                                            j, #5
                                            j, #6
                                            j, #7
                                            m, #8
                                            k, #9
                                            n, #10
                                            k, #11
                                            k, #12
                                            n, #13
                                            l, #14
                                            l, #15
                                            l, #16
                                        ]
                                    )
                                    surrogate.append(array)
    print(f"6D surrogate space: {len(surrogate)}")                            

elif config["Greedy"]["surrogate"]["dimension"] == 8:
    for i in parameter_per_field:
        for j in parameter_per_field:
            for k in parameter_per_field:
                for l in parameter_per_field:
                    for m in parameter_per_field:
                        for n in parameter_per_field:
                            for o in parameter_per_field:
                                for p in parameter_per_field:
                                    array = np.array(
                                        [
                                            i, #1
                                            j, #2
                                            j, #3
                                            i, #4
                                            k, #5
                                            l, #6
                                            l, #7
                                            k, #8
                                            m, #9
                                            n, #10
                                            n, #11
                                            m, #12
                                            o, #13
                                            p, #14
                                            p, #15
                                            o, #16
                                        ]
                                    )
                                    surrogate.append(array)
    print(f"8D surrogate space: {len(surrogate)}")
elif config["Greedy"]["surrogate"]["dimension"] == 16:
    surrogate = [np.array(combination) for combination in itertools.product(parameter_per_field, repeat=16)]
    print(len(surrogate))
    raise NotImplementedError("Not implemented yet")
else:
    raise ValueError("Dimension of surrogate space must be 1, 4 or 16")

if config["Postprocessing"]["compute_FOM"]:
    for sur in surrogate:
        fom = FOM(
            nx,
            ny,
            t,
            T,
            dt,
            parameter=sur,
            save_directory=config["Infrastructure"]["save_directory"],
        )
        fom.assemble_system(force_recompute=False)
        fom.solve_primal(force_recompute=False)  # True


if len(surrogate) != len(set(tuple(row) for row in surrogate)):
    raise ValueError("There are duplicates in the surrogate space")

# surrogate = [np.ones(16)]

print(f"Number of surrogate members: {len(surrogate)}")

greedy = Greedy(
    fom=fom,
    surrogate=surrogate,
    TOTAL_ENERGY=TOTAL_ENERGY,
    TOLERANCE=config["Greedy"]["tolerance"],
    EVAL_ERROR_LAST_TIME_STEP=config["Greedy"]["eval_error_last_time_step"],
    MAX_ITERATIONS=config["Greedy"]["max_iterations"],
    MAX_ITERATIONS_MORe_DWR=config["Greedy"]["MOReDWR"]["max_iterations"],
    COST_FCT_TRESHOLD=config["Greedy"]["MOReDWR"]["cost_function_treshold"],
    SAVE_DIR=config["Infrastructure"]["save_directory"],
    PLOT_DATA=config["Postprocessing"]["plot_data"],
    MEYER_MATTHIES_EXEC=config["Greedy"]["meyer_matthies"]["execute"],
    MEYER_MATTHIES_ITERATIONS=config["Greedy"]["meyer_matthies"]["min_iterations"],
)

greedy.greedy_enirchment()

greedy.validate_estimates()
