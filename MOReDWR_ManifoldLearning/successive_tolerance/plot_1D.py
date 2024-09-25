import argparse
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml

# %% ---------- Parameter file ------------
# specify yaml config file ove command line python3 main_greedy.py
# PATH_config_file
parser = argparse.ArgumentParser(description="Input file to specify the problem.")
parser.add_argument("yaml_config", nargs="?", help="Path/Name to the YAML config file")

# parse the arguments
args = parser.parse_args()

# ATTENTION: No sanity check for yaml config exists yet!
if args.yaml_config is None:
    print(
        "No YAML config file was specified. Thus standard config 'result_generation_1D.yaml' is used."
    )
    config_file = "config/result_generation_1D.yaml"
else:
    config_file = args.yaml_config

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

TOL = config["Greedy"]["tolerance"]
SAVE_DIR = config["Infrastructure"]["save_directory"]
TIME = np.arange(
    config["FOM"]["start_time"] + config["FOM"]["dt"],
    config["FOM"]["end_time"] + config["FOM"]["dt"],
    config["FOM"]["dt"],
)

# %% ----------- Plot parameters -----------
FONT_SIZE_AXIS = 15
FONT_LABEL_SIZE = 13
BLUE = "#1f77b4"
RED = "#FF0000"
GREEN = "#008000"

PLOT_DIR = config["Infrastructure"]["plot_directory"]
# check if PLOT_DIR exists, if not create it
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


# %% ----------- Import iteration data -----------
with open(SAVE_DIR + "iteration_data.pkl", "rb") as f:
    iteration = pickle.load(f)

with open(SAVE_DIR + "surrogate.pkl", "rb") as f:
    surrogate = pickle.load(f)

with open(SAVE_DIR + "absolute_errors.pkl", "rb") as f:
    absolute_error_estimates = pickle.load(f) # since absolute errors is just sum of errors instead of absolute need to change later

total_fom_solves = np.sum(np.array(iteration["fom_solves"]))
end_primal_POD_size = np.array(iteration["POD_size"]["primal"])[-1]
end_dual_POD_size = np.array(iteration["POD_size"]["dual"])[-1]
total_iterations = len(iteration["fom_solves"]) 

# write total valies to file total_values.txt
with open(PLOT_DIR + "total_values.txt", "w") as f:
    f.write(f"Total FOM solves:    {total_fom_solves}\n")
    f.write(f"End Primal POD size: {end_primal_POD_size}\n")
    f.write(f"End Dual POD size:   {end_dual_POD_size}")
    f.write(f"Total iterations:    {total_iterations}")

# %% ----------- Import cost functional data -----------
def load_FOM_functionals(parameter):
    # load functional_values of fom
    pattern = r"fom_functional_values_\d{6}\.npz"
    files = os.listdir(SAVE_DIR)
    files = [
        SAVE_DIR + f
        for f in files
        if os.path.isfile(os.path.join(SAVE_DIR, f)) and re.match(pattern, f)
    ]

    discretization_parameters = np.array(
        [
            config["FOM"]["mesh_size"],
            config["FOM"]["mesh_size"],
            config["FOM"]["dt"],
            config["FOM"]["end_time"],
        ]
    )

    for file in files:
        tmp = np.load(file)
        if np.allclose(discretization_parameters, tmp["discretization_parameters"], atol=1e-10):
            if np.allclose(parameter, tmp["parameter"], atol=1e-10):
                functional_values = tmp["functional_values"]
                print(f"Loaded {file}")
                return functional_values

    return False


def load_ROM_functionals(parameter):
    # load functional_values of fom
    pattern = r"ROM_functional_values_\d{6}\.npz"
    files = os.listdir(SAVE_DIR)
    files = [
        SAVE_DIR + f
        for f in files
        if os.path.isfile(os.path.join(SAVE_DIR, f)) and re.match(pattern, f)
    ]

    for file in files:
        tmp = np.load(file)
        if np.allclose(parameter, tmp["parameter"], atol=1e-10):
            functional_values = tmp["functional_values"]
            print(f"Loaded {file}")
            return functional_values

    return False


fom_functional_values = []
rom_functional_values = []


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

# wanted_parameter = [
#     np.concatenate([1.0 * np.ones(4), 2.0 * np.ones(4), 3.0 * np.ones(4), 5.0 * np.ones(4)]),
#     np.concatenate([4.0 * np.ones(4), 1.0 * np.ones(4), 5.0 * np.ones(4), 3.0 * np.ones(4)]),
#     np.concatenate([5.0 * np.ones(4), 1.0 * np.ones(4), 5.0 * np.ones(4), 1.0 * np.ones(4)]),
# ]

wanted_parameter = surrogate[::]

mean_error_histogram = []
max_error_histogram = []
rel_cf_error = []
effectivity_index = []



for i, sur in enumerate(wanted_parameter):
    fom_functional_values.append(load_FOM_functionals(sur))
    if fom_functional_values is False:
        raise ValueError("FOM functional values could not be loaded")
    rom_functional_values.append(load_ROM_functionals(sur))
    if rom_functional_values is False:
        raise ValueError("ROM functional values could not be loaded")

    mean_error = 100*np.mean(np.abs(fom_functional_values[i] - rom_functional_values[i])/np.abs(fom_functional_values[i]))
    max_error = 100*np.max(np.abs(fom_functional_values[i] - rom_functional_values[i])/np.abs(fom_functional_values[i]))
    
    J_h = np.sum(fom_functional_values[i])
    J_N = np.sum(rom_functional_values[i])
    rel_cf_error.append(100*np.abs(J_h - J_N)/np.abs(J_h))
    
    mean_error_histogram.append(mean_error)
    max_error_histogram.append(max_error)

    effectivity_index.append(np.abs(absolute_error_estimates[i])/np.abs((J_h - J_N)))

    # print(f"Max relative error: {np.mean(np.abs(fom_functional_values[i] - rom_functional_values[i])/np.abs(fom_functional_values[i]))} %")

max_cost_fct_value = max(np.max(array) for array in fom_functional_values)


diff = np.diff(iteration["TOL"])
indices = np.where(diff != 0)[0]
TOL_i = list(sorted(set(iteration["TOL"]), reverse=True))
print(f"TOL_i: {TOL_i}")
print(indices)
print(iteration["TOL"])

# %% ---------------- Plotting -----------------
indices = np.concatenate(([0], indices, [len(iteration["relative_error"])-1]))

alpha=np.linspace(0.1, 0.75, len(indices)-1)[::-1]
print(f"alpha: {alpha}")
# ----------------------------------
# Error development over iterations
# ----------------------------------

plt.plot(
    100 * np.array(iteration["relative_error"]),
    label="error estimate",
    linewidth=3,
    color=BLUE,
)
# plt.plot(
#     100 * np.array(iteration["fom_error_max"]),
#     label="exact error",
#     linestyle="--",
#     linewidth=3,
#     color=RED,
# )
# plt.plot(
#     100 * np.ones(len(iteration["relative_error"])) * TOL,
#     label="$\\mathrm{tol} = 1\\%$",
#     linestyle=":",
#     linewidth=3,
#     color=GREEN,
# )

# color the area between the lines
y1, y2 = plt.ylim()
# plot vertical lines at each indices
for i in indices:
    plt.axvline(x=i, color="grey", linestyle="--", linewidth=1.5, alpha=0.5)
for i in range(len(indices)-1):
    plt.fill_between([indices[i], indices[i+1]], y1, y2, color='gray', alpha=alpha[i])
    plt.plot(
        [indices[i], indices[i+1]],
        100 * np.ones(2) * TOL_i[i],
        # label="$\\mathrm{tol} = 1\\%$",
        linestyle=":",
        linewidth=3,
        color=GREEN,
    )


plt.xlabel("#iterations", fontsize=FONT_SIZE_AXIS)
plt.ylabel("relative error [%]", fontsize=FONT_SIZE_AXIS)
plt.yscale("log")
plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
plt.grid()
plt.savefig(PLOT_DIR + "error_over_iteration.pdf", bbox_inches="tight")
plt.close()


# ---------------------------------------
# Final error distribution over parameter space
# ---------------------------------------
parameter = np.zeros(len(surrogate))

for i, sur in enumerate(surrogate):
    parameter[i] = sur[0]
# print(np.array(iteration["arr_rel_error"][-1]))
# print(surrogate)
plt.plot(
    parameter,
    100 * np.array(iteration["arr_rel_error"][-1]),
    label="error estimate",
    linewidth=3,
    color=BLUE,
)
# plt.plot(
#     parameter,
#     100 * np.array(iteration["fom_error"][-1]),
#     label="exact error",
#     linestyle="--",
#     linewidth=3,
#     color=RED,
# )
plt.xlabel("parameter", fontsize=FONT_SIZE_AXIS)
plt.ylabel("relative error [%]", fontsize=FONT_SIZE_AXIS)
# plt.yscale("log")
plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
plt.grid()
plt.savefig(PLOT_DIR + "error_over_parameter_space.pdf", bbox_inches="tight")
plt.close()
# plt.show()

# ----------------------------------
# POD Size development over iterations
# ----------------------------------
plt.plot(
    np.array(iteration["POD_size"]["primal"]),
    label="primal",
    linewidth=3,
    color=BLUE,
)
plt.plot(
    np.array(iteration["POD_size"]["dual"]),
    label="dual",
    linestyle="--",
    linewidth=3,
    color=RED,
)

# color the area between the lines
y1, y2 = plt.ylim()
# plot vertical lines at each indices
for i in indices:
    plt.axvline(x=i, color="grey", linestyle="--", linewidth=1.5, alpha=0.5)
for i in range(len(indices)-1):
    plt.fill_between([indices[i], indices[i+1]], y1, y2, color='gray', alpha=alpha[i])

plt.xlabel("#iterations", fontsize=FONT_SIZE_AXIS)
plt.ylabel("POD size", fontsize=FONT_SIZE_AXIS)
plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
plt.grid()
plt.savefig(PLOT_DIR + "POD_size_over_iteration.pdf", bbox_inches="tight")
plt.close()

# ----------------------------------
# FOM solves per iteration
# ----------------------------------
plt.plot(
    np.array(iteration["fom_solves"]),
    label="primal",
    linewidth=3,
    color=BLUE,
)

# color the area between the lines
y1, y2 = plt.ylim()
# plot vertical lines at each indices
for i in indices:
    plt.axvline(x=i, color="grey", linestyle="--", linewidth=1.5, alpha=0.5)
for i in range(len(indices)-1):
    plt.fill_between([indices[i], indices[i+1]], y1, y2, color='gray', alpha=alpha[i])

plt.xlabel("#iterations", fontsize=FONT_SIZE_AXIS)
plt.ylabel("FOM solves", fontsize=FONT_SIZE_AXIS)
plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
plt.grid()
plt.savefig(PLOT_DIR + "FOM_solves_per_iteration.pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------
# Plot Cost functional for three parameters
# ---------------------------------------

# find 3 indices of parameter with highest errors, i. e., entries of mean_error_histogram
three_largest_errors = np.argsort(mean_error_histogram)[-3:]
print(f"three_largest_errors indices: {three_largest_errors}")
print(f"three_largest_errors values: {np.array(mean_error_histogram)[three_largest_errors]}")
for i in three_largest_errors:
    plt.plot(
        TIME,
        fom_functional_values[i],
        label="FOM",
        linewidth=3,
        color=RED,
    )

    plt.plot(
        TIME,
        rom_functional_values[i],
        label="ROM",
        linestyle=":",
        linewidth=3,
        color=BLUE,
    )

    plt.xlabel("time [t]", fontsize=FONT_SIZE_AXIS)
    plt.ylabel("$J(u)$", fontsize=FONT_SIZE_AXIS)
    # plt.yscale("log")
    # plt.ylim(0, max_cost_fct_value)
    plt.legend(fontsize=FONT_LABEL_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=13)
    plt.grid()
    plt.savefig(PLOT_DIR + f"cost_functional_{i}_err_{np.array(mean_error_histogram)[i]:.3f}.pdf", bbox_inches="tight")
    plt.close()
    # plt.show()




# ---------------------------------------
# Plot error histograms
# ---------------------------------------

plt.hist(mean_error_histogram, bins=20)
plt.xlabel("mean relative error [%]", fontsize=FONT_SIZE_AXIS)
plt.ylabel("#samples", fontsize=FONT_SIZE_AXIS)
# plt.yscale("log")
# plt.ylim(0, max_cost_fct_value)
# plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
plt.grid()
plt.savefig(PLOT_DIR + f"mean_error_histogram.pdf", bbox_inches="tight")
plt.close()



plt.hist(max_error_histogram, bins=20)
plt.xlabel("mean relative error [%]", fontsize=FONT_SIZE_AXIS)
plt.ylabel("#samples", fontsize=FONT_SIZE_AXIS)
# plt.yscale("log")
# plt.ylim(0, max_cost_fct_value)
# plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
plt.grid()
plt.savefig(PLOT_DIR + f"max_error_histogram.pdf", bbox_inches="tight")
plt.close()



plt.hist(rel_cf_error, bins=40)
plt.xlabel("relative cost functional error [%]", fontsize=FONT_SIZE_AXIS)
plt.ylabel("#samples", fontsize=FONT_SIZE_AXIS)
# plt.yscale("log")
# plt.ylim(0, max_cost_fct_value)
# plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
# plt.xticks([0.074,0.075,0.076,0.077, 0.078]) 
plt.grid()
plt.savefig(PLOT_DIR + f"cf_error_histogram.pdf", bbox_inches="tight")
plt.close()




plt.hist(effectivity_index, bins=20)
plt.xlabel("effectivity index", fontsize=FONT_SIZE_AXIS)
plt.ylabel("#samples", fontsize=FONT_SIZE_AXIS)
# plt.yscale("log")
# plt.ylim(0, max_cost_fct_value)
# plt.legend(fontsize=FONT_LABEL_SIZE)
plt.tick_params(axis="both", which="major", labelsize=13)
plt.grid()
plt.savefig(PLOT_DIR + f"effectivity_index_histogram.pdf", bbox_inches="tight")
plt.close()
