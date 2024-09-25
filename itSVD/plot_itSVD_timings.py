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

from tabulate import tabulate

# load results/itSVD/itSVD_timings_bunch_sizes.pkl
with open("results/itSVD/itSVD_timings_bunch_sizes.pkl", "rb") as f:
    timings = pickle.load(f)
    

# Example timing data (replace this with your actual data)
# This example data structure assumes timing data for 5 iterations with 6 different settings

BUNCH_SIZES = [1, 5, 10, 25, 50, 100]

QUANTITY = "velocity"

timings_array = []

for i in range(len(BUNCH_SIZES)):
    timing = np.array(
        [
            timings[i][QUANTITY]["SVD"],
            timings[i][QUANTITY]["QR"],
            timings[i][QUANTITY]["orthogonality"],
            timings[i][QUANTITY]["build_comps"],
            timings[i][QUANTITY]["prep"],
            timings[i][QUANTITY]["update_U"] + timings[i][QUANTITY]["update_V"] + timings[i][QUANTITY]["update_S"],
            timings[i][QUANTITY]["rank"] + timings[i][QUANTITY]["rank"],
        ]
    )
    timings_array.append(timing)

timing_data = {
    'iteration': [
        "SVD", "QR", "Reorthogonalization", "Build F", "Compute $H$ \& $P$", "Update SVD Matrices", "Miscellaneous"
                  ],  # Iterations 1 to 5
    'setting_500': timings_array[0],
    'setting_250': timings_array[1],
    'setting_125': timings_array[2],
    'setting_62': timings_array[3],
    'setting_31': timings_array[4],
    'setting_15': timings_array[5],
}

# Bar width
bar_width = 0.1

# Positions of the bars on the x-axis
r1 = np.arange(len(timing_data['iteration']))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]
r6 = [x + bar_width for x in r5]



# Create the plot

plt.rcParams['text.usetex'] = True

FONT_SIZE_AXIS = 27
FONT_LABEL_SIZE = 25
FONT_SIZE_AXIS_NUMBER = 23

plt.figure(figsize=(20, 6))




# Make the plots
plt.bar(r1, timing_data['setting_500'], color='blue', width=bar_width, edgecolor='grey', label='b = 1',)
plt.bar(r2, timing_data['setting_250'], color='red', width=bar_width, edgecolor='grey', label='b = 5')
plt.bar(r3, timing_data['setting_125'], color='green', width=bar_width, edgecolor='grey', label='b = 10')
plt.bar(r4, timing_data['setting_62'], color='purple', width=bar_width, edgecolor='grey', label='b = 25')
plt.bar(r5, timing_data['setting_31'], color='orange', width=bar_width, edgecolor='grey', label='b = 50')
plt.bar(r6, timing_data['setting_15'], color='darkgreen', width=bar_width, edgecolor='grey', label='b = 100')

# Add xticks on the middle of the group bars
plt.xlabel(r"Operations", fontweight='bold',fontsize = FONT_SIZE_AXIS)
plt.xticks(
    [r + bar_width*2.5 for r in range(len(timing_data['iteration']))],
    timing_data['iteration'], 
    fontsize = FONT_LABEL_SIZE
    )

plt.yscale('log')
plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE_AXIS_NUMBER)
plt.grid()
# Add labels and title
plt.ylabel('Time [s]', fontweight='bold', fontsize = FONT_SIZE_AXIS)
# plt.title('Operations', fontweight='bold')

# Create legend & Show graphic
plt.legend(fontsize = FONT_LABEL_SIZE, loc="upper right")

# Adjust layout to remove excess borders
plt.tight_layout()
plt.savefig(f"plots/itSVD/itSVD_timings_bunch_sizes_{QUANTITY}.pdf",bbox_inches='tight')

plt.show()


# read in pure svd timings results/itSVD/pure_SVD_timings.pkl
with open("results/itSVD/pure_SVD_timings.pkl", "rb") as f:
    timings_pure_SVD = pickle.load(f)


# plot total times for each bunch size
total_times = []
for i in range(len(BUNCH_SIZES)):
    total_times.append(timings[i][QUANTITY]["total"] - timings[i][QUANTITY]["expand"])
    
print(total_times)

total_timing_factors = []
for i in range(len(BUNCH_SIZES)):
    total_timing_factors.append(total_times[i]/timings_pure_SVD[QUANTITY])

## total times in latex table format with bunch sizes
data = list(zip(BUNCH_SIZES, total_times, total_timing_factors))

# Generate LaTeX table
latex_table = tabulate(data, headers=['Bunch Size', 'Total Time (s)', 'Factor to SVD'], tablefmt="latex_booktabs")

print(latex_table)

# save latex table to file
with open(f"plots/itSVD/itSVD_timings_bunch_sizes_{QUANTITY}.tex", "w") as f:
    f.write(latex_table)
