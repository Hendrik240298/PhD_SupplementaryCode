import time

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fenics import *
from petsc4py import PETSc
from sklearn.metrics import confusion_matrix

from FOM import FOM
from iROM import iROM

# ----------- FOM parameters -----------
# start time
t = 0.0
# end time
T = 5.0  # 10. #10. #10.0
# time step size
dt = 0.001
# defining the mesh
nx = ny = 50

# ----------- ROM parameters -----------
REL_ERROR_TOL = 1e-2
MAX_ITERATIONS = 200
PARENT_SLAB_SIZE = int((T / dt) / 1)  # 10000000  # 1000 #1250
TOTAL_ENERGY = {
    "primal": 1 - 1e-11,
    "dual": 1 - 1e-14,
}
# we have a parameter space P, its surrogate P_h and an living parameter
# parameter
parameter = np.random.uniform(0.05, 2.0, 16)
parameter = np.ones(16) * 0.5  # np.random.uniform(0.01, 2., 1)

# %% ---------------- FOM -----------------
fom = FOM(nx, ny, t, T, dt, parameter=parameter)
fom.assemble_system(force_recompute=False)
start_time = time.time()
fom.solve_primal(force_recompute=True)
end_time = time.time() - start_time
print(f"Time for FOM: {end_time:2.4f}")

# %% ---------------- ROM -----------------
rom = iROM(
    fom,
    REL_ERROR_TOL=REL_ERROR_TOL,
    MAX_ITERATIONS=MAX_ITERATIONS,
    PARENT_SLAB_SIZE=PARENT_SLAB_SIZE,
    TOTAL_ENERGY=TOTAL_ENERGY,
    parameter=parameter,
)
rom.run_parent_slab()

# %% Postprocessing
print("============= TIMINGS =============")
print(f"Time for iROM: {rom.timings['run']:2.4f}")
print(
    f"         iPOD: {rom.timings['iPOD']:2.4f} ({rom.timings['iPOD']/rom.timings['run']*100:2.2f}%)"
)
print(
    f"         pROM: {rom.timings['primal_ROM']:2.4f} ({rom.timings['primal_ROM']/rom.timings['run']*100:2.2f}%)"
)
print(
    f"         dROM: {rom.timings['dual_ROM']:2.4f} ({rom.timings['dual_ROM']/rom.timings['run']*100:2.2f}5)"
)
print(
    f"        Estim: {rom.timings['error_estimate']:2.4f} ({rom.timings['error_estimate']/rom.timings['run']*100:2.2f}%)"
)
print(
    f"       Enrich: {rom.timings['enrichment']:2.4f} ({rom.timings['enrichment']/rom.timings['run']*100:2.2f}%)"
)
print(
    f"        Other: "
    + "{:2.4f}".format(
        rom.timings["run"]
        - (
            rom.timings["iPOD"]
            + rom.timings["primal_ROM"]
            + rom.timings["dual_ROM"]
            + rom.timings["error_estimate"]
            + rom.timings["enrichment"]
        )
    )
    + " ("
    + "{:2.2f}".format(
        (
            rom.timings["run"]
            - (
                rom.timings["iPOD"]
                + rom.timings["primal_ROM"]
                + rom.timings["dual_ROM"]
                + rom.timings["error_estimate"]
                + rom.timings["enrichment"]
            )
        )
        / rom.timings["run"]
        * 100
    )
    + "%)"
)

print("============= RESULTS =============")

print(f"J(u_h):            {np.sum(fom.functional_values)}")
# print(f"J(u_N) - DEBUG:    {np.sum(cf)}")
print(f"J(u_N):            {np.sum(rom.functional_values)}")
print(
    f"Relative error:    {np.sum(fom.functional_values - rom.functional_values)/np.sum(fom.functional_values)*100}%"
)
print(f"True error:        {np.sum(fom.functional_values - rom.functional_values)}")
print(f"Estimated error:   {np.sum(np.abs(rom.errors))}")
print(
    f"Effectivity index: {np.abs(np.sum(np.abs(rom.errors))/np.sum(fom.functional_values - rom.functional_values))}"
)
print(
    f"Indicator index:   {np.sum(np.abs(np.abs(rom.errors)))/np.sum(np.abs(fom.functional_values - rom.functional_values))}"
)

# %% error classification
true_tol = (
    np.abs((fom.functional_values - rom.functional_values) / fom.functional_values) > REL_ERROR_TOL
)
esti_tol = np.abs(rom.errors) > REL_ERROR_TOL

if np.sum(true_tol) == np.sum(esti_tol):
    print("estimator works perfectly")
else:
    confusion_matrix = confusion_matrix(true_tol.astype(int), esti_tol.astype(int))
    eltl, egtl, eltg, egtg = confusion_matrix.ravel()
    # n_slabs=100

    print(f"(error > tol & esti < tol): {eltg} ({round(100 * eltg / (T/dt) ,1)} %)  (very bad)")
    print(f"(error < tol & esti > tol): {egtl} ({round(100 * egtl / (T/dt) ,1)} %)  (bad)")
    print(f"(error > tol & esti > tol): {egtg} ({round(100 * egtg / (T/dt) ,1)} %)  (good)")
    print(f"(error < tol & esti < tol): {eltl} ({round(100 * eltl / (T/dt) ,1)} %)  (very good)")


# %%
# ===========================================
# ================== PLOTS ==================
# ===========================================

FONT_SIZE_AXIS = 15
FONT_LABEL_SIZE = 13

plt.plot(fom.time_points[:-1], fom.functional_values, label="FOM")
# plt.plot(fom.time_points[1:], rom.functional_values)
plt.plot(fom.time_points[1:], rom.functional_values, label="ROM")
# plt.ylim([0, 0.014])
plt.legend()
plt.grid()
plt.show()

plt.plot(range(1, len(rom.functional_values) + 1), rom.functional_values, label="ROM")
# plt.ylim([0, 0.014])
plt.legend()
plt.grid()
plt.show()

plt.plot(
    range(1, len(rom.functional_values) + 1),
    np.abs(fom.functional_values - rom.functional_values),
    label="ROM - error",
)
plt.plot(range(1, len(rom.functional_values) + 1), np.abs(rom.errors), label="ROM - estimate")
# plt.ylim([0, 0.014])
# logy y axis
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()


plt.plot(
    range(1, len(rom.functional_values) + 1),
    np.abs(fom.functional_values - rom.functional_values) / np.abs(fom.functional_values),
    label="relative error",
)
# plt.ylim([0, 0.014])
# logy y axis
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()


# ---------------------------------
# Error over iterations
# ---------------------------------

fom_cf = np.sum(fom.functional_values)  # np.sum(fom.functional_values)

# print("iteration infos:", iterations_infos)
error_cf = np.abs(fom_cf - np.array(rom.iterations_infos[0]["functional"])) / np.abs(fom_cf)

# print(error_cf.shape)
# print(np.array(iterations_infos[0]["error"]).shape)
plt.semilogy(
    100 * error_cf,
    label="$e^{\\mathrm{rel}}$: error",
    linewidth=3,
)
plt.semilogy(
    100 * np.array(rom.iterations_infos[0]["error"]),
    label="$\\eta^{\\mathrm{rel}}$: estimate",
    linestyle=":",
    linewidth=3,
    color="red",
)
plt.xlabel("#iterations", fontsize=FONT_SIZE_AXIS)
plt.ylabel("relative error [%]", fontsize=FONT_SIZE_AXIS)
plt.legend(fontsize=FONT_LABEL_SIZE)
plt.grid()
# set the font size of the tick labels
plt.tick_params(axis="both", which="major", labelsize=13)

plt.show()


# %%
# ===========================================
# ================== Video ==================
# ===========================================

# make a video of the FOM solution out of snapshot matrix fom.Y
print(f"Save each {(T/dt)/100}-th FOM solution as vtk (100 frames per simulation second)")
vtkfile = File(f"plots/FOM/u.pvd")
for i in range(fom.Y.shape[1]):
    if i % ((T / dt) / 100) == 0:
        u = Function(fom.V, name="solution")
        u.vector().set_local(fom.Y[:, i])
        # vtkfile = File(f"plots/FOM/u_{str(i).zfill(6)}.pvd")
        vtkfile << u
