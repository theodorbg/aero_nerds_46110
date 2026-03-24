import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glauert_solver import solve_wing_glauert


alpha_plot_deg = 5
AR = 6
TR = [0.2, 0.4, 0.6, 0.8, 1.0]
alpha_L0_deg = 0


def chord_ratio_tapered(x_tilde, TR):
    """
    Returns c(x_tilde) / c_bar for a linearly tapered wing.

    x_tilde = x / (b/2) in [-1, 1]
    TR = c_tip / c_root
    """
    return 2.0 * (1.0 - (1.0 - TR) * np.abs(x_tilde)) / (1.0 + TR)


# ============================================================
# Question 3
# Tapered wing, AR = 6, AoA = 5 deg, TR = 0.2 ... 1.0
# ============================================================

AR_q3 = 6
alpha_q3 = 5.0
TR_list = [0.2, 0.4, 0.6, 0.8, 1.0]

def chord_ratio_tapered(x_tilde, TR):
    """
    Returns c(x_tilde) / c_bar for a linearly tapered wing.

    x_tilde = x / (b/2) in [-1, 1]
    TR = c_tip / c_root
    """
    return 2.0 * (1.0 - (1.0 - TR) * np.abs(x_tilde)) / (1.0 + TR)


# ------------------------------------------------------------
# Q3a: Plot nondimensional chord distribution c/c_bar
# ------------------------------------------------------------
x_tilde_plot = np.linspace(-1, 1, 500)

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

for tr in TR_list:
    c_ratio = chord_ratio_tapered(x_tilde_plot, tr)
    ax.plot(x_tilde_plot, c_ratio, label=f"TR={tr}")

ax.set_xlabel(r"Span coordinate, $\tilde{x}$ [-]")
ax.set_ylabel(r"$c(\tilde{x})/\bar{c}$ [-]")
ax.set_title(r"Tapered wing chord distribution, $AR=6$")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()


# ------------------------------------------------------------
# Q3b: Solve for each taper ratio
# ------------------------------------------------------------
solutions_q3 = {}

for tr in TR_list:
    sol = solve_wing_glauert(
        planform="Tapered",
        AR=AR_q3,
        alpha_deg=alpha_q3,
        alpha_L0_deg=alpha_L0_deg,
        taper_ratio=tr,
        N_terms=80,
        N_eval=500
    )
    solutions_q3[tr] = sol


# ------------------------------------------------------------
# Plot all requested spanwise distributions in one figure
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, sharex=True)
plt.subplots_adjust(wspace=0.4, hspace=0.4)   # more horizontal space

# 1) Dimensionless circulation
for tr in TR_list:
    sol = solutions_q3[tr]
    axes[0, 0].plot(sol["x_tilde"], sol["Gamma_tilde"], label=f"TR={tr}")

axes[0, 0].set_ylabel(r'$\tilde{\Gamma}$ [-]')   # removes the equation text
axes[0, 0].set_title(r"Dimensionless circulation")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc="upper left", fontsize=8, frameon=True)

# 2) Induced angle of attack
for tr in TR_list:
    sol = solutions_q3[tr]
    axes[0, 1].plot(sol["x_tilde"], sol["alpha_i_deg"], label=f"TR={tr}")

axes[0, 1].set_ylabel(r"$\alpha_i$ [deg]")
axes[0, 1].set_title(r"Induced angle of attack")
axes[0, 1].grid(True, alpha=0.3)

# 3) Local lift coefficient
for tr in TR_list:
    sol = solutions_q3[tr]
    axes[1, 0].plot(sol["x_tilde"], sol["cl_local"], label=f"TR={tr}")

axes[1, 0].set_xlabel(r"Span coordinate, $\tilde{x}$ [-]")
axes[1, 0].set_ylabel(r"$c_l$ [-]")
axes[1, 0].set_title(r"Local lift coefficient")
axes[1, 0].grid(True, alpha=0.3)

# 4) Local induced drag coefficient
for tr in TR_list:
    sol = solutions_q3[tr]
    axes[1, 1].plot(sol["x_tilde"], sol["cdi_local"], label=f"TR={tr}")

axes[1, 1].set_xlabel(r"Span coordinate, $\tilde{x}$ [-]")
axes[1, 1].set_ylabel(r'$c_{d,i}$ [-]', labelpad=10)
axes[1, 1].set_title(r"Local induced drag coefficient")
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(rf"Tapered wing, $AR={AR_q3}$, $\alpha={alpha_q3}^\circ$", y=0.98)
plt.show()


# ------------------------------------------------------------
# Optional: print total wing coefficients for each taper ratio
# ------------------------------------------------------------
print("Total wing coefficients for Q3:")
for tr in TR_list:
    sol = solutions_q3[tr]
    print(f"TR = {tr:>3} :  CL = {sol['CL']:.5f}   CDi = {sol['CDi']:.6f}")
