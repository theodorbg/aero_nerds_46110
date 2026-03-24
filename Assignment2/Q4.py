import matplotlib.pyplot as plt
import numpy as np
from glauert_solver import solve_wing_glauert

AR_q4 = 6
alpha_root_deg = 2.0
alpha_tip_list = [0, 2, 4, 6, 8]
alpha_L0_deg = 0.0

solutions_q4 = {}

for alpha_tip in alpha_tip_list:
    sol = solve_wing_glauert(
        planform="Rectangular",
        AR=AR_q4,
        alpha_L0_deg=alpha_L0_deg,
        twist_type="linear",
        alpha_root_deg=alpha_root_deg,
        alpha_tip_deg=alpha_tip,
        N_terms=80,
        N_eval=500
    )
    solutions_q4[alpha_tip] = sol


fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, sharex=True)
plt.subplots_adjust(wspace=0.30, hspace=0.28)

# 1) Dimensionless circulation
for alpha_tip in alpha_tip_list:
    sol = solutions_q4[alpha_tip]
    axes[0, 0].plot(sol["x_tilde"], sol["Gamma_tilde"], label=rf'$\alpha_{{g,\mathrm{{tip}}}}={alpha_tip}^\circ$')

axes[0, 0].set_ylabel(r'$\tilde{\Gamma}$ [-]')
axes[0, 0].set_title('Dimensionless circulation')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc="upper left", fontsize=8, frameon=True)

# 2) Induced angle of attack
for alpha_tip in alpha_tip_list:
    sol = solutions_q4[alpha_tip]
    axes[0, 1].plot(sol["x_tilde"], sol["alpha_i_deg"])

axes[0, 1].set_ylabel(r'$\alpha_i$ [deg]')
axes[0, 1].set_title('Induced angle of attack')
axes[0, 1].grid(True, alpha=0.3)

# 3) Local lift coefficient
for alpha_tip in alpha_tip_list:
    sol = solutions_q4[alpha_tip]
    axes[1, 0].plot(sol["x_tilde"], sol["cl_local"])

axes[1, 0].set_xlabel(r"Span coordinate, $\tilde{x}$ [-]")
axes[1, 0].set_ylabel(r'$c_l$ [-]')
axes[1, 0].set_title('Local lift coefficient')
axes[1, 0].grid(True, alpha=0.3)

# 4) Local induced drag coefficient
for alpha_tip in alpha_tip_list:
    sol = solutions_q4[alpha_tip]
    axes[1, 1].plot(sol["x_tilde"], sol["cdi_local"])

axes[1, 1].set_xlabel(r"Span coordinate, $\tilde{x}$ [-]")
axes[1, 1].set_ylabel(r'$c_{d,i}$ [-]')
axes[1, 1].set_title('Local induced drag coefficient')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(y=0.98)
plt.show()