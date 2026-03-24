import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glauert_solver import solve_wing_glauert

alpha_plot_deg = [0, 4, 8]
AR = [4, 6, 8, 10, np.inf]
alpha_L0_deg = 0


fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200, constrained_layout=True, sharex=True, sharey=True)
plt.subplots_adjust(top=0.82, wspace=0.18)

for i, a_plot in enumerate(alpha_plot_deg):
    ax = axes[i]

    for ar in AR:
        sol = solve_wing_glauert(
            planform="Rectangular",
            AR=ar,
            alpha_deg=a_plot,
            alpha_L0_deg=alpha_L0_deg,
            N_terms=60,
            N_eval=500
        )

        label = r"AR=$\infty$" if np.isinf(ar) else f"AR={ar}"
        ax.plot(sol["x_tilde"], sol["alpha_i_deg"], label=label)

    ax.set_xlabel(r"$\tilde{x}$ [-]")
    ax.set_ylabel(r"$\alpha_i$ [deg]")
    ax.set_title(rf"$\alpha = {a_plot}^\circ$")
    ax.grid(True, alpha=0.3)

axes[0].legend(loc="upper left", frameon=True)
plt.show()

plt.show()


# Q2b: CL and CDi vs alpha for rectangular wing
alpha_plot_deg = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

CL_rect_data = {}
CDi_rect_data = {}

for ar in AR:
    CL_list = []
    CDi_list = []

    for a_deg in alpha_plot_deg:
        sol = solve_wing_glauert(
            planform="Rectangular",
            AR=ar,
            alpha_deg=a_deg,
            alpha_L0_deg=alpha_L0_deg,
            N_terms=60,
            N_eval=400
        )

        CL_list.append(sol["CL"])
        CDi_list.append(sol["CDi"])

    col_name = "inf" if np.isinf(ar) else f"{ar}"
    CL_rect_data[col_name] = CL_list
    CDi_rect_data[col_name] = CDi_list

df_CL_rect = pd.DataFrame(CL_rect_data, index=alpha_plot_deg)
df_CDi_rect = pd.DataFrame(CDi_rect_data, index=alpha_plot_deg)

df_CL_rect.index.name = "alpha_deg"
df_CDi_rect.index.name = "alpha_deg"

fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200, constrained_layout=True)

# CL
for col in df_CL_rect.columns:
    axes[0].plot(df_CL_rect.index, df_CL_rect[col], marker="o", label=f"AR={col}")
axes[0].set_xlabel(r"$\alpha$ [deg]")
axes[0].set_ylabel(r"$C_L$ [-]")
axes[0].grid(True, alpha=0.3)

# CDi
for col in df_CDi_rect.columns:
    axes[1].plot(df_CDi_rect.index, df_CDi_rect[col], marker="o", label=f"AR={col}")
axes[1].set_xlabel(r"$\alpha$ [deg]")
axes[1].set_ylabel(r"$C_{D,i}$ [-]")
axes[1].grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.08))

plt.show()