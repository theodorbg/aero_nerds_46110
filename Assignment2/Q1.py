import numpy as np
import matplotlib.pyplot as plt
from panel_method.funaerotool.utils import generate_naca4_contour
import pandas as pd
from xfoil_reader import XFoil, load_xfoil

# Define Aspect Ratios and angles of attack
AR = [4, 6, 8, 10, np.inf]
airfoil = load_xfoil("2410")
alpha = airfoil.alpha
print(alpha)

alpha_L0_deg = 0 # Zero lift angle of attack


# REPLACE WITH ACTUAL DATA
print(airfoil.CL)
# cl_airfoil = np.array([-0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6])
print(airfoil.CD)
# cd_airfoil = np.array([0.018, 0.012, 0.009, 0.010, 0.014, 0.024, 0.040])


# Create empty dictionaries to collect results
CL_data = {}
CDi_data = {}
CD_data = {}

for ar in AR:
    CL_list = []
    CDi_list = []
    CD_list = []

    for a_deg in airfoil.alpha:
        # Convert to radians
        a = np.deg2rad(a_deg)
        alpha_L0 = np.deg2rad(alpha_L0_deg)

        if ar != np.inf:
            # Lift coefficient
            C_L = (2 * np.pi) / (1 + 2 / ar) * (a - alpha_L0)

            # Induced drag coefficient
            C_D_i = C_L**2 / (np.pi * ar)

            # Induced angle of attack [rad]
            alpha_i = C_L / (np.pi * ar)

            # Effective angle of attack [rad]
            alpha_eff = a - alpha_i

        else:
            C_L = 2 * np.pi * (a - alpha_L0)
            C_D_i = 0.0
            alpha_i = 0.0
            alpha_eff = a

        # Profile drag from airfoil polar: Cd = cd(Cl)
        C_D = np.interp(C_L, airfoil.CL, airfoil.CD)

        # Store values
        CL_list.append(C_L)
        CDi_list.append(C_D_i)
        CD_list.append(C_D)

    col_name = "inf" if ar == np.inf else f"{ar}"
    CL_data[col_name] = CL_list
    CDi_data[col_name] = CDi_list
    CD_data[col_name] = CD_list

# Build dataframes
df_CL = pd.DataFrame(CL_data, index=airfoil.alpha)
df_CDi = pd.DataFrame(CDi_data, index=airfoil.alpha)
df_CD = pd.DataFrame(CD_data, index=airfoil.alpha)

# Name the index
df_CL.index.name = "alpha_deg"
df_CDi.index.name = "alpha_deg"
df_CD.index.name = "alpha_deg"

fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=200, constrained_layout=True)

# Left: CL vs geometric angle of attack
for col in df_CL.columns:
    axes[0].plot(df_CL.index, df_CL[col], marker='o', label=f"AR={col}")
axes[0].set_xlabel(r'Angle of attack, $\alpha$ [deg]')
axes[0].set_ylabel(r'$C_L$ [-]')
axes[0].set_title(r'$C_L$ vs $\alpha$')
axes[0].grid(True, alpha=0.3)

# Right: CDi vs geometric angle of attack
for col in df_CDi.columns:
    axes[1].plot(df_CDi.index, df_CDi[col], marker='o', label=f"AR={col}")
axes[1].set_xlabel(r'Angle of attack, $\alpha$ [deg]')
axes[1].set_ylabel(r'$C_{D,i}$ [-]')
axes[1].set_title(r'$C_{D,i}$ vs $\alpha$')
axes[1].grid(True, alpha=0.3)

# Right: CD vs geometric angle of attack
for col in df_CD.columns:
    axes[2].plot(df_CD.index, df_CD[col], marker='o', label=f"AR={col}")
axes[2].set_xlabel(r'Angle of attack, $\alpha$ [deg]')
axes[2].set_ylabel(r'$C_{D}$ [-]')
axes[2].set_title(r'$C_{D}$ vs $\alpha \; NOT \; INTERPOLATED \; YET $')
axes[2].grid(True, alpha=0.3)

# One shared legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.08))

plt.show()