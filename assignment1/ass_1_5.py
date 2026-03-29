import numpy as np
from plot import plot_flexible
from xfoils_class import load_xfoils
xfoils_free, xfoils_fixed = load_xfoils("free_Re_e5", "fixed_Re_e5")
codes = ["2312", "2324", "4412", "4424"]

# create dict to store results
polars = {
    "free": {
        code: {"Cl": None, "Cd": None, "max_cl_cd": None,   }
        for code in codes
    },
    "fixed": {
        code: {"Cl": None, "Cd": None}
        for code in codes
    }
}

# add XFOIL data in one loop
for trans, xfoil_dict in {"free": xfoils_free, "fixed": xfoils_fixed}.items():
    for code in codes:
        xfoil = xfoil_dict[code]
        polars[trans][code]["Cl"] = xfoil.CL
        polars[trans][code]["Cd"] = xfoil.CD

# plot the polar Cl vs Cd
plot_flexible(
    x_val=[
        [polars["free"][code]["Cd"], polars["fixed"][code]["Cd"]]
        for code in codes
    ],
    y_vals=[
        [polars["free"][code]["Cl"], polars["fixed"][code]["Cl"]]
        for code in codes
    ],
    labels=[
        ["XFOIL free transition", "XFOIL fixed transition"]
        for _ in codes
    ],
    x_label="Cd [-]",
    y_units=[f"NACA {code} — Cl [-]" for code in codes],
    save_name="polar_cl_vs_cd_per_airfoil",
    xlims=(0.005, 0.06),
    ylims=[(-1, 2) for _ in codes]
)

# find maximum Cl/Cd for each airfoil and transition and AoA at max Cl/Cd
d_aoa = len(polars["free"][codes[0]]["Cl"])
print(d_aoa)
aoa = np.linspace(-10, 15, d_aoa)  # replace with your actual AoA array if different

for code in codes:
    for trans in ["free", "fixed"]:
        Cl = np.array(polars[trans][code]["Cl"])
        Cd = np.array(polars[trans][code]["Cd"])

        # only consider points where Cd > 0 (avoid division issues)
        valid = Cd > 0
        cl_cd = Cl[valid] / Cd[valid]
        polars[trans][code]["cl_cd"] = cl_cd
        aoa_valid = aoa[valid]

        idx = np.argmax(cl_cd)
        polars[trans][code]["max_cl_cd"] = cl_cd[idx]
        polars[trans][code]["aoa_at_max"] = aoa_valid[idx]
        

# print results to latex table:
# \begin{table}[]
# \centering
# \begin{tabular}{|l|l|l|}
# \hline
# \textbf{Case}                   & \textbf{Maximum Cl/Cd} & \textbf{The AoA at which Max Cl/Cd} \\ \hline
# NACA 2312 with free transition  &                        &                                     \\ \hline
# NACA 2312 with fixed transition &                        &                                     \\ \hline
# NACA 2324 with free transition  &                        &                                     \\ \hline
# NACA 2324 with fixed transition &                        &                                     \\ \hline
# NACA 4412 with free transition  &                        &                                     \\ \hline
# NACA 4412 with fixed transition &                        &                                     \\ \hline
# NACA 4424 with free transition  &                        &                                     \\ \hline
# NACA 4424 with fixed transition &                        &                                     \\ \hline
# \end{tabular}
# \caption{}
# \label{tab:my-table}
# \end{table}
print("\\begin{table}[]")
print("\\centering")
print("\\begin{tabular}{|l|l|l|}")
print("\\hline")
print("\\textbf{Case} & \\textbf{Maximum Cl/Cd} & \\textbf{AoA at Max Cl/Cd} \\\\ \\hline")
for code in codes:
    for trans in ["free", "fixed"]:
        case = f"NACA {code} with {trans} transition"
        max_cl_cd = polars[trans][code]["max_cl_cd"]
        aoa_at_max = polars[trans][code]["aoa_at_max"]
        print(f"{case} & {max_cl_cd:.2f} & {aoa_at_max:.1f}° \\\\ \\hline")
print("\\end{tabular}")
print("\\caption{Maximum Cl/Cd for each airfoil and transition case}")
print("\\label{tab:max_cl_cd}")
print("\\end{table}")

# plot cl/cd vs aoa for each airfoil and transition

plot_flexible(
    x_val=[
        [aoa_valid, aoa_valid]   # same x for both free and fixed
        for _ in codes
    ],
    y_vals=[
        [polars["free"][code]["cl_cd"], polars["fixed"][code]["cl_cd"]]
        for code in codes
    ],
    labels=[
        ["XFOIL free transition", "XFOIL fixed transition"]
        for _ in codes
    ],
    x_label="AoA [deg]",
    y_units=[f"NACA {code} — Cl/Cd [-]" for code in codes],
    save_name="cl_cd_vs_aoa_per_airfoil",
    xlims=(-10, 15),
    ylims=[(-50, 200) for _ in codes]
)