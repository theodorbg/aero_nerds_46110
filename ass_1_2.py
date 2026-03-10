import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *
from panel_method.exercise.solver import solve_closed_contour_panel_method



# task 1
# create dictionary to store the airfoils
airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}


#%% task 2 a
# Evaluate and compare the lift coefficient using Thin airfoil theory 
aoa = np.arange(-10, 16, 1)  # AoA from -10 to 15 degrees
for airfoil in airfoils.values():
    # create thin airfoil theory instance
    thin_af = ThinAirfoilTheory(airfoil)

    # Use thin airfoil theory to estimate Cl
    cl_thin = thin_af.compute_cl(aoa)
    # add cl to dictionary
    airfoil.cl_dict["Thin Airfoil"] = cl_thin
    # calculate slope of cl vs aoa
    slope, intercept = np.polyfit(np.radians(aoa), cl_thin, 1)
    # add slope to dictionary
    airfoil.cl_slopes_dict["Thin Airfoil"] = slope
    airfoil.cl_offsets_dict["Thin Airfoil"] = intercept

    print(fah)
    # Use panel method to estimate Cl
    # first get the x and y coordinates:
    x_contour, y_contour = airfoil.get_coordinates()
    # loop over each aoa since solver expects a single float
    cl_panel = []
    for alpha in aoa:
        panel_dict = solve_closed_contour_panel_method(
            x=x_contour,
            y=y_contour,
            aoa_deg=float(alpha),
            U_inf=1.0,
            kutta_condition=True
        )
        cl_panel.append(panel_dict["Cl"])

    cl_panel = np.array(cl_panel)
    airfoil.cl_dict["Panel Method"] = cl_panel
    # calculate slope of cl vs aoa
    slope_panel, intercept_panel = np.polyfit(np.radians(aoa), cl_panel, 1)
    # add slope to dictionary
    airfoil.cl_slopes_dict["Panel Method"] = slope_panel
    airfoil.cl_offsets_dict["Panel Method"] = intercept_panel


METHOD_LABELS = ["Thin Airfoil", "Panel Method", "XFOIL Free", "XFOIL Fixed"]

# Plot each airfoil with all methods
plot_flexible(
    x_val=aoa,
    y_vals=[
        [cl for cl in af.cl_dict.values() if cl is not None]
        for af in airfoils.values()
    ],
    labels=[
        [label for label, cl in af.cl_dict.items() if cl is not None]
        for af in airfoils.values()
    ],
    x_label="Angle of Attack (degrees)",
    y_units=[f"NACA {code} — Cl [-]" for code in airfoils],
    save_name="cl_vs_aoa_per_airfoil"
)

# Plot each method with all airfoils
plot_flexible(
    x_val=aoa,
    y_vals=[
        [af.cl_dict[method] for af in airfoils.values() if af.cl_dict[method] is not None]
        for method in METHOD_LABELS
        if any(af.cl_dict[method] is not None for af in airfoils.values())
    ],
    labels=[
        [f"NACA {code}" for code, af in airfoils.items() if af.cl_dict[method] is not None]
        for method in METHOD_LABELS
        if any(af.cl_dict[method] is not None for af in airfoils.values())
    ],
    x_label="Angle of Attack (degrees)",
    y_units=[
        method for method in METHOD_LABELS
        if any(af.cl_dict[method] is not None for af in airfoils.values())
    ],
    save_name="cl_vs_aoa_per_method"
)

# print the slopes of each method for each airfoil
# ...existing code...

# print slopes as latex table
available_methods = [m for m in METHOD_LABELS if any(af.cl_slopes_dict.get(m) is not None for af in airfoils.values())]

header = " & ".join(["Airfoil"] + available_methods) + r" \\"
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{l" + "c" * len(available_methods) + "}")
print(r"\hline")
print(header)
print(r"\hline")
for code, af in airfoils.items():
    row = [f"{code}"]
    for method in available_methods:
        slope = af.cl_slopes_dict.get(method)
        row.append(f"{slope:.2f}" if slope is not None else "-")
    print(" & ".join(row) + r" \\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{$dC_l/d\alpha$ per radian for each airfoil and method}")
print(r"\label{tab:cl_slopes}")
print(r"\end{table}")

print("\n\n")
# print offsets as latex table
available_methods = [m for m in METHOD_LABELS if any(af.cl_offsets_dict.get(m) is not None for af in airfoils.values())]

header = " & ".join(["Airfoil"] + available_methods) + r" \\"
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{l" + "c" * len(available_methods) + "}")
print(r"\hline")
print(header)
print(r"\hline")
for code, af in airfoils.items():
    row = [f"{code}"]
    for method in available_methods:
        offset = af.cl_offsets_dict.get(method)
        row.append(f"{offset:.3f}" if offset is not None else "-")
    print(" & ".join(row) + r" \\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Cl offsets for each airfoil and method}")
print(r"\label{tab:cl_offsets}")
print(r"\end{table}")


# ...existing code...
# for code, af in airfoils.items():
#     print(f"NACA {code} Cl slopes:")
#     for method in METHOD_LABELS:
#         slope = af.cl_slopes_dict.get(method)
#         if slope is not None:
#             print(f"  {method}: {slope:.5f} per radian")
# calculate the slope of the linear cl curves
# for airfoil in airfoils.values():
#     cl_thin = airfoil.cl_dict["Thin Airfoil"]
#     slope, intercept = np.polyfit(aoa, cl_thin, 1)
#     print(f"NACA {airfoil.code} Thin Airfoil Cl slope: {slope:.2f} per radian")
#     airfoil.cl_slopes_dict["Thin Airfoil"] = slope

#     # also calculate slope for panel method
#     cl_panel = airfoil.cl_dict["Panel Method"]
#     slope_panel, intercept_panel = np.polyfit(aoa, cl_panel, 1)
#     print(f"NACA {airfoil.code} Panel Method Cl slope: {slope_panel:.2f} per radian")
#     airfoil.cl_slopes_dict["Panel Method"] = slope_panel


# Old plot only Cl
# plot_flexible(x_val=aoa,
#               y_vals=[af.cl_thin_airfoil for af in airfoils.values()],
#               labels=[f"NACA {code} Thin Airfoil" for code in airfoils],
#               x_label="Angle of Attack (degrees)",
#               y_units=["Cl [-]", "Cl [-]", "Cl [-]", "Cl [-]"],
#               save_name="cl_vs_aoa_thin_airfoil")


# control plot to see if they were different
# plot_flexible(x_val=aoa,
#               y_vals=[[af.cl_thin_airfoil for af in airfoils.values()]],
#               labels=[[f"NACA {code} Thin Airfoil" for code in airfoils]],
#               x_label="Angle of Attack (degrees)",
#               y_units=["Cl [-]"],
#               save_name="cl_vs_aoa_thin_airfoil_single_subplot")
# plot Cl vs AoA -10..15 deg

#%% task 2 b
# Evaluate and compare the lift coefficient using panel method

# Use panel method to estimate Cl

# plot_flexible(x_val=aoa,
#               y_vals=[af.cl_panel_method for af in airfoils.values()],
#               labels=[f"NACA {code} Panel Method" for code in airfoils],
#               x_label="Angle of Attack (degrees)",
#               y_units=["Cl [-]", "Cl [-]", "Cl [-]", "Cl [-]"],
#               save_name="cl_vs_aoa_panel_method")



# create airfoil first
# af = NACA4Airfoil("2312")
# # first get the x and y coordinates:
# x_contour, y_contour = af.get_coordinates()
# # loop over each aoa since solver expects a single float
# cl_panel = []
# for alpha in aoa:
#     panel_dict = solve_closed_contour_panel_method(
#         x=x_contour,
#         y=y_contour,
#         aoa_deg=float(alpha),
#         U_inf=1.0,
#         kutta_condition=True
#     )
#     cl_panel.append(panel_dict["Cl"])

# af.cl_panel_method = np.array(cl_panel)

# plot_flexible(x_val=aoa,
#               y_vals=[[af.cl_panel_method]],
#               labels=[[f"NACA 2312 Panel Method"]],
#               x_label="Angle of Attack (degrees)",
#               y_units=["Cl [-]"],
#               save_name="cl_vs_aoa_panel_method_single_subplot")


#%% task 2 c I
# Evaluate and compare the lift coefficient using X foil: free transition BL


#%% task 2 c II
# Evaluate and compare the lift coefficient using X foil: fixed transition BL


