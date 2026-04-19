from turtle import pd
import pandas as pd
import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *
from panel_method.exercise.solver import compute_dCp_panel_method, solve_closed_contour_panel_method, compute_dCp_panel, get_dCp
from xfoils_class import load_xfoils
xfoils_free, xfoils_fixed = load_xfoils("free_trans", "fixed_xtr_25")
# check length of xfoil data
for code in xfoils_free:
    print(f"NACA {code} free transition: {len(xfoils_free[code].alpha)} AoA points")
for code in xfoils_fixed:
    print(f"NACA {code} fixed transition: {len(xfoils_fixed[code].alpha)} AoA points")

from tables import *
from panel_method.funaerotool.utils import generate_naca4_contour


METHOD_LABELS = ["Thin Airfoil", "Panel Method", "XFOIL Free", "XFOIL Fixed"]

# task 1
# create dictionary to store the airfoils
airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}


# task 2
# Cl Lift Coefficient vs AoA
aoa = np.linspace(-10, 15, 26)  # replace with your actual AoA array if different

for airfoil in airfoils.values():
    # create thin airfoil theory instance
    thin_af = ThinAirfoilTheory(airfoil)

    # Use thin airfoil theory to estimate Cl

    airfoil.cl_dict["Thin Airfoil"] = thin_af.compute_cl(aoa)
    # calculate slope of cl vs aoa
    slope, intercept = np.polyfit(np.radians(aoa), airfoil.cl_dict["Thin Airfoil"], 1)
    # add slope to dictionary
    airfoil.cl_slopes_dict["Thin Airfoil"] = slope
    airfoil.cl_offsets_dict["Thin Airfoil"] = intercept

    
    cl_panel = []
    x_contour, y_contour = generate_naca4_contour(airfoil.code, n_points=101)
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
    

# add free transition xfoil data to dictionary
for code, xfoil in xfoils_free.items():
    if code in airfoils:
        airfoil = airfoils[code]
        airfoil.cl_dict["XFOIL Free"] = xfoil.CL
        # slope_xfoil_free, intercept_xfoil_free = np.polyfit(np.radians(xfoil.alpha), xfoil.CL, 1)
        # airfoil.cl_slopes_dict["XFOIL Free"] = slope_xfoil_free
        # airfoil.cl_offsets_dict["XFOIL Free"] = intercept_xfoil_free

# add fixed transition xfoil data to dictionary
for code, xfoil in xfoils_fixed.items():
    if code in airfoils:
        airfoil = airfoils[code]
        airfoil.cl_dict["XFOIL Fixed"] = xfoil.CL



# Plot Cl for each airfoil with all methods
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
    save_name="cl_vs_aoa_per_airfoil")


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
print_latex_table(
    airfoils, METHOD_LABELS,
    data_attr="cl_slopes_dict",
    caption=r"$dC_l/d\alpha$ per radian for each airfoil and method",
    label="tab:cl_slopes",
    fmt=".2f"
)

print("\n\n")

print_latex_table(
    airfoils, METHOD_LABELS,
    data_attr="cl_offsets_dict",
    caption="Cl offsets for each airfoil and method",
    label="tab:cl_offsets",
    fmt=".3f"
)

