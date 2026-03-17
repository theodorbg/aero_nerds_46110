import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *
from panel_method.exercise.solver import solve_closed_contour_panel_method, get_dCp
from tables import *
from panel_method.funaerotool.utils import generate_naca4_contour
from xfoil_dcp_parser import load_all_xfoil_dcp

METHOD_LABELS = ["Thin Airfoil", "Panel Method", "XFOIL Free", "XFOIL Fixed"]

xfoil_results = load_all_xfoil_dcp()


# create dictionary to store the airfoils
airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}


# task 3
# calculate dCp distribution for AoA = 10 degrees for all airfoils and all methods

for airfoil in airfoils.values():

    #%% Use thin af theory
    # create thin airfoil theory instance
    thin_af = ThinAirfoilTheory(airfoil)
    
    # calculate dCp for thin airfoil theory for AoA = 10 degrees
    x_c, dCp = thin_af.compute_dCp(aoa_deg=10)
    airfoil.xc_dict ["Thin Airfoil"] = x_c
    airfoil.dCp_dict ["Thin Airfoil"] = dCp

    #%% Use panel method

    # generate airfoil contour
    x_contour, y_contour = generate_naca4_contour(airfoil.code, n_points=101)

    panel_dict = solve_closed_contour_panel_method(
            x=x_contour,
            y=y_contour,
            aoa_deg=10.0,
            U_inf=1.0,
            kutta_condition=True
        )
    xp = panel_dict["xp"] 
    yp = panel_dict["yp"]
    Cp = panel_dict["Cp"]

    x_dcp, dCp = get_dCp(xp, yp, Cp)

    airfoil.xc_dict ["Panel Method"] = x_dcp
    airfoil.dCp_dict ["Panel Method"] = dCp

    #%% get xfoil results and store them
    airfoil.dCp_dict["XFOIL Free"] = xfoil_results[airfoil.code]["free"]["dCp"]
    airfoil.xc_dict["XFOIL Free"] = xfoil_results[airfoil.code]["free"]["x_c"]

    airfoil.dCp_dict["XFOIL Fixed"] = xfoil_results[airfoil.code]["fixed"]["dCp"]
    airfoil.xc_dict["XFOIL Fixed"] = xfoil_results[airfoil.code]["fixed"]["x_c"]



#%% Plot the results
# Plot dCp for each airfoil with all methods
plot_flexible(
    x_val=[
        [xc for xc in af.xc_dict.values()]
        for af in airfoils.values()
    ],
    y_vals=[
        [dCp for dCp in af.dCp_dict.values()]
        for af in airfoils.values()
    ],
    labels=[
        [label for label, dCp in af.dCp_dict.items()]
        for af in airfoils.values()
    ],
    x_label="x/c [-]",
    y_units=[f"NACA {code} —  ΔCp  [-]" for code in airfoils],
    ylims=[(-0.2,5) for _ in airfoils],
    save_name="dCp_per_airfoil")


# Plot each method with all airfoils
plot_flexible(
    x_val=[
        [af.xc_dict[method] for af in airfoils.values()]
        for method in METHOD_LABELS
    ],
    y_vals=[
        [af.dCp_dict[method] for af in airfoils.values()]
        for method in METHOD_LABELS
    ],
    labels=[
        [f"NACA {code}" for code in airfoils.keys()]
        for _ in METHOD_LABELS
    ],
    x_label="x/c [-]",
    y_units=[f"{method} — ΔCp [-]" for method in METHOD_LABELS],
    save_name="dCp_per_method",
    ylims=[(-0.2, 5) for _ in METHOD_LABELS]
)

# ...existing code...
