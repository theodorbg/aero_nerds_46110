import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *
from panel_method.exercise.solver import solve_closed_contour_panel_method, get_dCp
from xfoils_class import load_xfoils
from tables import *
from panel_method.funaerotool.utils import generate_naca4_contour
from xfoil_dcp_parser import load_all_xfoil_Cp

METHOD_LABELS = ["Panel Method", "XFOIL Free", "XFOIL Fixed"]

xfoil_results = load_all_xfoil_Cp(aoa= 10.0)


# create dictionary to store the airfoils
airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}


# task 4
# Get the Cp distribution for AoA = 10 degrees for all airfoils and all methods

for airfoil in airfoils.values():

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

    airfoil.xp_dict ["Panel Method"] = panel_dict["xp"] 
    airfoil.Cp_dict ["Panel Method"] = panel_dict["Cp"]

    #%% get xfoil results and store them
    airfoil.Cp_dict["XFOIL Free"] = xfoil_results[airfoil.code]["free"]["Cp"]
    airfoil.xp_dict["XFOIL Free"] = xfoil_results[airfoil.code]["free"]["x"]

    airfoil.Cp_dict["XFOIL Fixed"] = xfoil_results[airfoil.code]["fixed"]["Cp"]
    airfoil.xp_dict["XFOIL Fixed"] = xfoil_results[airfoil.code]["fixed"]["x"]

#%% Plot the results
# Plot dCp for each airfoil with all methods
plot_flexible(
    x_val=[
        [xp for xp in af.xp_dict.values() if xp is not None]
        for af in airfoils.values()
    ],
    y_vals=[
        [Cp for Cp in af.Cp_dict.values() if Cp is not None]
        for af in airfoils.values()
    ],
    labels=[
        [label for label, Cp in af.Cp_dict.items() if Cp is not None]
        for af in airfoils.values()
    ],
    x_label="x/c [-]",
    y_units=[f"NACA {code} — Cp [-]" for code in airfoils],
    save_name="Cp_contour_per_airfoil",
    ylims=[(-5.5, 1.5), (-3.5, 1.5), (-5.5, 1.5), (-3.5, 1.5)],
    xlims=(0,1)
)

plot_flexible(
    x_val=[
        [af.xp_dict[method] for af in airfoils.values()]
        for method in METHOD_LABELS
    ],
    y_vals=[
        [af.Cp_dict[method] for af in airfoils.values()]
        for method in METHOD_LABELS
    ],
    labels=[
        [f"NACA {code}" for code in airfoils.keys()]
        for _ in METHOD_LABELS
    ],
    x_label="x/c [-]",
    y_units=[f"{method} — Cp [-]" for method in METHOD_LABELS],
    save_name="Cp_contour_per_method",
    ylims=[(-5.5, 1.5) for _ in METHOD_LABELS],
    xlims=(0,1)
)


