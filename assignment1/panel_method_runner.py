import pandas as pd
import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *
from panel_method.exercise.solver import compute_dCp_panel_method, solve_closed_contour_panel_method, compute_dCp_panel, get_dCp
from xfoils_class import xfoils_free, xfoils_fixed
from tables import *
from panel_method.funaerotool.utils import generate_naca4_contour

airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}
aoa = np.arange(-10, 16, 1)  # AoA from -10 to 15 degrees



for airfoil in airfoils.values():
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

    # get the dCp distribution for AoA = 10 degrees
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
    print(f"Panel method dCp for NACA {airfoil.code} at AoA=10 deg: {dCp}")
    print(f"Panel method x/c for NACA {airfoil.code} at AoA=10 deg: {x_dcp}")

    airfoil.xc_dict ["Panel Method"] = x_dcp
    airfoil.dCp_dict ["Panel Method"] = dCp


plot_flexible(
    x_val=[[af.xc_dict["Panel Method"]] for af in airfoils.values()],
    y_vals=[[af.dCp_dict["Panel Method"]] for af in airfoils.values()],
    labels=[[f"NACA {af.code}"] for af in airfoils.values()],
    x_label="x/c [-]",
    y_units=[f"NACA {af.code} — ΔCp [-]" for af in airfoils.values()],
    save_name="dCp_vs_xc_panel_method"
)

plot_flexible(
    x_val=aoa,
    y_vals=[[af.cl_dict["Panel Method"]] for af in airfoils.values()],
    labels=[[f"NACA {af.code} Panel Method"] for af in airfoils.values()],
    x_label="AoA [deg]",
    y_units=[f"NACA {af.code} — Cl [-]" for af in airfoils.values()],
    save_name="Cl_vs_AoA_panel_method"
)