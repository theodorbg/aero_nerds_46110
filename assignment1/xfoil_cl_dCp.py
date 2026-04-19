
from turtle import pd
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
from xfoil_dcp_parser import load_all_xfoil_dcp

xfoil_results = load_all_xfoil_dcp()

# implement the thin airfoil theory to calculate Cl and dCp
METHOD_LABELS = ["Thin Airfoil", "Panel Method", "XFOIL Free", "XFOIL Fixed"]

# task 1
# create dictionary to store the airfoils
airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}

aoa = np.arange(-10, 16, 1)  # AoA from -10 to 15 degrees
for airfoil in airfoils.values():
    airfoil.dCp_dict["XFOIL Free"] = xfoil_results[airfoil.code]["free"]["dCp"]
    airfoil.xc_dict["XFOIL Free"] = xfoil_results[airfoil.code]["free"]["x_c"]

    airfoil.dCp_dict["XFOIL Fixed"] = xfoil_results[airfoil.code]["fixed"]["dCp"]
    airfoil.xc_dict["XFOIL Fixed"] = xfoil_results[airfoil.code]["fixed"]["x_c"]

    
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



plot_flexible(
    x_val=[
        [af.xc_dict["XFOIL Free"], af.xc_dict["XFOIL Fixed"]]
        for af in airfoils.values()
    ],
    y_vals=[
        [af.dCp_dict["XFOIL Free"], af.dCp_dict["XFOIL Fixed"]]
        for af in airfoils.values()
    ],
    labels=[
        ["XFOIL Free", "XFOIL Fixed"]
        for af in airfoils.values()
    ],
    x_label="x/c [-]",
    y_units=[f"NACA {af.code} — ΔCp [-]" for af in airfoils.values()],
    save_name="dCp_vs_xc_xfoil_combined"
)

plot_flexible(
    x_val=aoa,
    y_vals=[
        [af.cl_dict["XFOIL Free"], af.cl_dict["XFOIL Fixed"]]
        for af in airfoils.values()
    ],
    labels=[
        ["XFOIL Free", "XFOIL Fixed"]
        for af in airfoils.values()
    ],
    x_label="AoA [deg]",
    y_units=[f"NACA {af.code} — Cl [-]" for af in airfoils.values()],
    save_name="Cl_vs_AoA_xfoil_combined"
)



