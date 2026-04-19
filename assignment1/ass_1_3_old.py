import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *
from panel_method.exercise.solver import compute_dCp_panel_method, solve_closed_contour_panel_method, compute_dCp_panel
from xfoils_class import xfoils_free, xfoils_fixed
from tables import *
from panel_method.funaerotool.utils import generate_naca4_contour




af = NACA4Airfoil("2312")


# Thin Airfoil
thin_af = ThinAirfoilTheory(af)
af.dCp_xc_tat, af.dCp_thin_airfoil = thin_af.compute_dCp(aoa_deg=10)

# Panel method
# Generate NACA 4412 airfoil contour
x, y = generate_naca4_contour("2312", n_points=401)
# calculate chord length
chord = max(x) - min(x)
print(f"Chord length: {chord}") 
results = solve_closed_contour_panel_method(x=x, y=y, aoa_deg=10)
print(f"Panel method results: {results}")
af.dCp_panel_method = results["Cp"]
af.dCp_xc_pm = results["xp"]

# af.dCp_xc_pm, af.dCp_pm = compute_dCp_panel(airfoil=af, aoa_deg=10.0, U_inf=1.0, n_interp=200)
# xc_common, dCp_chat = compute_dCp_panel_method(af, aoa_deg=10.0, U_inf=1.0)

from xfoil_dcp_parser import load_all_xfoil_dcp

xfoil_results = load_all_xfoil_dcp()
# Access individual result
af.dCp_xfoil_free = xfoil_results["2312"]["free"]["dCp"]
af.dCp_xfoil_fixed = xfoil_results["2312"]["fixed"]["dCp"]
af.dCp_xc_xfoil_free = xfoil_results["2312"]["free"]["x_c"]
af.dCp_xc_xfoil_fixed = xfoil_results["2312"]["fixed"]["x_c"]




plot_flexible(
    x_val=[[af.dCp_xc_tat, af.dCp_xc_pm, af.dCp_xc_xfoil_free, af.dCp_xc_xfoil_fixed]],       # 1 subplot, 2 x arrays
    y_vals=[[af.dCp_thin_airfoil, af.dCp_panel_method, af.dCp_xfoil_free, af.dCp_xfoil_fixed]], # 1 subplot, 2 y arrays
    labels=[["Thin Airfoil Theory", "Panel Method", "XFOIL_free", "XFOIL_fixed"]],  # 1 subplot, 4 labels
    x_label="x/c [-]",
    y_units=["ΔCp [-]"],
    save_name="dCp_vs_xc_single_airfoil",
    xlims=(0.0, 0.98),
    ylims=[(0, 10)]
)

