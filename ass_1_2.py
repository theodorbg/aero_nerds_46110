import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *
from panel_method.exercise.solver import compute_dCp_panel_method, solve_closed_contour_panel_method, compute_dCp_panel
from xfoils_class import xfoils_free, xfoils_fixed
from tables import *

METHOD_LABELS = ["Thin Airfoil", "Panel Method", "XFOIL Free", "XFOIL Fixed"]

# task 1
# create dictionary to store the airfoils
airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}


# task 2 a
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

    
    # Use panel method to estimate Cl
    # first get the x and y coordinates:
    x_contour, y_contour = airfoil.get_closed_contour()
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

    # calculate dCp for thin airfoil theory for AoA = 10 degrees
    x_c, dCp = thin_af.compute_dCp(aoa_deg=10)
    airfoil.dCp_thin_airfoil = dCp
    airfoil.dCp_xc_tat = x_c

    x_c, dCp = compute_dCp_panel(airfoil=airfoil, aoa_deg=10.0, U_inf=1.0, n_interp=200)
    airfoil.dCp_panel_method = dCp
    airfoil.dCp_xc_pm = x_c

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
    save_name="cl_vs_aoa_per_airfoil"
)


# Plot dCp for each airfoil with all methods
# choose one of the x/c distributions to use as the shared x-axis for each method
x_c_tat = airfoils["2312"].dCp_xc_tat  # using thin airfoil theory x/c as reference
x_c_pm = airfoils["2312"].dCp_xc_pm  # using panel method x/c as reference
# x_c_free
# x_c_fixed
plot_flexible(
    x_val=[
        [af.dCp_xc_tat, af.dCp_xc_pm]
        for af in airfoils.values()
    ],
    y_vals=[
        [af.dCp_thin_airfoil, af.dCp_panel_method]
        for af in airfoils.values()
    ],
    labels=[
        ["Thin Airfoil", "Panel Method"]
        for af in airfoils.values()
    ],
    x_label="x/c [-]",
    y_units=[f"NACA {code} — ΔCp [-]" for code in airfoils],
    xlims=(0.02, 0.98),
    ylims=[(0,10) for _ in airfoils],
    save_name="dCp_vs_xc_per_airfoil"
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


# help me use the panel method to estimate
# the pressure distribution (dCp) over the airfoil surface at a given angle of attack (e.g., 10 degrees). 
# Then, plot the dCp distribution as a function of x/c for an airfoil
"""
help me calculate the pressure difference distribution deltaCp as a function of x/c for an airfoil.
dCP=(p_upper-plower)/(0.5*rho*U0^2), for AoA=10 degrees.
I need to use equation:
dCP=2*gamma/U0
and
gamma(theta)= 2*U0*(A0*(1+cos(theta))/sin(theta)+sum(An*sin(n*theta),n=1..infty))

where the coefficients are given by:
A0=alpha-1/pi*int(camber_slope*dtheta,theta=0..pi)
and
An=2/pi*int(camber_slope*cos(n*theta)detheta,theta=0..pi), n=1,2,3...

I need to use thin airfoil theory, panel method and xfoil.
My best guess is that this is thin airfoil theory since the equations look very similar to when i computed the lift coefficient, but i dont know if i can use it for panel method also ? And what about xfoil, do i get the dCp directly with XFOIL or should i compute something as well here?

Also, what do i use for U0 / Qinfinity ? Its not given in the assignment. The assignment is for an aerodynamics engineering course
"""
af_test = airfoils["2312"]

x_c, dCp_perplex = compute_dCp_panel(airfoil=af_test, aoa_deg=10.0, U_inf=1.0, n_interp=200)
xc_common, dCp_chat = compute_dCp_panel_method(af_test, aoa_deg=10.0, U_inf=1.0)

plot_flexible(
    x_val=[[xc_common,x_c]],
    y_vals=[dCp_chat, dCp_perplex],
    labels=[f"NACA {af_test.code} Panel Method", f"NACA {af_test.code} Panel Method"],
    x_label="x/c [-]",
    y_units=["ΔCp [-]"],
    save_name="dCp_vs_xc_single_airfoil",
    xlims=(0, 0.1),
    ylims=[(-0.1, 1)]
)


