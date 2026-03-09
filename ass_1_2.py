import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoilTheory
from plot import *



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
    airfoil.cl_thin_airfoil =  thin_af.compute_cl(aoa)

plot_flexible(x_val=aoa,
              y_vals=[af.cl_thin_airfoil for af in airfoils.values()],
              labels=[f"NACA {code} Thin Airfoil" for code in airfoils],
              x_label="Angle of Attack (degrees)",
              y_units=["Cl [-]", "Cl [-]", "Cl [-]", "Cl [-]"],
              save_name="cl_vs_aoa_thin_airfoil")

plot_flexible(x_val=aoa,
              y_vals=[[af.cl_thin_airfoil for af in airfoils.values()]],
              labels=[[f"NACA {code} Thin Airfoil" for code in airfoils]],
              x_label="Angle of Attack (degrees)",
              y_units=["Cl [-]"],
              save_name="cl_vs_aoa_thin_airfoil_single_subplot")
# plot Cl vs AoA -10..15 deg

#%% task 2 b
# Evaluate and compare the lift coefficient using panel method


#%% task 2 c I
# Evaluate and compare the lift coefficient using X foil: free transition BL


#%% task 2 c II
# Evaluate and compare the lift coefficient using X foil: fixed transition BL


