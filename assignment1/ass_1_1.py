import numpy as np
from pyparsing import line
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt

# task 1
#. In the same figure plot and compare the geometries of the NACA 2312 and 2324 airfoils, 
# and the NACA 4412 and 4424 airfoils, both including also the chamber line. 
# **Remember to use axis equal to ensure that the x and y axes have a 1:1 aspect. 


# task 1
# create dictionary to store the airfoils
airfoils = {"2312": None, "2324": None, "4412": None, "4424": None}
# generate the airfoils
airfoils = {code: NACA4Airfoil(code) for code in airfoils}
colors = ["blue", "orange", "green", "red"]

fig, axes = plt.subplots(2, 1, figsize=(16, 9))
for ax, codes in zip(axes, [("2312", "2324"), ("4412", "4424")]):
    for code, color in zip(codes, colors):
        af = airfoils[code]
        ax.plot(af.xu, af.yu, color=color, label=f"NACA {code}")
        ax.plot(af.xl, af.yl, color=color)
        ax.plot(af.x, af.yc, color=color, linestyle='--', alpha=0.6, label=f"NACA {code} camberline")
        ax.set_title(f"NACA {codes[0][:2]}-Series")
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_ylim(-0.15, 0.2)
    ax.set_xlim(-0.1, 1.1)
    ax.grid(True)

plt.tight_layout()
plt.savefig("airfoil_comparison.png", dpi=300)
plt.close()

