import numpy as np
import matplotlib.pyplot as plt
from panel_method.funaerotool.utils import generate_naca4_contour
from xfoil_reader import XFoil, load_xfoil_data


Re = 5e6 # Reynolds number
code = "2410" # naca airfoil code
AR = np.array([4, 6, 8, 10, np.inf]) # create aspect ratio array, including infinite aspect ratio for 2D case
AoA = np.linspace(-4, 8, 13) # create angle of attack array
TR = np.linspace(0.2, 1, 5) # create taper ratios


# plan_forms = ["Elliptic", "Tapered", "Constant Chord"]
plan_form = "Elliptic"
x_contour, y_contour = generate_naca4_contour(code, n_points=101)

plt.plot(x_contour, y_contour, label=f"NACA {code}")
plt.title(f"NACA {code}")
plt.axis('equal')
plt.legend()
plt.xlabel("x/c")
plt.ylabel("y/c")
plt.grid(True)
plt.tight_layout()
plt.savefig("airfoil_comparison.png", dpi=300)
plt.close()