import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_airfoil(file):
    # file has: x y  (whole surface, upper then lower)
    df = pd.read_csv(
        file,
        sep=r'\s+',
        names=["x", "y"],
        skiprows=1,
    )

    # index of leading edge = minimum x
    i_le = df["x"].idxmin()

    # upper: TE -> LE, lower: LE -> TE
    upper = df.iloc[:i_le+1].copy()
    lower = df.iloc[i_le:].copy()

    # sort both surfaces to go from x = 0 -> 1
    upper = upper.sort_values("x")
    lower = lower.sort_values("x")

    # put them on the same x-grid (use upper.x as reference)
    x = upper["x"].values
    y_upper = upper["y"].values
    y_lower = np.interp(x, lower["x"].values, lower["y"].values)

    # camber line
    camber = 0.5 * (y_upper + y_lower)

    return x, y_upper, y_lower, camber

x, y_u, y_l, c_line = read_airfoil("AirfoilData/NACA2312.txt")
x = np.clip(x, -1, 1)
c = 1.0  # chord length

# Transform x to theta using cosine transformation: x = c/2 * (1 - cos(theta))
# Solving for theta: cos(theta) = 1 - 2*x/c, so theta = arccos(1 - 2*x/c)
theta = np.arccos(1 - 2*x/c)

# plt.plot(x, np.degrees(theta))
# plt.show()

print("Theta values (radians):", theta)

# differentiate camber line to get camber slope
dc_dtheta = np.gradient(c_line, theta)

aoa = np.arange(-10, 16, 1)  # Changed step from 26 to 1 for reasonable range
print(aoa)


def a0(alpha, dc_dtheta, theta):
    return alpha - 1/np.pi * np.trapezoid(dc_dtheta, theta)

def a1(dc_dtheta, theta):
    return 2/np.pi * np.trapezoid(dc_dtheta * np.cos(theta), theta)

def cl(a0, a1):
    return 2 * np.pi * (a0 + a1/2)

a0_vals = np.zeros(len(aoa))
a1_vals = np.zeros(len(aoa))
cl_vals = np.zeros(len(aoa))

# calculate a0 vals
for angle in aoa:
    alpha_rad = np.radians(angle)
    a0_val = a0(alpha_rad, dc_dtheta, theta)
    a1_val = a1(dc_dtheta, theta)
    a0_vals = np.append(a0_vals, a0_val)
    a1_vals = np.append(a1_vals, a1_val)
    cl_val = cl(a0_val, a1_val)
    cl_vals = np.append(cl_vals, cl_val)


print("Cl values:", cl_vals)

# plot cl vs aoa
plt.figure(figsize=(8, 5))
plt.plot(aoa, cl_vals, marker='o')
plt.title("Lift Coefficient (Cl) vs Angle of Attack (AoA)")
plt.xlabel("Angle of Attack (degrees)")
plt.ylabel("Lift Coefficient (Cl)")
plt.grid()
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            





# print("a0 values:", a0_vals)