"""thin airfoil theory"""
import numpy as np
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt
from plot import *


class ThinAirfoilTheory:
    """Computes lift coefficient and pressure difference using thin airfoil theory."""

    def __init__(self, airfoil: NACA4Airfoil):
        self.airfoil = airfoil
        self.c = airfoil.c
        self.x = airfoil.x / self.c
        self.camber_slope = airfoil.camber_slope
        self.theta = np.arccos(1 - 2 * self.x)

    def _a0(self, alpha: float) -> float:
        """A0 Fourier coefficient."""
        return alpha - 1/np.pi * np.trapezoid(self.camber_slope, self.theta)

    def _an(self, n: int) -> float:
        """General An Fourier coefficient for n >= 1.
        
        Args:
            n: Fourier mode number (1, 2, 3, ...)
        Returns:
            An coefficient (float)
        """
        return 2/np.pi * np.trapezoid(self.camber_slope * np.cos(n * self.theta), self.theta)

    def _gamma(self, alpha: float, U0: float = 1.0, N_terms: int = 50) -> np.ndarray:
        """Vortex sheet strength distribution gamma(theta).

        Uses the Fourier series:
            gamma(theta) = 2*U0 * [A0*(1+cos(theta))/sin(theta) + sum(An*sin(n*theta))]

        Args:
            alpha:   Angle of attack in radians.
            U0:      Freestream velocity (default 1.0, cancels in dCp).
            N_terms: Number of Fourier terms in the series summation.
        Returns:
            gamma: Array of vortex sheet strength values over theta grid.
        """
        a0 = self._a0(alpha)

        # Leading edge (theta=0) and trailing edge (theta=pi) are singular/zero,
        # so exclude endpoints to avoid division by zero in sin(theta)
        theta = self.theta[1:-1]

        # A0 term: singular at LE, well-behaved away from it
        gamma = a0 * (1 + np.cos(theta)) / np.sin(theta)

        # Sum An*sin(n*theta) for n=1..N_terms
        for n in range(1, N_terms + 1):
            gamma += self._an(n) * np.sin(n * theta)

        return 2 * U0 * gamma

    def compute_dCp(self, aoa_deg: float, U0: float = 1.0, N_terms: int = 50) -> tuple:
        """Compute pressure difference coefficient delta_Cp = 2*gamma/U0.

        Since dCp = (p_lower - p_upper) / (0.5*rho*U0^2) = 2*gamma/U0,
        and gamma ~ U0, the result is independent of U0.

        Args:
            aoa_deg: Angle of attack in degrees.
            U0:      Freestream velocity (default 1.0).
            N_terms: Number of Fourier terms in the series summation.
        Returns:
            x_c:  x/c coordinates (excluding LE and TE endpoints).
            dCp:  Pressure difference coefficient at each x/c location.
        """
        alpha = np.radians(aoa_deg)
        gamma = self._gamma(alpha, U0, N_terms)
        dCp = 2 * gamma / U0            # U0 cancels — kept explicit for clarity
        x_c = self.x[1:-1]             # match interior points used in _gamma
        return x_c, dCp

    def _a1(self) -> float:
        """A1 coefficient (convenience wrapper around _an)."""
        return self._an(1)

    def _cl(self, alpha: float) -> float:
        """Lift coefficient for a given angle of attack."""
        return 2 * np.pi * (self._a0(alpha) + self._an(1) / 2)

    def compute_cl(self, aoa_deg):
        """Compute Cl for an array of angles of attack in degrees."""
        aoa_rad = np.radians(aoa_deg)
        self.cl_vals = np.array([self._cl(alpha) for alpha in aoa_rad])
        return self.cl_vals


# implement the thin airfoil theory to calculate Cl and dCp
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

    
    # calculate dCp for thin airfoil theory for AoA = 10 degrees
    x_c, dCp = thin_af.compute_dCp(aoa_deg=10)
    airfoil.xc_dict ["Thin Airfoil"] = x_c
    airfoil.dCp_dict ["Thin Airfoil"] = dCp
    

plot_flexible(
    x_val=[[af.xc_dict["Thin Airfoil"]] for af in airfoils.values()],
    y_vals=[[af.dCp_dict["Thin Airfoil"]] for af in airfoils.values()],
    labels=[[f"NACA {af.code}"] for af in airfoils.values()],
    x_label="x/c [-]",
    y_units=[f"NACA {af.code} — ΔCp [-]" for af in airfoils.values()],
    save_name="dCp_vs_xc_thin_airfoil"
)

plot_flexible(
    x_val=aoa,
    y_vals=[[af.cl_dict["Thin Airfoil"]] for af in airfoils.values()],
    labels=[[f"NACA {af.code} Thin Airfoil"] for af in airfoils.values()],
    x_label="AoA [deg]",
    y_units=[f"NACA {af.code} — Cl [-]" for af in airfoils.values()],
    save_name="Cl_vs_AoA_thin_airfoil"
)
