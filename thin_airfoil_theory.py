"""thin airfoil theory"""
import numpy as np
from airfoils import NACA4Airfoil
import matplotlib.pyplot as plt


class ThinAirfoilTheory:
    """Computes lift coefficient using thin airfoil theory."""

    def __init__(self, airfoil: NACA4Airfoil):
        """Initialize with a NACA4Airfoil instance.
        
        Args:
            airfoil: An instance of NACA4Airfoil containing camber line data.
        """
        self.airfoil = airfoil
        self.c = airfoil.c
        self.x = airfoil.x / self.c          # normalize to [0, 1]
        self.camber_slope = airfoil.camber_slope
        self.theta = np.arccos(1 - 2 * self.x) # theta distribution for integration

    def _a0(self, alpha: float):
        """Calculate a0 coefficient for thin airfoil theory.
        Args:
            alpha: Angle of attack in radians.

        Returns:
            a0 coefficient for lift calculation. (float)
        """
        return alpha - 1/np.pi * np.trapezoid(self.camber_slope, self.theta)

    def _a1(self):
        """Calculate a1 coefficient for thin airfoil theory.
        
        Returns:
            a1 coefficient for lift calculation. (float)
        """

        return 2/np.pi * np.trapezoid(self.camber_slope * np.cos(self.theta), self.theta)

    def _cl(self, alpha: float):
        """Calculate lift coefficient (Cl) for a given angle of attack.

        Args:
            alpha: Angle of attack in radians.
        Returns:
                Lift coefficient (Cl) (float)
        """
        return 2 * np.pi * (self._a0(alpha) + self._a1() / 2)

    def compute_cl(self, aoa_deg):
        """Compute Cl for an array of angles of attack in degrees.
        
        Args:
            aoa_deg: Array of angles of attack in degrees.
        Returns:
            Array of lift coefficients (Cl) for the given angles of attack.
        """
        aoa_rad = np.radians(aoa_deg)

        self.cl_vals = np.array([self._cl(alpha) for alpha in aoa_rad])
        
        return self.cl_vals

    def plot_cl(self, aoa_range=(-10, 16)):
        aoa = np.arange(*aoa_range, 1)
        cl_vals = self.compute_cl(aoa)

        plt.figure(figsize=(8, 5))
        plt.plot(aoa, cl_vals, marker='o')
        plt.title(f"Cl vs AoA — NACA {self.airfoil.code}")
        plt.xlabel("Angle of Attack (degrees)")
        plt.ylabel("Lift Coefficient (Cl)")
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.grid()
        plt.show()


# Example usage
if __name__ == "__main__":
    af = NACA4Airfoil("2412")
    tat = ThinAirfoilTheory(af)
    tat.plot_cl()