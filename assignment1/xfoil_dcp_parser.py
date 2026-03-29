from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent

airfoils = ["2312", "2324", "4412", "4424"]
AOA = 10.0
transition_modes = ["free", "fixed"]

def parse_xfoil_cp(airfoil: str, mode: str, aoa: float = AOA) -> dict | None:
    """Parse a CPWR output file and return Cp distribution.

    Args:
        airfoil: 4-digit NACA code string, e.g. '2312'
        mode:    'free' or 'fixed'
        aoa:     Angle of attack in degrees

    Returns:
        dict with x_c, dCp, x_upper, Cp_upper, x_lower, Cp_lower
        or None if file not found.
    """
    cp_file = BASE_DIR / "XFOIL" / "Cp" / mode / f"cp_NACA{airfoil}_{mode}_{aoa:.1f}deg.txt"


    if not cp_file.exists():
        print(f"  ✗ File not found: {cp_file.name} — run xfoil_dcp_runner.py first")
        return None

    data = np.loadtxt(cp_file, skiprows=3)

    x = data[:, 0]
    Cp   = data[:, 2]

    return {
        "x":      x,
        "Cp":      Cp
    }


def parse_xfoil_dcp(airfoil: str, mode: str, aoa: float = AOA) -> dict | None:
    """Parse a CPWR output file and return Cp distribution.

    Args:
        airfoil: 4-digit NACA code string, e.g. '2312'
        mode:    'free' or 'fixed'
        aoa:     Angle of attack in degrees

    Returns:
        dict with x_c, dCp, x_upper, Cp_upper, x_lower, Cp_lower
        or None if file not found.
    """
    cp_file = BASE_DIR / "XFOIL" / "Cp" / mode / f"cp_NACA{airfoil}_{mode}_{aoa:.1f}deg.txt"


    if not cp_file.exists():
        print(f"  ✗ File not found: {cp_file.name} — run xfoil_dcp_runner.py first")
        return None

    data = np.loadtxt(cp_file, skiprows=3)
    x_xc = data[:, 0]
    y_xc = data[:, 1]
    Cp   = data[:, 2]

    upper_mask = y_xc >= 0
    lower_mask = y_xc <  0

    x_upper, Cp_upper = x_xc[upper_mask], Cp[upper_mask]
    x_lower, Cp_lower = x_xc[lower_mask], Cp[lower_mask]

    x_upper, Cp_upper = x_upper[np.argsort(x_upper)], Cp_upper[np.argsort(x_upper)]
    x_lower, Cp_lower = x_lower[np.argsort(x_lower)], Cp_lower[np.argsort(x_lower)]

    x_c = np.linspace(0.01, 0.98, 300)
    Cp_upper_interp = np.interp(x_c, x_upper, Cp_upper)
    Cp_lower_interp = np.interp(x_c, x_lower, Cp_lower)

    return {
        "x_c":      x_c,
        "dCp":      Cp_lower_interp - Cp_upper_interp,
        "x_upper":  x_upper,  "Cp_upper": Cp_upper,
        "x_lower":  x_lower,  "Cp_lower": Cp_lower,
    }


def load_all_xfoil_dcp(aoa: float = AOA) -> dict:
    """Load dCp results for all airfoils and both transition modes.

    Returns:
        results[airfoil][mode] = dict from parse_xfoil_cp()
    """
    return {
        af: {mode: parse_xfoil_dcp(af, mode, aoa) for mode in transition_modes}
        for af in airfoils
    }


def load_all_xfoil_Cp(aoa: float = AOA):
    return {
        af: {mode: parse_xfoil_cp(af, mode, aoa) for mode in transition_modes}
        for af in airfoils
        }