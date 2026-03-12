import subprocess
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
XFOIL_EXE = BASE_DIR / "XFOIL" / "xfoil.exe"

def run_xfoil_dCp(airfoil_code: str, aoa_deg: float, Re: float = 1.5e6,
                   mode: str = "free", n_crit: float = 9.0,
                   xtr: float = 0.1) -> dict:
    """
    Run XFOIL for a single AoA and return the Cp distribution.

    Args:
        airfoil_code: 4-digit NACA code, e.g. '2312'
        aoa_deg:      Angle of attack in degrees
        Re:           Reynolds number
        mode:         'free' (free transition) or 'fixed' (forced transition)
        n_crit:       Ncrit value for free transition (default 9)
        xtr:          Transition location x/c for fixed transition (default 0.1)

    Returns:
        dict with keys:
            'x'    : x/c locations
            'Cp'   : Cp values at each x/c
            'x_upper', 'Cp_upper': upper surface
            'x_lower', 'Cp_lower': lower surface
            'dCp'  : Cp_lower - Cp_upper interpolated onto common grid
            'x_c'  : common x/c grid for dCp
    """
    cp_file = BASE_DIR / f"cp_NACA{airfoil_code}_{mode}_{aoa_deg:.1f}deg.txt"

    # Build VPAR transition command
    if mode == "free":
        vpar_cmd = f"N {n_crit}"
    else:
        vpar_cmd = f"XTR {xtr} {xtr}"

    # Use a simple filename without any path — XFOIL writes to cwd (BASE_DIR)
    cp_filename = f"cp_NACA{airfoil_code}_{mode}_{aoa_deg:.1f}deg.txt"
    cp_file = BASE_DIR / cp_filename  # for reading back after

    commands = "\n".join([
        f"NACA {airfoil_code}",
        "OPER",
        f"VISC {Re}",
        "MACH 0.0",
        "ITER 100",
        "VPAR",
        vpar_cmd,
        "",
        f"ALFA {aoa_deg}",
        "CPWR",
        cp_filename,             # just the filename, no path — XFOIL uses cwd
        "",                      # exit OPER
        "QUIT",
        ""
    ])

    proc = subprocess.run(
        [str(XFOIL_EXE)],
        input=commands,
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )

    if not cp_file.exists():
        print(f"✗ XFOIL failed for NACA {airfoil_code} {mode} at {aoa_deg}°")
        print(proc.stdout[-2000:])
        return None

    # Parse CPWR output: columns are x/c, y/c, Cp
    data = np.loadtxt(cp_file, skiprows=3)
    x_xc = data[:, 0]
    y_xc = data[:, 1]
    Cp   = data[:, 2]


    # XFOIL CPWR goes: upper surface (LE→TE) then lower surface (LE→TE)
    # Split by finding where x/c resets back toward 0 (the LE start of lower surface)
    # Robust method: upper surface has y >= 0, lower has y < 0
    # XFOIL: upper surface y >= 0, lower surface y < 0 — clean split, no TE ambiguity
    upper_mask = y_xc >= 0
    lower_mask = y_xc <  0

    x_upper = x_xc[upper_mask];  Cp_upper = Cp[upper_mask]
    x_lower = x_xc[lower_mask];  Cp_lower = Cp[lower_mask]

    # Sort LE → TE
    x_upper, Cp_upper = x_upper[np.argsort(x_upper)], Cp_upper[np.argsort(x_upper)]
    x_lower, Cp_lower = x_lower[np.argsort(x_lower)], Cp_lower[np.argsort(x_lower)]

    # Interpolate onto common grid, clip TE to avoid singularity
    x_c = np.linspace(0.01, 0.98, 300)
    Cp_upper_interp = np.interp(x_c, x_upper, Cp_upper)
    Cp_lower_interp = np.interp(x_c, x_lower, Cp_lower)

    dCp = Cp_lower_interp - Cp_upper_interp

    print(f"✓ NACA {airfoil_code} | {mode} | AoA={aoa_deg}° | Cl from Cp integral ≈ {np.trapz(dCp, x_c):.4f}")

    return {
        "x": x_xc, "Cp": Cp,
        "x_upper": x_upper, "Cp_upper": Cp_upper,
        "x_lower": x_lower, "Cp_lower": Cp_lower,
        "x_c": x_c, "dCp": dCp,
    }


result_free = run_xfoil_dCp("2312", aoa_deg=10.0, mode="free")
result_fixed = run_xfoil_dCp("2312", aoa_deg=10.0, mode="fixed")

import matplotlib.pyplot as plt
plt.plot(result_free["x_c"], result_free["dCp"], label="XFOIL free transition")
plt.plot(result_fixed["x_c"], result_fixed["dCp"], label="XFOIL fixed transition")
plt.xlabel("x/c [-]")
plt.ylabel("ΔCp [-]")
plt.title("NACA 2312 ΔCp — XFOIL, AoA = 10°")
plt.legend()
plt.grid(True)
plt.show()
