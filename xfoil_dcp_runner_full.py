import subprocess
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
XFOIL_EXE = BASE_DIR / "XFOIL" / "xfoil.exe"

airfoils = ["2312", "2324", "4412", "4424"]
AOA = 10.0

transition_modes = {
    "free":  {"vpar_cmds": "N 9"},
    "fixed": {"vpar_cmds": "XTR 0.1 0.1"},
}

# Nested dict to store results_dCp_xfoil: results_dCp_xfoil[airfoil][mode] = dict
results_dCp_xfoil = {af: {} for af in airfoils}

for airfoil in airfoils:
    for mode_name, mode in transition_modes.items():
        cp_filename = f"cp_NACA{airfoil}_{mode_name}_{AOA:.1f}deg.txt"
        cp_file = BASE_DIR / cp_filename

        commands = "\n".join([
            f"NACA {airfoil}",
            "OPER",
            "VISC 1.5e6",
            "MACH 0.0",
            "ITER 100",
            "VPAR",
            mode["vpar_cmds"],
            "",                     # exit VPAR
            f"ALFA {AOA}",
            "CPWR",
            cp_filename,            # plain filename — XFOIL writes to cwd
            "",                     # exit OPER
            "QUIT",
            ""
        ])

        print(f"Running NACA {airfoil} | {mode_name} transition | AoA={AOA}°...")

        proc = subprocess.run(
            [str(XFOIL_EXE)],
            input=commands,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

        if not cp_file.exists():
            print(f"  ✗ Failed — no output file. XFOIL stdout:")
            print(proc.stdout[-1000:])
            continue

        # Parse CPWR file: 3 header lines, then x/c  y/c  Cp
        data = np.loadtxt(cp_file, skiprows=3)
        x_xc = data[:, 0]
        y_xc = data[:, 1]
        Cp   = data[:, 2]

        # Split by y-sign — clean in XFOIL output, no TE ambiguity
        upper_mask = y_xc >= 0
        lower_mask = y_xc <  0

        x_upper = x_xc[upper_mask];  Cp_upper = Cp[upper_mask]
        x_lower = x_xc[lower_mask];  Cp_lower = Cp[lower_mask]

        # Sort LE → TE
        x_upper, Cp_upper = x_upper[np.argsort(x_upper)], Cp_upper[np.argsort(x_upper)]
        x_lower, Cp_lower = x_lower[np.argsort(x_lower)], Cp_lower[np.argsort(x_lower)]

        # Interpolate onto common grid
        x_c = np.linspace(0.01, 0.98, 300)
        Cp_upper_interp = np.interp(x_c, x_upper, Cp_upper)
        Cp_lower_interp = np.interp(x_c, x_lower, Cp_lower)
        dCp = Cp_lower_interp - Cp_upper_interp

        results_dCp_xfoil[airfoil][mode_name] = {
            "x_c":      x_c,
            "dCp":      dCp,
            "x_upper":  x_upper,  "Cp_upper": Cp_upper,
            "x_lower":  x_lower,  "Cp_lower": Cp_lower,
        }

        print(f"  ✓ Saved: {cp_filename}  |  Cl ≈ {np.trapezoid(dCp, x_c):.4f}")
