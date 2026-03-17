import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
XFOIL_EXE = BASE_DIR / "XFOIL" / "xfoil.exe"

airfoils = ["2312", "2324", "4412", "4424"]
AOA = 10.0

transition_modes = {
    "free":  {"vpar_cmds": "N 9"},
    "fixed": {"vpar_cmds": "XTR 0.1 0.1"},
}

for airfoil in airfoils:
    for mode_name, mode in transition_modes.items():
        cp_filename = f"cp_NACA{airfoil}_{mode_name}_{AOA:.1f}deg.txt"
        cp_file = BASE_DIR / cp_filename    

        # Skip if already generated
        if cp_file.exists():
            print(f"  — Skipping {cp_filename} (already exists)")
            continue

        commands = "\n".join([
            f"NACA {airfoil}",
            "OPER",
            "VISC 1.5e6",
            "MACH 0.0",
            "ITER 100",
            "VPAR",
            mode["vpar_cmds"],
            "",
            f"ALFA {AOA}",
            "CPWR",
            cp_filename,
            "",
            "QUIT",
            ""
        ])

        print(f"Running NACA {airfoil} | {mode_name} | AoA={AOA}°...")

        proc = subprocess.run(
            [str(XFOIL_EXE)],
            input=commands,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

        if cp_file.exists():
            print(f"  ✓ Saved: {cp_filename}")
        else:
            print(f"  ✗ Failed. XFOIL stdout:")
            print(proc.stdout[-1000:])
