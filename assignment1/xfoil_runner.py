import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
XFOIL_EXE = BASE_DIR / "XFOIL" / "xfoil.exe"

airfoils = ["2312", "2324", "4412", "4424"]

transition_modes = {
    "free": {
        "vpar_cmds": "N 9",
    },
    "fixed": {
        "vpar_cmds": "XTR 0.25 0.25",
    },
}

for airfoil in airfoils:
    for mode_name, mode in transition_modes.items():
        output_filename = f"NACA{airfoil}_{mode_name}.txt"  # just the filename


        commands = "\n".join([
            f"NACA {airfoil}",
            "OPER",
            "VISC 1.5e6",
            "MACH 0.0",
            "ITER 100",
            "VPAR",
            mode["vpar_cmds"],
            "",             # exit VPAR
            "PACC",
            output_filename,   # relative, not str(output_file)
            "",             # no drag polar file
            "ASEQ -10 15 1",
            "PACC",         # toggle PACC off
            "",             # exit OPER
            "QUIT",
            ""
        ])

        print(f"Running NACA {airfoil} | {mode_name} transition...")

        proc = subprocess.run(
            [str(XFOIL_EXE)],
            input=commands,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

        generated = BASE_DIR / output_filename
        if generated.exists():
            print(f"  ✓ Saved: {generated.name}")
        else:
            print(f"  ✗ Failed — no output file created. XFOIL stdout:")
            print(proc.stdout[-1000:])