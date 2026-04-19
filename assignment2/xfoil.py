import subprocess
from pathlib import Path
from xfoil_reader import load_xfoil

import os
XFOIL_EXE = Path(os.environ.get("XFOIL_PATH", r"C:\Users\sebas\Downloads\XFOIL6.99\xfoil.exe"))

BASE_DIR = Path(__file__).parent
# XFOIL_EXE = BASE_DIR / "XFOIL" / "xfoil.exe"
from main import code

airfoil = code

transition_mode = "free"
vpar_cmd = "N 12"  # Natural transition at 12% chord
    


output_filename = f"NACA{airfoil}_{transition_mode}.txt"
output_path = BASE_DIR / output_filename

# Remove existing file so XFOIL doesn't append
if output_path.exists():
    output_path.unlink()

commands = "\n".join([
    f"NACA {airfoil}",
    "OPER",
    "VISC 5e6",
    "MACH 0.0",
    "ITER 100",
    "VPAR",
    vpar_cmd,
    "",
    "PACC",
    output_filename,
    "",
    "ASEQ -4 8 1",
    "PACC",
    "",
    "QUIT",
    ""
])

print(f"Running NACA {airfoil} | {transition_mode} transition...")

print(f"Looking for XFoil at: {XFOIL_EXE}")
print(f"Exists: {XFOIL_EXE.exists()}")


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