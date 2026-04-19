import re
import numpy as np
from airfoils import NACA4Airfoil
from plot import *


class XFoil:
    def __init__(self, filepath):
        self.filepath = filepath
        self.airfoil_name = None
        self.mach = None
        self.reynolds = None
        self.ncrit = None
        self.xtrf_top = None
        self.xtrf_bot = None
        self.alpha = []
        self.CL = []
        self.CD = []
        self.CDp = []
        self.CM = []
        self.Top_Xtr = []
        self.Bot_Xtr = []
        self._parse(filepath)

    def _parse(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # Airfoil name
            if 'Calculated polar for:' in line:
                self.airfoil_name = line.split('Calculated polar for:')[1].strip()

            # Mach, Re, Ncrit
            elif 'Mach =' in line and 'Re =' in line:
                mach_match = re.search(r'Mach\s*=\s*([\d.]+)', line)
                re_match = re.search(r'Re\s*=\s*([\d.]+)\s*e\s*([\d]+)', line)
                ncrit_match = re.search(r'Ncrit\s*=\s*([\d.]+)', line)
                if mach_match:
                    self.mach = float(mach_match.group(1))
                if re_match:
                    self.reynolds = float(re_match.group(1)) * 10**int(re_match.group(2))
                if ncrit_match:
                    self.ncrit = float(ncrit_match.group(1))

            # xtrf top and bottom
            elif 'xtrf =' in line:
                xtrf_match = re.search(r'xtrf\s*=\s*([\d.]+)\s*\(top\)\s*([\d.]+)\s*\(bottom\)', line)
                if xtrf_match:
                    self.xtrf_top = float(xtrf_match.group(1))
                    self.xtrf_bot = float(xtrf_match.group(2))

            # Data rows
            else:
                stripped = line.strip()
                if stripped and stripped[0] in '0123456789-':
                    # Skip separator lines like '------ -------- ...'
                    if re.match(r'^[-\s]+$', stripped):
                        continue
                    values = stripped.split()
                    if len(values) == 7:
                        self.alpha.append(float(values[0]))
                        self.CL.append(float(values[1]))
                        self.CD.append(float(values[2]))
                        self.CDp.append(float(values[3]))
                        self.CM.append(float(values[4]))
                        self.Top_Xtr.append(float(values[5]))
                        self.Bot_Xtr.append(float(values[6]))

        # Convert to numpy arrays
        self.alpha   = np.array(self.alpha)
        self.CL      = np.array(self.CL)
        self.CD      = np.array(self.CD)
        self.CDp     = np.array(self.CDp)
        self.CM      = np.array(self.CM)
        self.Top_Xtr = np.array(self.Top_Xtr)
        self.Bot_Xtr = np.array(self.Bot_Xtr)

    def __repr__(self):
        return (f"XFoil(airfoil='{self.airfoil_name}', "
                f"Mach={self.mach}, Re={self.reynolds:.2e}, "
                f"Ncrit={self.ncrit}, points={len(self.alpha)})")

# create xfoil objects and load to dictionary

from pathlib import Path

def load_xfoils(path_free, path_fixed):
    base = Path(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL")

    xfoils_free = {
        "2312": XFoil(str(base / path_free / "NACA2312_free.txt")),
        "2324": XFoil(str(base / path_free / "NACA2324_free.txt")),
        "4412": XFoil(str(base / path_free / "NACA4412_free.txt")),
        "4424": XFoil(str(base / path_free / "NACA4424_free.txt")),
    }

    xfoils_fixed = {
        "2312": XFoil(str(base / path_fixed / "NACA2312_fixed.txt")),
        "2324": XFoil(str(base / path_fixed / "NACA2324_fixed.txt")),
        "4412": XFoil(str(base / path_fixed / "NACA4412_fixed.txt")),
        "4424": XFoil(str(base / path_fixed / "NACA4424_fixed.txt")),
    }

    return xfoils_free, xfoils_fixed