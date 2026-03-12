import re
import numpy as np

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

xfoil2312_free = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\free_trans\NACA2312_free.txt")
xfoil2324_free = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\free_trans\NACA2324_free.txt")
xfoil4412_free = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\free_trans\NACA4412_free.txt")
xfoil4424_free = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\free_trans\NACA4424_free.txt")

xfoil2312_fixed = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\fixed_trans\NACA2312_fixed.txt")
xfoil2324_fixed = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\fixed_trans\NACA2324_fixed.txt")
xfoil4412_fixed = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\fixed_trans\NACA4412_fixed.txt")
xfoil4424_fixed = XFoil(r"c:\Users\tgilh\git_master_laptop\aero_nerds_46110\XFOIL\fixed_trans\NACA4424_fixed.txt")

xfoils_free = {
    "2312": xfoil2312_free,
    "2324": xfoil2324_free,
    "4412": xfoil4412_free,
    "4424": xfoil4424_free
}


xfoils_fixed = {
    "2312": xfoil2312_fixed,
    "2324": xfoil2324_fixed,
    "4412": xfoil4412_fixed,
    "4424": xfoil4424_fixed
}

# print the shape of the data arrays for all xfoils to verify they were loaded correctly
# for code, xfoil in xfoils_fixed.items():
#     print(f"{code} Fixed: alpha={xfoil.alpha.shape}, CL={xfoil.CL.shape}, CD={xfoil.CD.shape}, CDp={xfoil.CDp.shape}, CM={xfoil.CM.shape}, Top_Xtr={xfoil.Top_Xtr.shape}, Bot_Xtr={xfoil.Bot_Xtr.shape}")