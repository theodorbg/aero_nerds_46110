import numpy as np
import matplotlib.pyplot as plt
from xfoil_reader import load_xfoil

TITLE_FS = 30
LABEL_FS = 20
TICK_FS = 20
LEGEND_FS = 15
LEGEND_TITLE_FS = 10


class EllipticWing:
    def __init__(self, AR, airfoil, alpha_L0_deg=0.0):
        self.AR = AR
        self.airfoil = airfoil
        self.alpha_deg = airfoil.alpha
        self.alpha_rad = np.deg2rad(self.alpha_deg)
        self.alpha_L0_rad = np.deg2rad(alpha_L0_deg)
        self.label = "inf" if AR == np.inf else f"{AR}"

        self._compute()

    def _compute(self):
        ar = self.AR
        a = self.alpha_rad
        alpha_L0 = self.alpha_L0_rad

        if ar != np.inf:
            self.CL = (2 * np.pi) / (1 + 2 / ar) * (a - alpha_L0)
            self.CDi = self.CL**2 / (np.pi * ar)
            alpha_i = self.CL / (np.pi * ar)
            self.alpha_eff = a - alpha_i
        else:
            self.CL = 2 * np.pi * (a - alpha_L0)
            self.CDi = np.zeros_like(a)
            self.alpha_eff = a.copy()

        # Profile drag: interpolate from 2D polar at the EFFECTIVE alpha
        # (the angle the local section actually "sees")
        cl_2d = np.interp(np.rad2deg(self.alpha_eff), self.airfoil.alpha, self.airfoil.CL)
        self.CD0 = np.interp(cl_2d, self.airfoil.CL, self.airfoil.CD)

        # Total drag
        self.CD = self.CD0 + self.CDi


# ── Setup ──────────────────────────────────────────────────────────────────────
Re = 5e6
AR_list = [4, 6, 8, 10, np.inf]
alpha_L0_deg = 0.0

airfoil = load_xfoil("2410")

# interpolate to find the zero lift angle
alpha_L0_deg = np.interp(0.0, airfoil.CL, airfoil.alpha)

print(f"Zero lift angle of attack: {alpha_L0_deg:.2f} deg")

wings = [EllipticWing(ar, airfoil, alpha_L0_deg) for ar in AR_list]


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200, constrained_layout=True, sharex=True)

for wing in wings:
    label = f"AR={wing.label}"
    axes[0].plot(wing.alpha_deg, wing.CL,  marker='o', label=label)
    axes[1].plot(wing.alpha_deg, wing.CDi, marker='o', label=label)
    axes[2].plot(wing.alpha_deg, wing.CD,  marker='o', label=label)

axes[0].set_ylabel(r'$C_L$ [-]', fontsize=LABEL_FS)
axes[0].set_title(r'$C_L$ vs $\alpha$', fontsize=TITLE_FS)

axes[1].set_ylabel(r'$C_{D,i}$ [-]', fontsize=LABEL_FS)
axes[1].set_title(r'$C_{D,i}$ vs $\alpha$', fontsize=TITLE_FS)

axes[2].set_xlabel(r'$\alpha$ [deg]', fontsize=LABEL_FS)
axes[2].set_ylabel(r'$C_D$ [-]', fontsize=LABEL_FS)
axes[2].set_title(r'$C_D$ vs $\alpha$', fontsize=TITLE_FS)

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=8))
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, title="Aspect Ratio", title_fontsize=LEGEND_TITLE_FS)

plt.savefig("Q1.png", dpi=300)
plt.close()
