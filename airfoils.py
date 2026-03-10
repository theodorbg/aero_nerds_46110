import numpy as np

class NACA4Airfoil:
    def __init__(self, code: str, n_points: int = 200, chord: float = 1.0):
        """
        code: e.g. '2312', '4412', '4424'
        n_points: number of points on upper/lower surfaces (per side uses same x-distribution)
        chord: chord length c (default 1.0)

        Parameters:
        code: 4-digit NACA code as string, e.g. '2312'
        n_points: number of points to generate on each surface (upper/lower)
        chord: chord length (default 1.0)

        """

        # Validate that the NACA code is exactly 4 numeric digits
        if len(code) != 4 or not code.isdigit():
            raise ValueError("Code must be a 4-digit NACA string, e.g. '2312'.")

        self.code = code
        self.c = chord

        # Parse NACA 4-digit code: NACA mpxx
        # 1st digit (m): max camber as % of chord -> divide by 100 to get fraction
        self.m = int(code[0]) / 100.0           # maximum camber (fraction of chord)
        # 2nd digit (p): location of max camber in tenths of chord -> divide by 10
        self.p = int(code[1]) / 10.0            # location of max camber (fraction of chord)
        # 3rd+4th digits (xx): max thickness as % of chord -> divide by 100
        self.t = int(code[2:]) / 100.0          # maximum thickness (fraction of chord)

        # Use cosine spacing to cluster points near the leading edge,
        # where curvature is highest and resolution matters most.
        # beta goes from 0 to pi, producing x values from 0 (LE) to 1 (TE)
        beta = np.linspace(0.0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))       # normalized x in [0, 1]
        self.x = x * self.c                 # scale to dimensional chord length

        # Compute the camber line height (yc) and its slope (dyc_dx)
        # at each x location using the NACA 4-digit analytic formula
        yc, dyc_dx = self._camber_line(x, self.m, self.p)

        # Compute the half-thickness distribution at each x location
        yt = self._thickness(x, self.t)

        # theta is the local camber line angle; used to offset thickness
        # perpendicular to the camber line (not just vertically)
        theta = np.arctan(dyc_dx)

        # Upper surface
        self.xu = self.x - yt * np.sin(theta)
        self.yu = yc * self.c + yt * np.cos(theta)
        

        # Lower surface
        self.xl = self.x + yt * np.sin(theta)
        self.yl = yc * self.c - yt * np.cos(theta)

        # Also store camber line coordinates
        self.yc = yc * self.c

        

        # store slope of camber line
        self.camber_slope = dyc_dx

        # store Cl for various methods (to be computed later)
        self.cl_thin_airfoil = None
        self.cl_panel_method = None
        self.cl_xfoil_free = None
        self.cl_xfoil_fixed = None

        self.cl_dict = {
                    "Thin Airfoil": None,
                    "Panel Method": None,
                    "XFOIL Free":   None,
                    "XFOIL Fixed":  None,
                }

        self.cl_slopes_dict = {
                    "Thin Airfoil": None,
                    "Panel Method": None,
                    "XFOIL Free":   None,
                    "XFOIL Fixed":  None,
                }
        self.cl_offsets_dict = {
                    "Thin Airfoil": None,
                    "Panel Method": None,
                    "XFOIL Free":   None,
                    "XFOIL Fixed":  None,
                }
        
    def _camber_line(self, x: np.ndarray, m: float, p: float):
        """
        Camber line and slope for NACA 4-digit.[file:1]
        x is in [0,1].
        Returns yc (camber height) and dyc_dx (camber slope) as arrays.

        Parameters:
        x: array of x/c locations (normalized chordwise positions)
        m: maximum camber as fraction of chord (e.g. 0.02 for NACA 2312)
        p: location of max camber as fraction of chord (e.g. 0.3 for NACA 2312)

        returns:
        yc: array of camber line heights at each x location
        dyc_dx: array of camber line slopes at each x location

        """
        # Initialize output arrays to zero (symmetric/uncambered airfoil default)
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        # If p == 0, the airfoil is symmetric (no camber), return zeros
        if p == 0:
            return yc, dyc_dx

        # Region 1: forward of max camber location (0 <= x <= p)
        # Parabolic arc rising from LE to peak camber at x = p
        i1 = x <= p
        xc1 = x[i1]
        yc[i1] = (m / p**2) * (2 * p * xc1 - xc1**2)          # camber height
        dyc_dx[i1] = (2 * m / p**2) * (p - xc1)                # positive slope, decreasing to 0 at x=p

        # Region 2: aft of max camber location (p < x <= 1)
        # Parabolic arc falling from peak camber at x = p down to TE
        i2 = x > p
        xc2 = x[i2]
        yc[i2] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*xc2 - xc2**2)   # camber height
        dyc_dx[i2] = (2 * m / (1 - p)**2) * (p - xc2)                  # negative slope toward TE
        
        return yc, dyc_dx

    def _thickness(self, x: np.ndarray, t: float):
        """
        Thickness distribution y_t for NACA 4-digit.[file:1]
        x is in [0,1].

        Parameters:
        x: array of x/c locations (normalized chordwise positions)
        t: maximum thickness as fraction of chord (e.g. 0.12 for NACA 4412)

        Returns:
        y_t: array of thickness distribution values at each x location (half-thickness)
        """
        return 5 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

    def get_coordinates(self):
        """
        Return concatenated surface coordinates suitable for panel method etc.
        Order: TE lower -> LE -> TE upper.

        Parameters:
        None

        Returns:
        x_coords: array of x coordinates for the full airfoil contour
        y_coords: array of y coordinates for the full airfoil contour
        """
        # Ensure correct ordering: lower surface from TE (x≈1) to LE (x≈0),
        # then upper surface from LE (x≈0) to TE (x≈1)
        xl_rev = self.xl[::-1]
        yl_rev = self.yl[::-1]
        xu_fwd = self.xu
        yu_fwd = self.yu

        self.x_coords = np.concatenate([xl_rev, xu_fwd[1:]])  # skip duplicate LE point
        self.y_coords = np.concatenate([yl_rev, yu_fwd[1:]])
        return self.x_coords, self.y_coords

    def get_camber_line(self):
        """
        Return x, y_c for camber line.

        Parameters:
        None

        Returns:
        x: array of x coordinates along the chord (same as self.x)
        y_c: array of camber line heights at each x location
        """
        return self.x, self.yc

# # NACA 2312
# af_2312 = NACA4Airfoil("2312")
# x2312, y2312 = af_2312.get_coordinates()
# xc, yc = af_2312.get_camber_line()

# # NACA 2324
# af_2324 = NACA4Airfoil("2324")

# # NACA 4412 and 4424
# af_4412 = NACA4Airfoil("4412")
# af_4424 = NACA4Airfoil("4424")

