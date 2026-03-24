import numpy as np


def solve_wing_glauert(
    planform,
    AR,
    alpha_deg,
    alpha_L0_deg=0.0,
    taper_ratio=1.0,
    N_terms=60,
    N_eval=400,
    m0=2*np.pi
):
    """
    Unified Glauert lifting-line solver for rectangular or tapered wings.

    Parameters
    ----------
    planform : str
        "Rectangular" or "Tapered"
    AR : float
        Aspect ratio
    alpha_deg : float
        Geometric angle of attack [deg]
    alpha_L0_deg : float
        Zero-lift angle [deg]
    taper_ratio : float
        c_tip / c_root, only used if planform == "Tapered"
    N_terms : int
        Number of Fourier terms / collocation points
    N_eval : int
        Number of points for smooth plotting
    m0 : float
        2D lift-curve slope [1/rad]

    Returns
    -------
    dict
        Spanwise distributions and integrated wing coefficients
    """

    planform = planform.strip().lower()

    if planform not in ["rectangular", "tapered"]:
        raise ValueError("planform must be 'Rectangular' or 'Tapered'")

    if planform == "tapered" and taper_ratio <= 0:
        raise ValueError("taper_ratio must be > 0 for tapered wings")

    # Convert angles to radians.
    # rhs_scalar is the effective angle of attack (alpha - alpha_L0),
    # which is the RHS of the Glauert equation at every collocation point.
    # For a uniform wing at constant alpha, this is the same everywhere.
    alpha    = np.deg2rad(alpha_deg)
    alpha_L0 = np.deg2rad(alpha_L0_deg)
    rhs_scalar = alpha - alpha_L0  # [rad]

    def chord_ratio(x_tilde):
        """
        Returns the local chord normalised by the mean chord: c(y) / c_bar.

        x_tilde = 2y/b is the dimensionless spanwise coordinate in [-1, 1].

        Rectangular:  c(y) = c_bar  everywhere  =>  ratio = 1
        Tapered:      c(y) varies linearly from c_root at the root (x_tilde=0)
                      to c_tip at the tips (|x_tilde|=1).
                      Using c_bar = (c_root + c_tip)/2 and lambda = c_tip/c_root:
                          c(y)/c_bar = 2*(1 - (1-lambda)*|x_tilde|) / (1+lambda)
        """
        if planform == "rectangular":
            return np.ones_like(x_tilde)
        elif planform == "tapered":
            TR = taper_ratio
            return 2.0 * (1.0 - (1.0 - TR) * np.abs(x_tilde)) / (1.0 + TR)

    # ------------------------------------------------------------------ #
    #  SPECIAL CASE: 2D (infinite-span) limit                             #
    #  AR -> inf means no tip vortices, no downwash, no induced drag.     #
    #  The full Fourier machinery is not needed; thin airfoil theory       #
    #  gives CL = m0 * (alpha - alpha_L0) directly.                       #
    # ------------------------------------------------------------------ #
    if np.isinf(AR):
        theta_eval   = np.linspace(1e-6, np.pi - 1e-6, N_eval)
        x_tilde      = np.cos(theta_eval)          # spanwise coordinate
        c_ratio_eval = chord_ratio(x_tilde)

        # Sort from left tip (x_tilde=-1) to right tip (x_tilde=+1)
        sort_idx     = np.argsort(x_tilde)
        x_tilde      = x_tilde[sort_idx]
        theta_eval   = theta_eval[sort_idx]
        c_ratio_eval = c_ratio_eval[sort_idx]

        # Thin airfoil theory: CL = m0 * alpha_eff, no 3D correction
        CL_2d = m0 * rhs_scalar

        # No trailing vortices => zero induced angle and zero induced drag
        alpha_i  = np.zeros_like(theta_eval)
        alpha_eff = np.full_like(theta_eval, rhs_scalar)

        # Every section sees the same cl (uniform, 2D)
        cl_local = np.full_like(theta_eval, CL_2d)

        # Dimensionless circulation: cl = 2*Gamma/(U_inf*c)
        # => Gamma/(c_bar*U_inf) = 0.5 * cl * (c/c_bar)
        Gamma_tilde = 0.5 * cl_local * c_ratio_eval

        cdi_local = np.zeros_like(theta_eval)  # no induced drag in 2D

        return {
            "planform": planform, "AR": AR,
            "theta": theta_eval, "x_tilde": x_tilde, "c_ratio": c_ratio_eval,
            "alpha_i_rad": alpha_i, "alpha_i_deg": np.rad2deg(alpha_i),
            "alpha_eff_rad": alpha_eff, "alpha_eff_deg": np.rad2deg(alpha_eff),
            "Gamma_tilde": Gamma_tilde, "cl_local": cl_local,
            "cdi_local": cdi_local, "CL": CL_2d, "CDi": 0.0, "A": None,
        }

    # ------------------------------------------------------------------ #
    #  STEP 1 — Collocation points (Glauert substitution)                 #
    #                                                                      #
    #  The Glauert substitution maps the span onto a half-circle:         #
    #      y = (b/2) * cos(theta),  theta in [0, pi]                     #
    #  theta=0 is the right tip, theta=pi is the left tip.               #
    #                                                                      #
    #  N collocation points are placed at                                 #
    #      theta_j = j*pi/(N+1),  j = 1..N                               #
    #  which keeps them away from the tips where sin(theta)=0 (singular). #
    # ------------------------------------------------------------------ #
    n         = np.arange(1, N_terms + 1)           # Fourier mode indices: 1, 2, ..., N
    theta_col = np.arange(1, N_terms + 1) * np.pi / (N_terms + 1)  # collocation angles
    x_tilde_col = np.cos(theta_col)                  # corresponding spanwise positions

    # Chord ratio at each collocation point
    c_ratio_col = chord_ratio(x_tilde_col)

    # ------------------------------------------------------------------ #
    #  STEP 2 — Chord term in the Glauert equation                        #
    #                                                                      #
    #  The governing LLT equation at each spanwise station is:            #
    #                                                                      #
    #   alpha - alpha_L0 = sum_n A_n sin(n*theta) * [4b/(m0*c) + n/sin(theta)]
    #                                                                      #
    #  The first bracket term comes from the local section lift law        #
    #  (Kutta-Joukowski + thin airfoil): Gamma = 0.5 * m0 * c * U * alpha_eff
    #  Substituting c = c_bar*(c/c_bar) and c_bar = b/AR gives:          #
    #      4b/(m0*c) = 4*AR / (m0 * (c/c_bar))                           #
    # ------------------------------------------------------------------ #
    chord_term_col = 4.0 * AR / (m0 * c_ratio_col)  # shape: (N_terms,)

    # ------------------------------------------------------------------ #
    #  STEP 3 — Assemble and solve the linear system  M * A = rhs         #
    #                                                                      #
    #  Each row j corresponds to one collocation point theta_j.           #
    #  Each column n corresponds to one Fourier coefficient A_n.          #
    #                                                                      #
    #  M[j, n] = sin(n*theta_j) * (chord_term_j + n/sin(theta_j))        #
    #  rhs[j]  = alpha - alpha_L0  (same scalar for all j here)           #
    #                                                                      #
    #  Solving gives the Fourier coefficients A_1, A_2, ..., A_N that     #
    #  represent the spanwise circulation distribution.                    #
    # ------------------------------------------------------------------ #
    M   = np.zeros((N_terms, N_terms))
    rhs = np.full(N_terms, rhs_scalar)

    for j, th in enumerate(theta_col):
        # n * th broadcasts over all mode indices at once
        M[j, :] = np.sin(n * th) * (chord_term_col[j] + n / np.sin(th))

    A = np.linalg.solve(M, rhs)  # shape: (N_terms,)

    # ------------------------------------------------------------------ #
    #  STEP 4 — Dense evaluation grid for smooth spanwise plots           #
    #                                                                      #
    #  Use N_eval points in theta, avoiding exactly 0 and pi              #
    #  (the tips) where sin(theta)=0 causes division issues.              #
    # ------------------------------------------------------------------ #
    theta_eval   = np.linspace(1e-4, np.pi - 1e-4, N_eval)
    x_tilde      = np.cos(theta_eval)        # dimensionless span: -1 (left) to +1 (right)
    c_ratio_eval = chord_ratio(x_tilde)

    # sin_n_theta[n, i] = sin(n * theta_i) — used repeatedly below.
    # np.outer(n, theta_eval) builds the full (N_terms x N_eval) argument matrix.
    sin_n_theta = np.sin(np.outer(n, theta_eval))  # shape: (N_terms, N_eval)

    # Circulation series: Gamma(theta) = 2*b*U_inf * sum_n A_n * sin(n*theta)
    # series_sum is the dimensionless part: sum_n A_n * sin(n*theta)
    series_sum = np.sum(A[:, None] * sin_n_theta, axis=0)  # shape: (N_eval,)

    # ------------------------------------------------------------------ #
    #  STEP 5 — Induced angle of attack                                   #
    #                                                                      #
    #  The downwash from all trailing vortices reduces the effective AoA. #
    #  After applying the Glauert integral identity, the induced angle is: #
    #                                                                      #
    #      alpha_i(theta) = sum_n  n * A_n * sin(n*theta) / sin(theta)   #
    #                                                                      #
    #  Higher harmonics (n > 1) always increase induced angle, which is   #
    #  why the elliptic distribution (A_n=0 for n>1) is optimal.          #
    # ------------------------------------------------------------------ #
    alpha_i = np.sum((n[:, None] * A[:, None]) * sin_n_theta, axis=0) / np.sin(theta_eval)

    # Effective AoA = geometric AoA minus the induced downwash angle
    alpha_eff = rhs_scalar - alpha_i

    # ------------------------------------------------------------------ #
    #  STEP 6 — Dimensionless circulation                                 #
    #                                                                      #
    #  Gamma(theta) = 2*b*U_inf * series_sum                              #
    #  Normalise by c_bar*U_inf  (with c_bar = b/AR):                    #
    #      Gamma_tilde = Gamma / (c_bar * U_inf) = 2*AR * series_sum     #
    # ------------------------------------------------------------------ #
    Gamma_tilde = 2.0 * AR * series_sum

    # ------------------------------------------------------------------ #
    #  STEP 7 — Local section lift coefficient                            #
    #                                                                      #
    #  From Kutta-Joukowski: L' = rho * U_inf * Gamma                    #
    #  And cl = L'/(0.5*rho*U_inf^2*c) = 2*Gamma/(U_inf*c)              #
    #  In terms of Gamma_tilde and c/c_bar:                               #
    #      cl = 2 * Gamma_tilde / (c/c_bar)                              #
    # ------------------------------------------------------------------ #
    cl_local = 2.0 * Gamma_tilde / c_ratio_eval

    # ------------------------------------------------------------------ #
    #  STEP 8 — Local induced drag coefficient                            #
    #                                                                      #
    #  The induced drag is lift tilted backward by the induced angle.     #
    #  For small angles: cdi = cl * alpha_i   (alpha_i in radians)       #
    # ------------------------------------------------------------------ #
    cdi_local = cl_local * alpha_i

    # ------------------------------------------------------------------ #
    #  STEP 9 — Integrated wing coefficients                              #
    #                                                                      #
    #  Integrating the circulation over the span and using Fourier        #
    #  orthogonality on [0, pi] eliminates all cross terms:               #
    #                                                                      #
    #      CL  = pi * AR * A_1                                            #
    #      CDi = pi * AR * sum_n (n * A_n^2)                             #
    #                                                                      #
    #  Only A_1 contributes to lift. All higher harmonics only add drag   #
    #  (the n*A_n^2 terms for n>1 are always positive). This is the       #
    #  mathematical proof that the elliptic wing is aerodynamically ideal. #
    # ------------------------------------------------------------------ #
    CL  = np.pi * AR * A[0]               # A[0] is A_1 (0-indexed)
    CDi = np.pi * AR * np.sum(n * A**2)   # Oswald efficiency implicitly included

    # Sort all arrays from left tip (x_tilde=-1) to right tip (x_tilde=+1)
    # cos(theta) decreases from 1 to -1 as theta goes 0 -> pi, so without
    # sorting the arrays run right-tip to left-tip.
    sort_idx = np.argsort(x_tilde)

    return {
        "planform": planform,
        "AR": AR,
        "theta":        theta_eval[sort_idx],
        "x_tilde":      x_tilde[sort_idx],       # dimensionless span [-1, 1]
        "c_ratio":      c_ratio_eval[sort_idx],   # c(y)/c_bar
        "alpha_i_rad":  alpha_i[sort_idx],
        "alpha_i_deg":  np.rad2deg(alpha_i[sort_idx]),
        "alpha_eff_rad": alpha_eff[sort_idx],
        "alpha_eff_deg": np.rad2deg(alpha_eff[sort_idx]),
        "Gamma_tilde":  Gamma_tilde[sort_idx],    # Gamma / (c_bar * U_inf)
        "cl_local":     cl_local[sort_idx],       # local section cl
        "cdi_local":    cdi_local[sort_idx],      # local section cdi
        "CL":           CL,                       # wing lift coefficient
        "CDi":          CDi,                      # wing induced drag coefficient
        "A":            A,                        # Fourier coefficients A_1..A_N
    }
