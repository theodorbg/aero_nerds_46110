import numpy as np

def solve_wing_glauert(
    planform,
    AR,
    alpha_deg=None,
    alpha_L0_deg=0.0,
    taper_ratio=1.0,
    twist_type=None,
    alpha_root_deg=None,
    alpha_tip_deg=None,
    N_terms=60,
    N_eval=400,
    m0=2*np.pi
):
    """
    Unified Glauert lifting-line solver for:
    - rectangular untwisted wings
    - tapered untwisted wings
    - rectangular/tapered wings with spanwise-varying geometric AoA

    Parameters
    ----------
    planform : str
        "Rectangular" or "Tapered"
    AR : float
        Aspect ratio
    alpha_deg : float, optional
        Constant geometric angle of attack [deg] for untwisted wings
    alpha_L0_deg : float
        Zero-lift angle [deg]
    taper_ratio : float
        c_tip / c_root, only used if planform == "Tapered"
    twist_type : str or None
        None or "linear"
    alpha_root_deg : float or None
        Root/midpoint geometric angle [deg] for twisted case
    alpha_tip_deg : float or None
        Tip geometric angle [deg] for twisted case
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

    if twist_type is None:
        if alpha_deg is None:
            raise ValueError("For untwisted wings, alpha_deg must be provided.")
    elif twist_type.lower() == "linear":
        if alpha_root_deg is None or alpha_tip_deg is None:
            raise ValueError("For linear twist, alpha_root_deg and alpha_tip_deg must be provided.")
    else:
        raise ValueError("twist_type must be None or 'linear'")

    alpha_L0 = np.deg2rad(alpha_L0_deg)

    def chord_ratio(x_tilde):
        """
        Returns c(x_tilde) / c_bar
        """
        if planform == "rectangular":
            return np.ones_like(x_tilde)
        elif planform == "tapered":
            TR = taper_ratio
            return 2.0 * (1.0 - (1.0 - TR) * np.abs(x_tilde)) / (1.0 + TR)

    def alpha_geom_deg_distribution(x_tilde):
        """
        Returns local geometric angle of attack in degrees.
        """
        if twist_type is None:
            return np.full_like(x_tilde, alpha_deg, dtype=float)
        elif twist_type.lower() == "linear":
            return alpha_root_deg + (alpha_tip_deg - alpha_root_deg) * np.abs(x_tilde)

    # 2D infinite-AR limit
    if np.isinf(AR):
        theta_eval = np.linspace(1e-6, np.pi - 1e-6, N_eval)
        x_tilde = np.cos(theta_eval)
        c_ratio_eval = chord_ratio(x_tilde)

        alpha_geom_deg_eval = alpha_geom_deg_distribution(x_tilde)
        alpha_geom_eval = np.deg2rad(alpha_geom_deg_eval)

        sort_idx = np.argsort(x_tilde)
        x_tilde = x_tilde[sort_idx]
        theta_eval = theta_eval[sort_idx]
        c_ratio_eval = c_ratio_eval[sort_idx]
        alpha_geom_deg_eval = alpha_geom_deg_eval[sort_idx]
        alpha_geom_eval = alpha_geom_eval[sort_idx]

        # In 2D there is no induced angle or induced drag
        alpha_i = np.zeros_like(theta_eval)
        alpha_eff = alpha_geom_eval - alpha_L0

        # Local section lift coefficient
        cl_local = m0 * alpha_eff

        # Gamma_tilde = Gamma / (c_bar U_inf)
        Gamma_tilde = 0.5 * cl_local * c_ratio_eval

        cdi_local = np.zeros_like(theta_eval)

        # Representative integrated CL for the untwisted infinite-wing case
        # For twisted infinite-wing case this is just spanwise mean of local cl
        CL_2d = np.mean(cl_local)

        return {
            "planform": planform,
            "AR": AR,
            "theta": theta_eval,
            "x_tilde": x_tilde,
            "c_ratio": c_ratio_eval,
            "alpha_geom_deg": alpha_geom_deg_eval,
            "alpha_geom_rad": alpha_geom_eval,
            "alpha_i_rad": alpha_i,
            "alpha_i_deg": np.rad2deg(alpha_i),
            "alpha_eff_rad": alpha_eff,
            "alpha_eff_deg": np.rad2deg(alpha_eff),
            "Gamma_tilde": Gamma_tilde,
            "cl_local": cl_local,
            "cdi_local": cdi_local,
            "CL": CL_2d,
            "CDi": 0.0,
            "A": None,
        }

    # Collocation points
    n = np.arange(1, N_terms + 1)
    theta_col = np.arange(1, N_terms + 1) * np.pi / (N_terms + 1)
    x_tilde_col = np.cos(theta_col)

    # Local chord and local geometric AoA
    c_ratio_col = chord_ratio(x_tilde_col)
    alpha_geom_deg_col = alpha_geom_deg_distribution(x_tilde_col)
    alpha_geom_col = np.deg2rad(alpha_geom_deg_col)

    # Local chord term in Glauert system
    chord_term_col = 4.0 * AR / (m0 * c_ratio_col)

    # Build linear system M A = rhs
    M = np.zeros((N_terms, N_terms))
    rhs = alpha_geom_col - alpha_L0

    for j, th in enumerate(theta_col):
        M[j, :] = np.sin(n * th) * (chord_term_col[j] + n / np.sin(th))

    A = np.linalg.solve(M, rhs)

    # Dense evaluation grid
    theta_eval = np.linspace(1e-4, np.pi - 1e-4, N_eval)
    x_tilde = np.cos(theta_eval)
    c_ratio_eval = chord_ratio(x_tilde)

    alpha_geom_deg_eval = alpha_geom_deg_distribution(x_tilde)
    alpha_geom_eval = np.deg2rad(alpha_geom_deg_eval)

    sin_n_theta = np.sin(np.outer(n, theta_eval))
    series_sum = np.sum(A[:, None] * sin_n_theta, axis=0)

    # Induced angle
    alpha_i = np.sum((n[:, None] * A[:, None]) * sin_n_theta, axis=0) / np.sin(theta_eval)

    # Effective angle
    alpha_eff = alpha_geom_eval - alpha_L0 - alpha_i

    # Dimensionless circulation using mean chord
    Gamma_tilde = 2.0 * AR * series_sum

    # Local section lift coefficient
    cl_local = 2.0 * Gamma_tilde / c_ratio_eval

    # Local induced drag coefficient approximation
    cdi_local = cl_local * alpha_i

    # Wing coefficients
    CL = np.pi * AR * A[0]
    CDi = np.pi * AR * np.sum(n * A**2)

    # Sort left tip -> right tip
    sort_idx = np.argsort(x_tilde)

    return {
        "planform": planform,
        "AR": AR,
        "theta": theta_eval[sort_idx],
        "x_tilde": x_tilde[sort_idx],
        "c_ratio": c_ratio_eval[sort_idx],
        "alpha_geom_deg": alpha_geom_deg_eval[sort_idx],
        "alpha_geom_rad": alpha_geom_eval[sort_idx],
        "alpha_i_rad": alpha_i[sort_idx],
        "alpha_i_deg": np.rad2deg(alpha_i[sort_idx]),
        "alpha_eff_rad": alpha_eff[sort_idx],
        "alpha_eff_deg": np.rad2deg(alpha_eff[sort_idx]),
        "Gamma_tilde": Gamma_tilde[sort_idx],
        "cl_local": cl_local[sort_idx],
        "cdi_local": cdi_local[sort_idx],
        "CL": CL,
        "CDi": CDi,
        "A": A,
    }