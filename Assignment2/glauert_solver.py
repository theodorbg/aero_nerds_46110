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

    alpha = np.deg2rad(alpha_deg)
    alpha_L0 = np.deg2rad(alpha_L0_deg)
    rhs_scalar = alpha - alpha_L0

    def chord_ratio(x_tilde):
        """
        Returns c(x_tilde) / c_bar
        """
        if planform == "rectangular":
            return np.ones_like(x_tilde)
        elif planform == "tapered":
            TR = taper_ratio
            return 2.0 * (1.0 - (1.0 - TR) * np.abs(x_tilde)) / (1.0 + TR)

    # 2D infinite-AR limit
    if np.isinf(AR):
        theta_eval = np.linspace(1e-6, np.pi - 1e-6, N_eval)
        x_tilde = np.cos(theta_eval)
        c_ratio_eval = chord_ratio(x_tilde)

        sort_idx = np.argsort(x_tilde)
        x_tilde = x_tilde[sort_idx]
        theta_eval = theta_eval[sort_idx]
        c_ratio_eval = c_ratio_eval[sort_idx]

        CL_2d = m0 * rhs_scalar

        # In 2D there is no induced angle or induced drag
        alpha_i = np.zeros_like(theta_eval)
        alpha_eff = np.full_like(theta_eval, rhs_scalar)

        # For consistency, define local cl from 2D section law
        cl_local = np.full_like(theta_eval, CL_2d)

        # Gamma_tilde = Gamma / (c_bar U_inf)
        # Since cl = 2*Gamma/(U_inf*c), then Gamma/(c_bar U_inf) = 0.5 * cl * c/c_bar
        Gamma_tilde = 0.5 * cl_local * c_ratio_eval

        cdi_local = np.zeros_like(theta_eval)

        return {
            "planform": planform,
            "AR": AR,
            "theta": theta_eval,
            "x_tilde": x_tilde,
            "c_ratio": c_ratio_eval,
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

    # Local nondimensional chord c/c_bar at collocation points
    c_ratio_col = chord_ratio(x_tilde_col)

    # Local chord term in the Glauert system
    # c = c_bar * (c/c_bar), and c_bar = b/AR  =>  4b/(m0*c) = 4AR/(m0*(c/c_bar))
    chord_term_col = 4.0 * AR / (m0 * c_ratio_col)

    # Build linear system M A = rhs
    M = np.zeros((N_terms, N_terms))
    rhs = np.full(N_terms, rhs_scalar)

    for j, th in enumerate(theta_col):
        M[j, :] = np.sin(n * th) * (chord_term_col[j] + n / np.sin(th))

    A = np.linalg.solve(M, rhs)

    # Dense evaluation grid
    theta_eval = np.linspace(1e-4, np.pi - 1e-4, N_eval)
    x_tilde = np.cos(theta_eval)
    c_ratio_eval = chord_ratio(x_tilde)

    sin_n_theta = np.sin(np.outer(n, theta_eval))
    series_sum = np.sum(A[:, None] * sin_n_theta, axis=0)

    # Induced angle
    alpha_i = np.sum((n[:, None] * A[:, None]) * sin_n_theta, axis=0) / np.sin(theta_eval)

    # Effective angle
    alpha_eff = rhs_scalar - alpha_i

    # Dimensionless circulation using mean chord:
    # Gamma_tilde = Gamma / (c_bar U_inf)
    # Gamma = 2 b U_inf sum(A_n sin(n theta)), c_bar = b/AR
    # => Gamma_tilde = 2 AR sum(...)
    Gamma_tilde = 2.0 * AR * series_sum

    # Local section lift coefficient
    # cl = 2 Gamma / (U_inf c) = 2 * Gamma_tilde / (c/c_bar)
    cl_local = 2.0 * Gamma_tilde / c_ratio_eval

    # Local induced drag coefficient approximation
    cdi_local = cl_local * alpha_i   # alpha_i in radians

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