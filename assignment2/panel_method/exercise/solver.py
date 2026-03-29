import numpy as np

from funaerotool.panel_method.freestream import freestream_components
from funaerotool.panel_method.induction_matrix import (
    global_panel_induced_velocity_matrices,)
from funaerotool.panel_method.postprocessing import compute_pressure_coefficient
from funaerotool.panel_method.preprocessing import panel_geometry
from funaerotool.panel_method.transformations import global_to_local
from funaerotool.panel_method.vortex import parabolic_vortex_distribution

import numpy as np
import matplotlib.pyplot as plt

def compute_dCp_panel_method(airfoil, aoa_deg=10.0, U_inf=1.0):
    """
    Compute pressure difference distribution ΔCp = Cp_lower - Cp_upper
    as a function of x/c using the panel method.

    Parameters
    ----------
    airfoil : NACA4Airfoil
    aoa_deg : float
        Angle of attack in degrees
    U_inf : float
        Freestream velocity

    Returns
    -------
    xc : ndarray
        chordwise locations (x/c)
    dCp : ndarray
        pressure difference distribution
    """

    # --- 1 get closed contour ---
    x, y = airfoil.get_closed_contour()

    # --- 2 run panel solver ---
    sol = solve_closed_contour_panel_method(x, y, aoa_deg=aoa_deg, U_inf=U_inf)

    Cp = sol["Cp"]
    xp = sol["xp"]
    yp = sol["yp"]

    # normalize chord location
    chord = np.max(x) - np.min(x)
    xc = (xp - np.min(x)) / chord

    # --- 3 split upper and lower surfaces ---
    upper_mask = yp > 0
    lower_mask = yp < 0

    xc_upper = xc[upper_mask]
    Cp_upper = Cp[upper_mask]

    xc_lower = xc[lower_mask]
    Cp_lower = Cp[lower_mask]

    # sort so interpolation works
    i_u = np.argsort(xc_upper)
    i_l = np.argsort(xc_lower)

    xc_upper = xc_upper[i_u]
    Cp_upper = Cp_upper[i_u]

    xc_lower = xc_lower[i_l]
    Cp_lower = Cp_lower[i_l]

    # --- 4 common x grid ---
    xc_common = np.linspace(0, 1, 200)

    Cp_upper_i = np.interp(xc_common, xc_upper, Cp_upper)
    Cp_lower_i = np.interp(xc_common, xc_lower, Cp_lower)

    # --- 5 pressure difference ---
    dCp = Cp_lower_i - Cp_upper_i

    return xc_common, dCp

def compute_dCp_panel(
    airfoil,
    aoa_deg: float = 10.0,
    U_inf: float = 1.0,
    n_interp: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute ΔCp = Cp_lower - Cp_upper as a function of x/c using the panel method.

    Args:
        airfoil:   NACA4Airfoil instance (provides get_closed_contour).
        aoa_deg:   Angle of attack in degrees.
        U_inf:     Freestream speed (cancels in Cp; kept for API consistency).
        n_interp:  Number of x/c points in [0, 1] for output.

    Returns:
        x_c:    x/c array from 0 to 1.
        dCp:    ΔCp = Cp_lower − Cp_upper at each x/c (positive = net lifting force).
        result: Raw dict returned by solve_closed_contour_panel_method.
    """
    x, y = airfoil.get_closed_contour()

    result = solve_closed_contour_panel_method(
        x, y, aoa_deg=aoa_deg, U_inf=U_inf, kutta_condition=True
    )

    xp = result["xp"]          # panel midpoint x  (N panels)
    yp = result["yp"]          # panel midpoint y
    Cp = result["Cp"]          # Cp at each panel midpoint
    n_panels = len(xp)

    # ── Index-based surface split ─────────────────────────────────────────────
    # Contour order: TE → lower (TE→LE) → upper (LE→TE) → TE
    # n_panels = 2*(n_points - 1), so each surface occupies exactly half
    half = n_panels // 2

    # Lower surface panels: indices 0 … half-1  (TE → LE direction)
    xp_lower = xp[:half]
    Cp_lower = Cp[:half]

    # Upper surface panels: indices half … n_panels-1  (LE → TE direction)
    xp_upper = xp[half:]
    Cp_upper = Cp[half:]

    # ── Sort each surface LE → TE (x ascending) ──────────────────────────────
    idx_l = np.argsort(xp_lower)
    idx_u = np.argsort(xp_upper)
    xp_lower, Cp_lower = xp_lower[idx_l], Cp_lower[idx_l]
    xp_upper, Cp_upper = xp_upper[idx_u], Cp_upper[idx_u]

    # ── Normalise by chord ────────────────────────────────────────────────────
    chord = airfoil.c
    xp_lower /= chord
    xp_upper /= chord

    # ── Interpolate onto uniform x/c grid ────────────────────────────────────
    x_c = np.linspace(0.0, 1.0, n_interp)          # force full 0→1 range
    Cp_lower_i = np.interp(x_c, xp_lower, Cp_lower)
    Cp_upper_i = np.interp(x_c, xp_upper, Cp_upper)

    # ΔCp sign convention matches your assignment: (p_upper - p_lower) / q_inf
    # which equals Cp_upper - Cp_lower, negative for a lifting airfoil.
    # Flip sign if your assignment uses (p_lower - p_upper):
    dCp = Cp_lower_i - Cp_upper_i   # positive → net upward force

    # Store on the airfoil object for later comparison / plotting
    airfoil.dCp_panel_method = dCp
    airfoil.dCp_xc_pm = x_c
    airfoil.cl_panel_method = result["Cl"]

    return x_c, dCp

def solve_closed_contour_panel_method(
    x: np.ndarray,
    y: np.ndarray,
    aoa_deg: float = 0.0,
    U_inf: float = 1.0,
    kutta_condition: bool = True,
) -> dict:
    """Solve a closed-contour panel system with optional Kutta circulation.

    Parameters
    ----------
    x, y : np.ndarray
        Closed contour coordinates (first point == last point).
    aoa_deg : float
        Angle of attack in degrees.
    U_inf : float
        Freestream speed magnitude.
    kutta_condition : bool
        When True, solve a coupled system with a parabolic vortex distribution
        to enforce a Kutta condition.
    """
    # Flatten inputs to 1D arrays so geometry math stays vectorized
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Panel geometry: midpoints (xp, yp), tangents (Tx, Ty), normals (Nx, Ny), lengths
    plength, xp, yp, Tx, Ty, Nx, Ny = panel_geometry(x, y)
    n_panels = plength.size

    # Influence matrices for source panels evaluated at panel control points
    # A_g_source[i,j] = x-wise velocity at panel i due to panel j
    # B_g_source[i,j] = y-wise velocity at panel i due to panel j
    A_g_source, B_g_source = global_panel_induced_velocity_matrices(
        x_eval=xp,
        y_eval=yp,
        xp=xp,
        yp=yp,
        Tx=Tx,
        Ty=Ty,
        Nx=Nx,
        Ny=Ny,
        panel_lengths=plength,
        panel_type="source",
    )
    # None, None  # TODO: Ex1 - Replace `None` and add function here
    
    # A_g_source = 

    # Transform induced velocities into each panel's local frame
    # A_l_source[i,j] = normal velocity at panel i due to panel j
    # B_l_source[i,j] = tangential velocity at panel i due to panel j
    B_l_source, A_l_source = global_to_local(
        ux=A_g_source,
        uy=B_g_source,
        tx=Tx[:,None],
        ty=Ty[:,None],
        nx=Nx[:,None],
        ny=Ny[:,None],
    )
    #None, None  # TODO: Ex1 - Replace `None` and add function here

    # Freestream components in global coordinates
    u_inf_x, u_inf_y = freestream_components(aoa_deg=aoa_deg, U_inf=U_inf)

    # Freestream components in each panel's local frame
    u_inf_t, u_inf_n = global_to_local(
                                        ux=u_inf_x,
                                        uy=u_inf_y,
                                        tx=Tx,
                                        ty=Ty,
                                        nx=Nx,
                                        ny=Ny
                                        )
                                             #None, None  # TODO: Ex1 - Replace `None` and add function here

    if kutta_condition is False:
        # Source-only solve: A * σ = F  →  σ = A⁻¹ F
        sigma = np.linalg.solve(A_l_source, -u_inf_n)  # TODO: Ex1 - Replace `None` and add function here
        circulation = 0.0
        gamma = None
    else:
        # Parabolic vortex distribution used to satisfy the Kutta condition
        gamma_hat = parabolic_vortex_distribution(plength)

        # Influence matrices for vortex panels evaluated at the same control points
        A_g_vortex, B_g_vortex = global_panel_induced_velocity_matrices(
            x_eval=xp,
            y_eval=yp,
            xp=xp,
            yp=yp,
            Tx=Tx,
            Ty=Ty,
            Nx=Nx,
            Ny=Ny,
            panel_lengths=plength,
            panel_type="vortex",
        )

        # Transform vortex-induced velocities into local panel frames
        B_l_vortex, A_l_vortex = global_to_local(
            ux=A_g_vortex,
            uy=B_g_vortex,
            tx=Tx[:,None],
            ty=Ty[:,None],
            nx=Nx[:,None],
            ny=Ny[:,None])

        # Assemble coupled source-vortex linear system enforcing Kutta condition
        lhs = np.zeros((n_panels + 1, n_panels + 1), dtype=float)
        lhs[:n_panels, :n_panels] = A_l_source
        lhs[:n_panels, -1] = A_l_vortex @ gamma_hat
        lhs[n_panels, :n_panels] = B_l_source[0, :] + B_l_source[-1, :]  # Tangential velocity at TE control point from all sources
        lhs[n_panels, n_panels] = (B_l_vortex[0, :] + B_l_vortex[-1, :]) @ gamma_hat  # Tangential velocity at TE control point from vortex distribution
        # TODO: Ex2 - LHS content here

        # Right-hand side: no-penetration plus Kutta tangency constraint
        rhs = np.zeros(n_panels + 1, dtype=float)
        rhs[:n_panels] = -u_inf_n  # No-penetration constraint
        rhs[-1] = -(u_inf_t[0] + u_inf_t[-1])  # Kutta tangency constraint at TE control point

        # Solve for source strengths and circulation, then recover distributed gamma
        solution = np.linalg.solve(lhs, rhs)
        sigma = solution[:n_panels]
        circulation = solution[-1]
        gamma = circulation * gamma_hat

    # Total tangential and normal velocities including optional circulation
    Vt = B_l_source @ sigma + u_inf_t
    Vn = A_l_source @ sigma + u_inf_n
    if gamma is not None:
        Vt = Vt + B_l_vortex @ gamma
        Vn = Vn + A_l_vortex @ gamma

    # Pressure coefficient from local velocities
    Cp = compute_pressure_coefficient(ux=Vt, uy=Vn, U_inf=U_inf)

    # Lift coefficient from Kutta circulation; chord inferred from x-span
    chord = np.max(x) - np.min(x)
    Cl = -2.0 * circulation / (chord * U_inf)

    # Bundle outputs for downstream analysis and plotting
    return {
        "sigma": sigma,
        "strengths": sigma,
        "circulation": circulation,
        "gamma": gamma,
        "Vt": Vt,
        "Vn": Vn,
        "Cp": Cp,
        "Cl": Cl,
        "xp": xp,
        "yp": yp,
        "panel_lengths": plength,
        "Tx": Tx,
        "Ty": Ty,
        "Nx": Nx,
        "Ny": Ny,
    }

def get_dCp(xp, yp, Cp, n_interp=100):
    """
    Return x/c and ΔCp = Cp_lower - Cp_upper on a common chordwise grid.
    """
    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    Cp = np.asarray(Cp, dtype=float)

    # Split by index (robust for closed contour ordering from this solver)
    n = len(xp)
    half = n // 2

    x_lower, Cp_lower = xp[:half], Cp[:half]
    x_upper, Cp_upper = xp[half:], Cp[half:]

    # Sort LE->TE
    il = np.argsort(x_lower)
    iu = np.argsort(x_upper)
    x_lower, Cp_lower = x_lower[il], Cp_lower[il]
    x_upper, Cp_upper = x_upper[iu], Cp_upper[iu]

    # Normalize x to x/c using global chord from panel midpoints
    x_min = np.min(xp)
    chord = np.max(xp) - x_min
    x_lower = (x_lower - x_min) / chord
    x_upper = (x_upper - x_min) / chord

    # Interpolate both surfaces to same x/c grid
    x_c = np.linspace(0.0, 1.0, n_interp)
    Cp_l_i = np.interp(x_c, x_lower, Cp_lower)
    Cp_u_i = np.interp(x_c, x_upper, Cp_upper)

    dCp = (Cp_u_i - Cp_l_i)

    return x_c, dCp
