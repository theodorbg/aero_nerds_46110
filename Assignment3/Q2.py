import numpy as np
from funcs import rotor_power

# ── Ingenuity reference parameters ──────────────────────────────────────────
m_ingenuity             = 1.8        # kg
m_ingenuity_no_fuselage = 1.8 - 0.3 # kg
R_ingenuity             = 0.6        # m
P_ingenuity             = 88.64      # W

# ── Fixed drone parameters ───────────────────────────────────────────────────
N_blade  = 3
N_rotor  = 2
R        = 1.0   # m
c_mean   = 0.05  # m  ← set your mean chord
rpm      = 2500  # ← set your operating RPM

m_blade         = 70/4 * N_blade * R / R_ingenuity  # kg
m_payload       = 2.0   # kg
m_battery_pack  = 0.5   # kg
m_components    = 1.0   # kg
m_propellers    = N_blade * m_blade * N_rotor        # kg

# ── Iterative solver ─────────────────────────────────────────────────────────
P_drone   = 200.0   # initial guess [W] — pick something reasonable
tol       = 1e-4    # convergence threshold [W]
max_iter  = 100
alpha     = 0.5     # relaxation factor (0 < alpha ≤ 1), helps stability

for i in range(max_iter):

    # 1. Estimate mass components that depend on P_drone
    m_motor_per_rotor = (0.25 / 2) * (P_drone / P_ingenuity)   # kg
    m_motor           = m_motor_per_rotor * N_rotor             # kg

    # 2. Build up total mass
    m_no_fuselage = m_payload + m_battery_pack + m_components + m_propellers + m_motor
    m_fuselage    = m_no_fuselage * (m_ingenuity / m_ingenuity_no_fuselage)
    m_total       = m_fuselage + m_no_fuselage

    # 3. Compute required power for this mass
    P_new = rotor_power(m_total, R, c_mean, rpm, N_blade, N_rotor)

    # 4. Check convergence
    if abs(P_new - P_drone) < tol:
        print(f"Converged in {i+1} iterations")
        break

    # 5. Relaxed update — blend old and new estimate to avoid oscillation
    P_drone = alpha * P_new + (1 - alpha) * P_drone

else:
    print(f"Warning: did not converge after {max_iter} iterations. Residual = {abs(P_new - P_drone):.4f} W")

# ── Results ──────────────────────────────────────────────────────────────────
print(f"P_drone  = {P_drone:.3f} W")
print(f"P_new    = {P_new:.3f} W")
print(f"m_total  = {m_total:.3f} kg")
print(f"m_motor  = {m_motor:.3f} kg")