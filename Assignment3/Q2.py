import numpy as np


# Ingenuity parameters
m_ingenuity = 1.8 # kg
m_ingenuity_no_fuselage = 1.8 - 0.3 # kg
R_ingenuity = 0.6 # m
P_ingenuity = 88.64 # W

# Drone parameters
N_blade = 3 # blades per rotor
N_rotor = 2 # number of rotors
R = 1 # m
m_blade = 70/4 * N_blade * R/R_ingenuity # kg
P_new = 

m_payload = 2 # kg
m_battery_pack = 0.5 # kg
m_components = 1.0 # kg
m_propellers = N_blade * m_blade * N_rotor # kg
m_motor_per_rotor = (0.25/2 * P_new/P_ingenuity) # kg
m_motor = m_motor_per_rotor * N_rotor # kg
m_no_fuselage = m_payload + m_battery_pack + m_components + m_propellers + m_motor # kg
m_fuselage = m_no_fuselage * (m_ingenuity)/(m_ingenuity_no_fuselage) # kg

# The Drones Total Weight
m_total = m_fuselage + m_no_fuselage # kg