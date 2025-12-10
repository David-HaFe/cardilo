

"""
    This script implements the rotating bar
    The used system of ODEs is
        q̇₁ = q₂
        q̇₂ = −dq₂ − c(q₁−φₑ)

    Date:   09.12.2025
    Author: David Hambach Ferrer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi

### tuning zone ### -> use degrees, kilograms, meters
initial_position =  90
initial_velocity =  0
neutral_position =  0

spring_constant =   1
damper_constant =   1

mass = 1
length = 1

simulation_time =   10
time_step =         .001
###################

# simulation parameters
t_end = simulation_time
dt = time_step
time = np.linspace(0, t_end, int(t_end/dt))

# model parameters
m = mass
l = length
Theta = ((m*l)^2)/12
phi_e = DEG2RAD * neutral_position

c = spring_constant
d = damper_constant

# initial condition
phi_0 = DEG2RAD * initial_position
phi_dot_0 = DEG2RAD * initial_velocity
q_0 = np.array([phi_0, phi_dot_0])

# dynamics
def equations(q,t):
    q_dot_1 = q[1]
    q_dot_2 = - d/Theta*q[1] - c/Theta*(q[0]-phi_e)
    return np.array([q_dot_1, q_dot_2])

# solve ODE and plot solution
sol = odeint(equations, q_0, time)

plt.plot(time, sol[:,0]*RAD2DEG, 'g', label="phi [deg]")
plt.plot(time, sol[:,1]*RAD2DEG, 'b', label="phi dot [deg/s]")
plt.xlabel("time [s]")
plt.ylabel("q")
plt.legend()
plt.grid()
plt.show()


