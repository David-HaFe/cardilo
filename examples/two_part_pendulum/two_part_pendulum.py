

"""
    This script implements the pendulum with two segments
    The used system of ODEs is
        q̇₁ = q₂
        q̇₂ = (−dq₂ − c(q₁−φₑ) − (½l₁m₁ + (l₁+½l₂)m₂)(xₘ(t)+g)sin(q₁))/(Θ₁+Θ₂)

    Date:   09.12.2025
    Author: David Hambach Ferrer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.constants as constants

DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi

### tuning zone ### -> use degrees, kilograms, meters, seconds
initial_position =  1
initial_velocity =  0
neutral_position =  45

spring_constant =   0
damper_constant =   0
excitation = lambda t : 20*np.sin(.1*t)

mass_1 = 1
mass_2 = 1
length_1 = 1
length_2 = 1

simulation_time =   100
time_step =         .001
###################

# simulation parameters
t_end = simulation_time
dt = time_step
time = np.linspace(0, t_end, int(t_end/dt))

# model parameters
m_1 = mass_1
m_2 = mass_2
l_1 = length_1
l_2 = length_2
Theta_1 = m_1*l_1**2/4
Theta_2 = m_2*(l_1 + l_2/2)**2
phi_e = DEG2RAD * neutral_position

c = spring_constant
d = damper_constant
x_m_dot = lambda t: 3*np.cos(4*t)# lambda t : excitation(t)

g = 9.81

# initial condition
phi_0 = DEG2RAD * initial_position
phi_dot_0 = DEG2RAD * initial_velocity
q_0 = np.array([phi_0, phi_dot_0])

# dynamics
def equations(q,t):
    q_dot_1 = q[1]
    q_dot_2 = (-d*q[1] - c*(q[0] - phi_e) - (l_1/2*m_1 + (l_1+l_2/2)*m_2)*(x_m_dot(t) + g)*np.sin(q[0]))/(Theta_1 + Theta_2)
    return np.array([q_dot_1, q_dot_2])

# solve ODE and plot solution
sol = odeint(equations, q_0, time)

fig, (plt_high, plt_low) = plt.subplots(2)

plt_high.plot(time, sol[:,0]*RAD2DEG, 'g', label="phi [deg]")
plt_high.plot(time, sol[:,1]*RAD2DEG, 'b', label="phi dot [deg/s]")
plt_high.set_xlabel("time [s]")
plt_high.set_ylabel("q")
plt_high.legend()
plt_high.grid()

plt_low.plot(time, x_m_dot(time), 'r', label="x_m [m/s]")
plt_low.set_xlabel("time [s]")
plt_low.set_ylabel("x_m_dot(t) [m/s]")
plt_low.grid()

plt.show()


