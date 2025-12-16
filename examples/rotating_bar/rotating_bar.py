

"""
    This script implements the rotating bar
    The used system of ODEs is
        q̇₁ = q₂
        q̇₂ = −dq₂ − c(q₁−φₑ)

    Date:   09.12.2025
    Author: David Hambach Ferrer
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Cylinder
from cardillo.forces import Moment
from cardillo.solver import Moreau

if __name__ == "__main__":

    ##############
    # Parameters #
    ##############

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

    simulation_time =   5
    time_step =         .005
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

    ###################
    # assemble system #
    ###################

    # initial condition
    # TODO: probably adjust initial condition to 3d case
    phi_0 = (DEG2RAD * initial_position)
    phi_dot_0 = DEG2RAD * initial_velocity

    r_OC0 = np.zeros(3)
    v_C0 = np.zeros(3)
    A_IB0 = np.array([
        [np.cos(phi_0), -np.sin(phi_0), 0],
        [np.sin(phi_0), np.cos(phi_0), 0],
        [0, 0, 1],
    ])
    B_Omega0 = np.array([0, 0, phi_dot_0])
    # A_IB0 = np.zeros((3,3))
    # B_Omega0 = np.zeros(3)

    inertia_tensor = np.array([
        [Theta, 0, 0],
        [0, Theta, 0],
        [0, 0, Theta],
    ])

    system = System(t0 = 0)
    q_0 = RigidBody.pose2q(r_OC0, A_IB0)
    u_0 = np.hstack([v_C0, B_Omega0])

    # create bar
    bar = RigidBody(
        mass = m,
        B_Theta_C = inertia_tensor,
        q0 = q_0,
        u0 = u_0,
        name = "bar",
    )

    # forces
    # CAUTION:  I wanted to get the rotation of the bar, and I assumed that
    #           this is given by q0[6]. This is just my guess though
    spring = Moment(np.array([0, 0, c*(bar.q0[6]-phi_e)]), bar, name = "spring force")
    damper = Moment(np.array([0, 0, d*bar.u0[5]]), bar, name = "damper force")

    system.add(bar)
    system.add(spring)
    system.add(damper)
    system.assemble()

    ##############
    # simulation #
    ##############

    solver = Moreau(system, t_end, dt)
    # dynamics
    def equations(q,t):
        q_dot_1 = q[1]
        q_dot_2 = - d/Theta*q[1] - c/Theta*(q[0]-phi_e)
        return np.array([q_dot_1, q_dot_2])

    # solve ODE and plot solution
    sol = solver.solve()
    time = sol.t
    phi = sol.q
    phi_dot = sol.u

    plt.plot(time, phi*RAD2DEG, 'g', label="phi [deg]")
    plt.plot(time, phi_dot*RAD2DEG, 'b', label="phi dot [deg/s]")
    plt.xlabel("time [s]")
    plt.ylabel("q")
    plt.legend()
    plt.grid()
    plt.show()


