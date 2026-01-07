

"""
    This script implements the rotating bar in cardillo as well as with an ODE.
    The following ODE is used
        q₁= φ, q₂= φ̇
        q̇₁ = q₂
        q̇₂ = −dq₂ − c(q₁−φₑ)

    Date:   09.12.2025
    Author: David Hambach Ferrer
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.integrate import odeint

from cardillo import System
from cardillo.discrete import RigidBody, Frame
from cardillo.constraints import Revolute
from cardillo.forces import Moment
from cardillo.solver import Moreau
from cardillo.math import quat2axis_angle
from cardillo.interactions import TwoPointInteraction
from cardillo.force_laws._base import ScalarForceLawComplianceForm

"""
    Implements the spring damper located at the base of the pendulum
"""
class rotational_spring_damper(ScalarForceLawComplianceForm):

    def __init(
        self,
        subsystem,
        c,
        d,
        phi_0=0,
        compliance_form=True,
        name="rotational_spring_damper",
    ):
        super().__init__(sybsystem, compliance_form)
        self.c = c
        self.d = d
        self.phi_0 = phi_0
        self.name = name

    def assembler_callback(self):
        super().assembler_callback()

    # TODO: you need the rotation here instead of the distance.
    #       find out where you can get that from.
    # spring
    def _E_pot(self, t, phi):
        return .5* self.c * (phi-self.phi_0) ** 2

    # lambda_c -> TODO: find out what means here? [constraint?]
    def _la_c(self, t, phi, phi_dot):
        return -self.c * (phi-self.phi_0) - self.d*(phi_dot)

    # lambda_c_l -> TODO: find out what l means here [lambda_c * l?
    def _la_c_l(self, t, phi, phi_dot):
        return -self.c

    # lambda_c_l_dot ->
    def _la_c_l_dot(self, t, phi, phi_dot):
        return self.d

    # c ->
    def _c(self, t, phi, phi_dot, lambda_c):
        return lambda_c / self.c + (phi-self.phi_0) + (self.d/self.c) * phi_dot

    # c -> what does this mean?
    def _c_l(self, t, phi, phi_dot, lambda_c):
        return 1

    # c ->
    def _c_l_dot(self, t, phi, phi_dot, lambda_c):
        return self.d / self.c

    def c_la_c(self):
        return 1/self.k

if __name__ == "__main__":

    ###########################################################################
    # Parameters                                                              #
    ###########################################################################

    DEG2RAD = np.pi/180
    RAD2DEG = 180/np.pi

    ###########################################################################
    # tuning zone -> use degrees, kilograms, meters                           #
    ###########################################################################
    initial_position =  90
    initial_velocity =  10
    neutral_position =  0

    spring_constant =   0
    damper_constant =   0

    mass =      1
    length =    1

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

    # initial condition
    phi_0 = DEG2RAD * initial_position
    phi_dot_0 = DEG2RAD * initial_velocity

    ###########################################################################
    # cardillo setup                                                          #
    ###########################################################################

    r_OC0 = np.zeros(3)
    v_C0 = np.zeros(3)
    A_IB0 = np.array([
        [np.cos(phi_0), -np.sin(phi_0), 0],
        [np.sin(phi_0), np.cos(phi_0), 0],
        [0, 0, 1],
    ])
    B_Omega0 = np.array([0, 0, phi_dot_0])
    q_0 = RigidBody.pose2q(r_OC0, A_IB0)
    u_0 = np.hstack([v_C0, B_Omega0])

    # CAUTION:  This tensor might be incorrect, namely one Theta might be on
    #           the other diagonal element.
    inertia_tensor = np.array([
        [0, 0, 0],
        [0, Theta, 0],
        [0, 0, Theta],
    ])

    system = System(t0 = 0)

    # fixed point
    fixed_point = Frame(
        r_OP = np.zeros(3),
        A_IB = np.eye(3),
        name = "fixed point",
    )

    # bar
    bar = RigidBody(
        mass = m,
        B_Theta_C = inertia_tensor,
        q0 = q_0,
        u0 = u_0,
        name = "bar",
    )

    # bearing
    bearing = Revolute(
        fixed_point,
        bar,
        axis = 2,
        r_OJ0 = np.zeros(3),
        A_IJ0 = np.eye(3),
        angle0 = phi_0,
        name = "bearing",
    )

    # forces
    # CAUTION:  I wanted to get the rotation of the bar, and I assumed that
    #           this is given by q0[6].
    ground_to_bar = TwoPointInteraction(
        subsystem1 = fixed_point,
        subsystem2 = bar,
        name = "ground-to-bar connection",
    )
    spring = Moment(
        np.array([0, 0, -c*(ground_to_bar.r_OP2-phi_e)]),
        bar,
        name = "spring force",
    )
    damper = Moment(
        np.array([0, 0, -d*bar.u0[5]]),
        bar,
        name = "damper force",
    )

    system.add(fixed_point)
    system.add(bar)
    system.add(bearing)
    system.add(spring)
    system.add(damper)
    system.assemble()

    ###########################################################################
    # simulation                                                              #
    ###########################################################################

    solver = Moreau(system, t_end, dt)
    solution_cardillo = solver.solve()

    ###########################################################################
    # ODE setup                                                               #
    ###########################################################################

    # initial condition
    q_0 = np.array([phi_0, phi_dot_0])

    # dynamics
    def equations(q,t):
        q_dot_1 = q[1]
        q_dot_2 = - d/Theta*q[1] - c/Theta*(q[0]-phi_e)
        return np.array([q_dot_1, q_dot_2])

    time = np.linspace(0, t_end, int(t_end/dt))
    solution_ODE = odeint(equations, q_0, time)

    ###########################################################################
    # plot both results                                                       #
    ###########################################################################

    # extract ODE params
    time_o = time
    phi_o = solution_ODE[:,0] * RAD2DEG
    phi_dot_o = solution_ODE[:,1] * RAD2DEG

    # extract cardillo params
    # q is converted from quarterions to angles here, which is done
    # two lines below
    time_c = solution_cardillo.t
    q_c = np.stack([quat2axis_angle(row) for row in solution_cardillo.q[:,3:]], axis=1)
    phi_c = q_c[2] * RAD2DEG
    phi_dot_c = solution_cardillo.u[:,5] * RAD2DEG

    # ODE solution
    plt.plot(time_o, phi_o, 'b', label="phi [deg]")
    plt.plot(time_o, phi_dot_o, 'r', label="phi dot [deg/s]")
    plt.xlabel("time [s]")
    plt.ylabel("q")
    plt.title("Solution from ODE")
    plt.legend()
    plt.grid()
    plt.show()

    # cardillo solution
    plt.plot(time_c, phi_c, 'c', label="phi [deg]")
    plt.plot(time_c, phi_dot_c, 'm', label="phi dot [deg/s]")
    plt.xlabel("time [s]")
    plt.ylabel("q")
    plt.title("Solution from Cardillo")
    plt.legend()
    plt.grid()
    plt.show()

    # side by side
    fig, ax = plt.subplots(2,1)
    plt.title("Side by side comparison")

    ax[0].plot(time_o, phi_o, 'b', label="phi ODE")
    ax[0].plot(time_c, phi_c, 'c', label="phi cardillo")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("phi [deg]")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(time_o, phi_dot_o, 'r', label="phi dot ODE")
    ax[1].plot(time_c, phi_dot_c, 'm', label="phi dot cardillo")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("phi dot [deg/s]")
    ax[1].legend()
    ax[1].grid()

    plt.show()

