

"""
    This script implements the pendulum with two segments
    The used system of ODEs is
        q̇₁ = q₂
        TODO: update this, no longer accurate
        q̇₂ = (−dq₂ − c(q₁−φₑ) − (½l₁m₁ + (l₁+½l₂)m₂)(xₘ(t)+g)sin(q₁))/(Θ₁+Θ₂)

    Date:   09.12.2025
    Author: David Hambach Ferrer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from cardillo import System
from cardillo.discrete import RigidBody, Frame
from cardillo.constraints import Revolute, RigidConnection
from cardillo.solver import Moreau
from cardillo.forces import Force
from cardillo.force_laws._base import ScalarForceLawComplianceForm
from cardillo.math import quat2axis_angle

"""
    Implements the spring damper located at the base of the pendulum
    This is not an exact copy of the Kelvin-Voigt-Element, because here the
    force law uses rotational displacement instead of distance.
"""
class RotationalSpringDamper(ScalarForceLawComplianceForm):

    def __init__(
        self,
        subsystem,
        c,
        d,
        phi_0=0,
        compliance_form=True,
        name="rotational_spring_damper",
    ):
        super().__init__(subsystem, compliance_form)
        self.c = c
        self.d = d
        self.phi_0 = phi_0
        self.name = name

    def assembler_callback(self):
        super().assembler_callback()

    # spring
    def _E_pot(self, t, phi):
        return .5* self.c * (phi-self.phi_0) ** 2

    # lambda_c -> TODO: find out what this means here? [constraint?]
    def _la_c(self, t, phi, phi_dot):
        return -self.c * (phi-self.phi_0) - self.d*(phi_dot)

    # # lambda_c_l -> TODO: find out what l means here [lambda_c * l?
    def _la_c_l(self, t, phi, phi_dot):
        return -self.c

    # # lambda_c_l_dot ->
    def _la_c_l_dot(self, t, phi, phi_dot):
        return -self.d

    # c ->
    def _c(self, t, phi, phi_dot, lambda_c):
        return lambda_c / self.c + (phi-self.phi_0) + (self.d/self.c) * phi_dot

    # # c -> what does this mean?
    def _c_l(self, t, phi, phi_dot, lambda_c):
        return 1

    # # c ->
    def _c_l_dot(self, t, phi, phi_dot, lambda_c):
        return self.d / self.c

    def c_la_c(self):
        return 1/self.c

if __name__ == "__main__":
    ###########################################################################
    # Parameters                                                              #
    ###########################################################################

    DEG2RAD = np.pi/180
    RAD2DEG = 180/np.pi

    ###########################################################################
    # tuning zone -> use degrees, kilograms, meters                           #
    ###########################################################################
    initial_position =  45
    initial_velocity =  0
    neutral_position =  45

    spring_constant =   0
    damper_constant =   0
    excitation =  lambda t: 0*t # 3*np.sin(1*t)

    mass_1 = 1
    mass_2 = 1
    length_1 = 1
    length_2 = 1

    simulation_time =   10
    time_step =         .005
    ###########################################################################

    # simulation parameters
    t_end = simulation_time
    dt = time_step
    time = np.linspace(0, t_end, int(t_end/dt))

    # model parameters
    m_1 = mass_1
    m_2 = mass_2
    l_1 = length_1
    l_2 = length_2
    l_1_hat = l_1/2
    l_2_hat = l_1 + l_2/2
    Theta_1 = m_1*l_1**2/3
    Theta_2 = m_2*((l_2/3)**2 + l_1**2 + l_1*l_2)
    phi_e = DEG2RAD * neutral_position

    c = spring_constant
    d = damper_constant
    # x_m = lambda t: 3/4*np.sin(4*t)
    # x_m_dot = lambda t: 3*np.cos(4*t)# lambda t : excitation(t)
    x_m_ddot = lambda t: excitation(t)
    x_m_0 = 0
    x_m_dot_0 = 0

    g = 9.81

    # initial condition
    phi_0 = DEG2RAD * initial_position
    phi_dot_0 = DEG2RAD * initial_velocity

    ###########################################################################
    # cardillo setup                                                          #
    ###########################################################################

    # initial condition setup bar 1
    r_OC01 = np.array([
        l_1_hat*np.sin(phi_0) + x_m_0,
        -l_1_hat*np.cos(phi_0),
        0,
    ])
    v_C01 = np.array([
        l_1_hat*phi_dot_0*np.cos(phi_0) + x_m_dot_0,
        l_1_hat*phi_dot_0*np.sin(phi_0),
        0,
    ])
    A_IB01 = np.array([
        [np.cos(phi_0), -np.sin(phi_0), 0],
        [np.sin(phi_0), np.cos(phi_0), 0],
        [0, 0, 1],
    ])
    B_Omega01 = np.array([0, 0, phi_dot_0])
    q_01 = RigidBody.pose2q(r_OC01, A_IB01)
    u_01 = np.hstack([v_C01, B_Omega01])
    inertia_tensor_1 = np.array([
        [0, 0, 0],
        [0, Theta_1, 0],
        [0, 0, Theta_1],
    ])

    # initial condition setup bar 2
    r_OC02 = np.array([
        l_2_hat*np.sin(phi_0) + x_m_0,
        -l_2_hat*np.cos(phi_0),
        0,
    ])
    v_C02 = np.array([
        l_2_hat*phi_dot_0*np.cos(phi_0) + x_m_dot_0,
        l_2_hat*phi_dot_0*np.sin(phi_0),
        0,
    ])
    A_IB02 = np.array([
        [np.cos(phi_0), -np.sin(phi_0), 0],
        [np.sin(phi_0), np.cos(phi_0), 0],
        [0, 0, 1],
    ])
    B_Omega02 = np.array([0, 0, phi_dot_0])
    q_02 = RigidBody.pose2q(r_OC02, A_IB02)
    u_02 = np.hstack([v_C02, B_Omega02])
    inertia_tensor_2 = np.array([
        [0, 0, 0],
        [0, Theta_2, 0],
        [0, 0, Theta_2],
    ])

    system = System(t0=0)

    # fixed point
    fixed_point = Frame(
        r_OP = np.zeros(3),
        A_IB = np.eye(3),
        name = "fixed point",
    )

    # bars
    bar_1 = RigidBody(
        mass=m_1,
        B_Theta_C=inertia_tensor_1,
        q0=q_01,
        u0=u_01,
        name="bar 1",
    )
    bar_2 = RigidBody(
        mass=m_2,
        B_Theta_C=inertia_tensor_2,
        q0=q_02,
        u0=u_02,
        name="bar 2",
    )

    # pivot point
    pivot = Revolute(
        subsystem1=fixed_point,
        subsystem2=bar_1,
        axis=2,
        r_OJ0=np.zeros(3),
        A_IJ0=np.eye(3),
        angle0=phi_0,
        name="pivot",
    )

    # connection between bars
    rigid_bar_connection = RigidConnection(
        subsystem1=bar_1,
        subsystem2=bar_2,
        r_OJ0=np.zeros(3),
        A_IJ0=np.eye(3),
        name="connection bar 1 and 2",
    )

    # spring damper coupling
    spring_damper = RotationalSpringDamper(
        subsystem=pivot,
        c=c,
        d=d,
        phi_0=0,
        compliance_form=False,
        name="rotational spring damper",
    )

    # gravity
    gravity_1 = Force(
        force=np.array([0, -m_1*g, 0]),
        subsystem=bar_1,
        B_r_CP=np.zeros(3),
        name="gravity_bar_1",
    )
    gravity_2 = Force(
        force=np.array([0, -m_2*g, 0]),
        subsystem=bar_2,
        B_r_CP=np.zeros(3),
        name="gravity_bar_2",
    )

    # excitation
    excitation_1 = Force(
        force=lambda t: np.array([m_1*x_m_ddot(t), 0, 0]),
        subsystem=bar_1,
        B_r_CP=np.zeros(3),
        name="excitation force bar 1",
    )
    excitation_2 = Force(
        force=lambda t: np.array([m_2*x_m_ddot(t), 0, 0]),
        subsystem=bar_2,
        B_r_CP=np.zeros(3),
        name="excitation force bar 2",
    )

    system.add(fixed_point)
    system.add(bar_1)
    system.add(bar_2)
    system.add(pivot)
    system.add(rigid_bar_connection)
    system.add(spring_damper)
    system.add(gravity_1)
    system.add(gravity_2)
    system.add(excitation_1)
    system.add(excitation_2)
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
        q_0_dot = q[1]
        q_1_dot = (
            -((m_1*l_1_hat + m_2*l_2_hat)*x_m_ddot(t)*np.cos(q[0])
            + c*(q[0]-phi_e) + d*q[1]
            + (m_1*l_1_hat + m_2*l_2_hat)*g*np.sin(q[0]))
            /(m_1*l_1_hat**2 + m_2*l_2_hat**2 + Theta_1 + Theta_2)
        )
        return np.array([q_0_dot, q_1_dot])

    # solve ODE and plot solution
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
    q_c = np.stack(
        [quat2axis_angle(row) for row in solution_cardillo.q[:,3:]], axis=1,
    )
    print(solution_cardillo.q)

    phi_c = q_c[2] * RAD2DEG
    phi_dot_c = solution_cardillo.u[:,5] * RAD2DEG
    # ODE solution
    plt.figure(1)
    plt.plot(time_o, phi_o, 'b', label="phi [deg]")
    plt.plot(time_o, phi_dot_o, 'r', label="phi dot [deg/s]")
    plt.xlabel("time [s]")
    plt.ylabel("q")
    plt.title("Solution from ODE")
    plt.legend()
    plt.grid()

    # cardillo solution
    plt.figure(2)
    plt.plot(time_c, phi_c, 'c', label="phi [deg]")
    plt.plot(time_c, phi_dot_c, 'm', label="phi dot [deg/s]")
    plt.xlabel("time [s]")
    plt.ylabel("q")
    plt.title("Solution from Cardillo")
    plt.legend()
    plt.grid()

    # side by side
    fig, (plt_top, plt_bottom) = plt.subplots(2,1,num=3)
    plt.suptitle("Side by side comparison")

    plt_top.plot(time_o, phi_o, 'b', label="phi ODE")
    plt_top.plot(time_c, phi_c, 'c', label="phi cardillo")
    plt_top.set_xlabel("time [s]")
    plt_top.set_ylabel("phi [deg]")
    plt_top.legend()
    plt_top.grid()

    plt_bottom.plot(time_o, phi_dot_o, 'r', label="phi dot ODE")
    plt_bottom.plot(time_c, phi_dot_c, 'm', label="phi dot cardillo")
    plt_bottom.set_xlabel("time [s]")
    plt_bottom.set_ylabel("phi dot [deg/s]")
    plt_bottom.legend()
    plt_bottom.grid()

    # excitation
    plt.figure(4)
    plt.plot(time_o, x_m_ddot(time_o), 'g', label="excitation x_m_ddot (from ODE sol)")
    plt.xlabel("time [s]")
    plt.ylabel("x_m ddot [m/s]")
    plt.legend()
    plt.grid()

    plt.show()


