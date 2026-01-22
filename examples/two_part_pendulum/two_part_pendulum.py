

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
from cardillo.force_laws.kelvin_voigt_element import KelvinVoigtElement
from cardillo.math import quat2axis_angle
from cardillo.utility.check_time_derivatives import check_time_derivatives
import csv

###############################################################################
# Parameters                                                                  #
###############################################################################

DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi


class TwoPartPendulum():
    # tuning zone -> use degrees, kilograms, meters

    def __init__(
        self,
        initial_position=0,
        initial_velocity=0,
        neutral_position=0,
        spring_constant=0,
        damper_constant=0,
        excitation=lambda t: 0*t,
        mass_1=1,
        mass_2=1,
        length_1=1,
        length_2=1,
        simulation_time=20,
        time_step=.001,
    ):

        # simulation parameters
        self.t_end = simulation_time
        self.dt = time_step
        self.time = np.linspace(0, self.t_end, int(self.t_end/self.dt))

        # model parameters
        self.m_1 = mass_1
        self.m_2 = mass_2
        l_1 = length_1
        l_2 = length_2
        self.l_1_hat = l_1/2
        self.l_2_hat = l_1 + l_2/2
        self.Theta_1 = self.m_1*l_1**2/3
        self.Theta_2 = self.m_2*((l_2/3)**2 + l_1**2 + l_1*l_2)
        self.phi_e = DEG2RAD * neutral_position

        self.c = spring_constant
        self.d = damper_constant
        # x_m = lambda t: 3/4*np.sin(4*t)
        # x_m_dot = lambda t: 3*np.cos(4*t)# lambda t : excitation(t)
        self.x_m = lambda t: excitation(t)

        # take first and second derivative
        _, self.x_m_dot, self.x_m_ddot = check_time_derivatives(
            f=self.x_m,
            f_t=None,
            f_tt=None,
        )

        self.g = 9.81

        # initial condition
        phi_0 = DEG2RAD * initial_position
        phi_dot_0 = DEG2RAD * initial_velocity

        #######################################################################
        # cardillo setup                                                      #
        #######################################################################

        # initial condition setup bar 1
        r_OC01 = np.array([
            self.l_1_hat*np.sin(phi_0) + self.x_m(0),
            -self.l_1_hat*np.cos(phi_0),
            0,
        ])
        v_C01 = np.array([
            self.l_1_hat*phi_dot_0*np.cos(phi_0) + self.x_m_dot(0),
            self.l_1_hat*phi_dot_0*np.sin(phi_0),
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
            [0, self.Theta_1, 0],
            [0, 0, self.Theta_1],
        ])

        # initial condition setup bar 2
        r_OC02 = np.array([
            self.l_2_hat*np.sin(phi_0) + self.x_m(0),
            -self.l_2_hat*np.cos(phi_0),
            0,
        ])
        v_C02 = np.array([
            self.l_2_hat*phi_dot_0*np.cos(phi_0) + self.x_m_dot(0),
            self.l_2_hat*phi_dot_0*np.sin(phi_0),
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
            [0, self.Theta_2, 0],
            [0, 0, self.Theta_2],
        ])

        self.system = System(t0=0)

        # fixed point
        fixed_point = Frame(
            r_OP=lambda t: np.array([self.x_m(t), 0, 0]),
            r_OP_t=lambda t: np.array([self.x_m_dot(t), 0, 0]),
            r_OP_tt=lambda t: np.array([self.x_m_ddot(t), 0, 0]),
            A_IB=np.eye(3),
            name="fixed point",
        )

        # bars
        bar_1 = RigidBody(
            mass=self.m_1,
            B_Theta_C=inertia_tensor_1,
            q0=q_01,
            u0=u_01,
            name="bar 1",
        )
        bar_2 = RigidBody(
            mass=self.m_2,
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
        spring_damper = KelvinVoigtElement(
            subsystem=pivot,
            k=self.c,
            d=self.d,
            l_ref=self.phi_e,
            compliance_form=False,
            name="rotational spring damper",
        )

        # gravity
        gravity_1 = Force(
            force=np.array([0, -self.m_1*self.g, 0]),
            subsystem=bar_1,
            B_r_CP=np.zeros(3),
            name="gravity_bar_1",
        )
        gravity_2 = Force(
            force=np.array([0, -self.m_2*self.g, 0]),
            subsystem=bar_2,
            B_r_CP=np.zeros(3),
            name="gravity_bar_2",
        )

        self.system.add(fixed_point)
        self.system.add(bar_1)
        self.system.add(bar_2)
        self.system.add(pivot)
        self.system.add(rigid_bar_connection)
        self.system.add(spring_damper)
        self.system.add(gravity_1)
        self.system.add(gravity_2)
        self.system.add(excitation_1)
        self.system.add(excitation_2)
        self.system.assemble()

        # initial condition ODE
        self.q_0 = np.array([phi_0, phi_dot_0])

        # variables to save the results
        self.solution_cardillo = None
        self.solution_ODE = None
        self.time = np.linspace(0, self.t_end, int(self.t_end/self.dt))

    ###########################################################################
    # simulation                                                              #
    ###########################################################################

    def simulate_cardillo(self):
        solver = Moreau(self.system, self.t_end, self.dt)
        self.solution_cardillo = solver.solve()

    # dynamics
    def equations_ODE(self, q, t):
        q_0_dot = q[1]
        q_1_dot = (
            -((self.m_1*self.l_1_hat + self.m_2*self.l_2_hat)
            * self.x_m_ddot(t)*np.cos(q[0])
            + self.c*(q[0]-self.phi_e) + self.d*q[1]
            + (self.m_1*self.l_1_hat + self.m_2*self.l_2_hat)
            * self.g*np.sin(q[0]))
            / (self.m_1*self.l_1_hat**2 + self.m_2*self.l_2_hat**2
            + self.Theta_1 + self.Theta_2)
        )
        return np.array([q_0_dot, q_1_dot])

    # solve ODE and plot solution


    def simulate_ODE(self):
        self.solution_ODE = odeint(self.equations_ODE, self.q_0, self.time)

    ###########################################################################
    # plot both results                                                       #
    ###########################################################################

    # extract ODE params
    def plot_results(self):
        time_o = self.time
        phi_o = self.solution_ODE[:, 0] * RAD2DEG
        phi_dot_o = self.solution_ODE[:, 1] * RAD2DEG

        # extract cardillo params
        # q is converted from quarterions to angles here, which is done
        # two lines below
        time_c = self.solution_cardillo.t
        q_c = np.stack(
            [quat2axis_angle(row) for row in self.solution_cardillo.q[:, 3:7]],
            axis=1,
        )

        phi_c = q_c[2] * RAD2DEG
        phi_dot_c = self.solution_cardillo.u[:, 5] * RAD2DEG

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
        fig, (plt_top, plt_bottom) = plt.subplots(2, 1, num=3)
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
        label = "excitation x_m [m] (from ODE sol)"
        plt.plot(time_o, self.x_m(time_o), 'r', label=label)
        label = "excitation x_m_dot [m/s] (from ODE sol)"
        plt.plot(time_o, self.x_m_dot(time_o), 'g', label=label)
        label = "excitation x_m_ddot [m/s²] (from ODE sol)"
        plt.plot(time_o, self.x_m_ddot(time_o), 'b', label=label)
        plt.xlabel("time [s]")
        plt.ylabel("[unit]")
        plt.legend()
        plt.grid()

        plt.show()

    ###########################################################################
    # write csv file                                                          #
    ###########################################################################

    def write_results_to_csv(self):
        file_path = 'examples/two_part_pendulum/two_part_pendulum_log.csv'

        q_c = np.stack(
            [quat2axis_angle(row) for row in self.solution_cardillo.q[:, 3:7]],
            axis=1,
        )

        phi_c = q_c[2] * RAD2DEG
        phi_dot_c = self.solution_cardillo.u[:, 5] * RAD2DEG

        with open(file_path, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            line = np.zeros(3)
            for index in range(0, self.solution_cardillo.t.size):
                line[0] = self.solution_cardillo.t[index]
                line[1] = phi_c[index]
                line[2] = phi_dot_c[index]
                writer.writerow(line)


if __name__ == "__main__":

    # tuning zone -> use degrees, kilograms, meters
    two_part_pendulum = TwoPartPendulum(
        initial_position=45,
        initial_velocity=0,
        neutral_position=90,
        spring_constant=4,
        damper_constant=2,
        excitation=lambda t: .3*np.sin(1*t),
        mass_1=1,
        mass_2=1,
        length_1=1,
        length_2=1,
        simulation_time=20,
        time_step=.001,
    )

    two_part_pendulum.simulate_cardillo()
    two_part_pendulum.simulate_ODE()
    two_part_pendulum.plot_results()

