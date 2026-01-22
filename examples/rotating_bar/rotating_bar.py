

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
from scipy.integrate import odeint

from cardillo import System
from cardillo.discrete import RigidBody, Frame
from cardillo.constraints import Revolute
from cardillo.solver import Moreau
from cardillo.force_laws._base import ScalarForceLawComplianceForm
from cardillo.force_laws.kelvin_voigt_element import KelvinVoigtElement
from cardillo.math import quat2axis_angle

###############################################################################
# Parameters                                                                  #
###############################################################################

DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi

class RotatingBar():

    def __init__(
        self,
        initial_position=0,
        initial_velocity=0,
        neutral_position=0,
        spring_constant=0,
        damper_constant=0,
        mass=1,
        length=1,
        simulation_time=10,
        time_step=.005,
    ):
        # simulation parameters
        self.t_end = simulation_time
        self.dt = time_step
        self.time = np.linspace(0, self.t_end, int(self.t_end/self.dt))

        # model parameters
        self.m = mass
        self.l = length
        self.Theta = ((self.m*self.l)**2)/12
        self.phi_e = DEG2RAD * neutral_position

        self.c = spring_constant
        self.d = damper_constant

        # initial condition
        phi_0 = DEG2RAD * initial_position
        phi_dot_0 = DEG2RAD * initial_velocity

        #######################################################################
        # cardillo setup                                                      #
        #######################################################################

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

        # CAUTION:  This tensor might be incorrect, namely one Theta might
        #           be on the other diagonal element.
        inertia_tensor = np.array([
            [0, 0, 0],
            [0, self.Theta, 0],
            [0, 0, self.Theta],
        ])

        self.system = System(t0=0)

        # fixed point
        fixed_point = Frame(
            r_OP=np.zeros(3),
            A_IB=np.eye(3),
            name="fixed point",
        )

        # bar
        bar = RigidBody(
            mass=self.m,
            B_Theta_C=inertia_tensor,
            q0=q_0,
            u0=u_0,
            name="bar",
        )

        # pivot point
        pivot = Revolute(
            subsystem1=fixed_point,
            subsystem2=bar,
            axis=2,
            r_OJ0=np.zeros(3),
            A_IJ0=np.eye(3),
            angle0=phi_0,
            name="pivot",
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

        self.system.add(fixed_point)
        self.system.add(bar)
        self.system.add(pivot)
        self.system.add(spring_damper)
        self.system.assemble()

        # initial condition ODE
        self.q_0 = np.array([phi_0, phi_dot_0])

        # save results
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
        q_dot_1 = q[1]
        q_dot_2 = (
            - self.d/self.Theta*q[1]
            - self.c/self.Theta*(q[0]-self.phi_e)
        )
        return np.array([q_dot_1, q_dot_2])

    def simulate_ODE(self):
        self.solution_ODE = odeint(self.equations_ODE, self.q_0, self.time)

    ###########################################################################
    # plot both results                                                       #
    ###########################################################################

    def plot_results(self):
        # extract ODE params
        time_o = self.time
        phi_o = self.solution_ODE[:, 0] * RAD2DEG
        phi_dot_o = self.solution_ODE[:, 1] * RAD2DEG

        # extract cardillo params
        # q is converted from quarterions to angles here, which is done
        # two lines below
        time_c = self.solution_cardillo.t
        q_c = np.stack(
            [quat2axis_angle(row) for row in self.solution_cardillo.q[:, 3:]],
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

        plt.show()

if __name__ == "__main__":
    # tuning zone -> use degrees, kilograms, meters
    rotating_bar = RotatingBar(
        initial_position=90,
        initial_velocity=20,
        neutral_position=45,
        spring_constant=1,
        damper_constant=1,
        mass=1,
        length=1,
        simulation_time=5,
        time_step=.005,
    )
    rotating_bar.simulate_cardillo()
    rotating_bar.simulate_ODE()
    rotating_bar.plot_results()


