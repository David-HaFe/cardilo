

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from cardillo import System
from cardillo.discrete import RigidBody, Frame
from cardillo.constraints import Revolute
from cardillo.solver import Moreau
from cardillo.force_laws.kelvin_voigt_element import KelvinVoigtElement
from cardillo.math import quat2axis_angle

DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi

"""
    This class implements the rotating bar in cardillo as well as with an ODE.
    The following ODE is used with q₁= φ, q₂= φ̇
        q̇₁ = q₂
               -dq₂ - c(q₁−φₑ)
        q̇₂ = ------------------
                     Θ
    Date:   09.12.2025
    Author: David Hambach Ferrer
"""
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

        # variables to save the results
        self.solution_cardillo = None
        self.solution_ODE = None
        self.time = np.linspace(0, self.t_end, int(self.t_end/self.dt))

    ###########################################################################
    # simulation                                                              #
    ###########################################################################

    """
        simulates everything according to the defined setup using cardillo
    """
    def simulate_cardillo(self):
        solver = Moreau(self.system, self.t_end, self.dt)
        self.solution_cardillo = solver.solve()

    """
        contains system dynamics that are passed on to the ode
    """
    def equations_ODE(self, q, t):
        q_dot_1 = q[1]
        q_dot_2 = (
            - self.d/self.Theta*q[1]
            - self.c/self.Theta*(q[0]-self.phi_e)
        )
        return np.array([q_dot_1, q_dot_2])

    """
        simulates everything according to the defined setup using the ode
    """
    def simulate_ODE(self):
        self.solution_ODE = odeint(self.equations_ODE, self.q_0, self.time)

    ###########################################################################
    # plotting                                                                #
    ###########################################################################

    """
        takes the cardillo result vector and returns the kinetic and potential
        energy for the system.
                         1                           1
        kinetic energy = - Θ (φ̇ )², potential energy = - c (φ-φₑ)²
                         2                           2
    """
    def _calculate_energy(
        self,
        phi: np.array,
        phi_dot: np.array,
    ):
        kinetic_energy = .5*self.Theta*np.square(phi_dot)
        potential_energy = .5*self.c*np.square(phi-self.phi_e)
        total_energy = kinetic_energy + potential_energy
        return kinetic_energy, potential_energy, total_energy

    """
        takes the cardillo result and converts the quarternion to a proper
        angle and angular velocity
    """
    def _extract_cardillo_results(self):
        q_c = np.stack(
            [quat2axis_angle(row) for row in self.solution_cardillo.q[:, 3:7]],
            axis=1,
        )

        phi_c = q_c[2]
        phi_dot_c = self.solution_cardillo.u[:, 5]
        return phi_c, phi_dot_c

    """
        generates plots to visually compare the two results to each other
        simulate_cardillo and simulate_ODE have to be called first
    """
    def plot_results(self):
        # extract ODE params -> '_o' means variable comes from ODE
        time_o = self.time
        phi_o = self.solution_ODE[:, 0]
        phi_dot_o = self.solution_ODE[:, 1]
        np.asarray(phi_o)
        np.asarray(phi_dot_o)
        e_kin_o, e_pot_o, energy_o = self._calculate_energy(phi_o, phi_dot_o)

        # extract cardillo params -> '_c' means variable comes from cardillo
        time_c = self.solution_cardillo.t
        phi_c, phi_dot_c = self._extract_cardillo_results()
        e_kin_c, e_pot_c, energy_c = self._calculate_energy(phi_c, phi_dot_c)

        # convert everything from rad to degrees for plotting
        phi_o = RAD2DEG * phi_o
        phi_dot_o = RAD2DEG * phi_dot_o
        phi_c = RAD2DEG * phi_c
        phi_dot_c = RAD2DEG * phi_dot_c

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

        # energy of the systems
        fig, (plt_top, plt_bottom) = plt.subplots(2, 1, num=4)
        plt.suptitle("Energy of the system (ODE)")

        plt_top.plot(time_o, e_kin_o, 'r', label="kinetic energy of ODE")
        plt_top.plot(time_o, e_pot_o, 'g', label="potential energy of ODE")
        plt_top.plot(time_o, energy_o, 'b', label="total energy of ODE")
        plt_top.set_xlabel("time [s]")
        plt_top.set_ylabel("Energy [J]")
        plt_top.legend()
        plt_top.grid()

        plt_bottom.plot(time_c, e_kin_c, 'r', label="kinetic energy cardillo")
        plt_bottom.plot(time_c, e_pot_c, 'g',
                        label="potential energy cardillo")
        plt_bottom.plot(time_c, energy_c, 'b', label="total energy cardillo")
        plt_bottom.set_xlabel("time [s]")
        plt_bottom.set_ylabel("Energy [J]")
        plt_bottom.legend()
        plt_bottom.grid()

        plt.show()


if __name__ == "__main__":
    # tuning -> use degrees, kilograms, meters
    rotating_bar = RotatingBar(
        initial_position=90,
        initial_velocity=20,
        neutral_position=35,
        spring_constant=1,
        damper_constant=0,
        mass=1,
        length=1,
        simulation_time=5,
        time_step=.005,
    )
    rotating_bar.simulate_cardillo()
    rotating_bar.simulate_ODE()
    rotating_bar.plot_results()


