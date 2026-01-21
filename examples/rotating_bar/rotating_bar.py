

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

    # TODO: you need the rotation here instead of the distance.
    #       find out where you can get that from.
    # spring
    def _E_pot(self, t, phi):
        return .5* self.c * (phi-self.phi_0) ** 2

    # lambda_c -> TODO: find out what means here? [constraint?]
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
    initial_position =  90
    initial_velocity =  10
    neutral_position =  0

    spring_constant =   1
    damper_constant =   1

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

    # pivot
    pivot = Revolute(
        fixed_point,
        bar,
        axis = 2,
        r_OJ0 = np.zeros(3),
        A_IJ0 = np.eye(3),
        angle0 = phi_0,
        name = "pivot",
    )

    # forces
    ground_to_bar = TwoPointInteraction(
        subsystem1 = fixed_point,
        subsystem2 = bar,
        name = "ground-to-bar connection",
    )
    spring_damper = RotationalSpringDamper(
        subsystem=pivot,
        c=c,
        d=d,
        phi_0=0,
        compliance_form=False,
        name="rotational spring damper",
    )

    system.add(fixed_point)
    system.add(bar)
    system.add(pivot)
    system.add(spring_damper)
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
    q_c = np.stack(
        [quat2axis_angle(row) for row in solution_cardillo.q[:,3:]], axis=1,
    )
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
    fig, ax = plt.subplots(2,1,num=3)
    plt.suptitle("Side by side comparison")

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

"""
    This is the start of the rotational two point interaction,
    which seems to be obsolete at the time of writing this comment.
    I will still leave it here for now just in case.
"""
# class TwoPointRotationalInteraction:
#         r"""Interface for scalar force interaction between two points.
#         Provides rotation between the points, its time derivatives and the
#         generalized force direction of the scalar force acting along the
#         connection line between the two points.
#         This is similar to the two point interaction, but instead of measuring
#         the euclidean distance,
#         the rotation around a specified axis is measured.
#         This works with the Moreau solver, and methods needed for other solvers
#         may or may not be implemented yet.
#
#         Parameters
#         ----------
#         subsystem1 : object
#             Object containing first point of interaction (P1)
#         subsystem2 : object
#             Object containing second point of interaction (P2)
#         xi1 : #TODO
#         xi2 : #TODO
#         axis: defines which axes should be compared (0->x, 1->y, 2->z)
#         : np.ndarray (3,)
#             Position vector of first point (P1) w.r.t. center of mass (C) in
#             body-fixed K-basis of subsystem1.
#         : np.ndarray (3,)
#             Position vector of second point (P2) w.r.t. center of mass (C) in
#             body-fixed K-basis of subsystem2.
#         name : str
#             Name of contribution.
#         """
#     def __init__(
#         self,
#         subsystem1,
#         subsystem2,
#         xi1 = None,
#         xi2 = None,
#         axis = 2,
#         B_r_CP1 = np.zeros(3, dtype=float),
#         B_r_CP2 = np.zeros(3, dtype=float),
#         A_IB1 = np.eye(3, dtype=float),
#         A_IB2 = np.eye(3, dtype=float),
#         name = "two_point_rotational_interaction",
#     ):
#         self.subsystem1 = subsystem1
#         self.xi1 = xi1
#         self.B_r_CP1 = B_r_CP1
#         self.A_IB1 = A_IB1
#
#         self.subsystem2 = subsystem2
#         self.xi2 = xi2
#         self.B_r_CP2 = B_r_CP2
#         self.A_IB2 = A_IB2
#
#         self.name = name
#
#     def assembler_callback(self):
#         qDOF1 = self.subsystem1.qDOF
#         qDOF2 = self.subsystem2.qDOF
#         local_qDOF1 = self.subsystem1.local_qDOF_P(self.xi1)
#         local_qDOF2 = self.subsystem2.local_qDOF_P(self.xi2)
#         self.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
#         self._nq1 = len(local_qDOF1)
#         self._nq2 = len(local_qDOF2)
#         self._nq = self._nq1 + self._nq2
#         self.t0 = self.subsystem1.t0
#         q01 = self.subsystem1.q0
#         q02 = self.subsystem2.q0
#         self.q0 = np.concatenate((q01[local_qDOF1], q02[local_qDOF2]))
#
#         uDOF1 = self.subsystem1.uDOF
#         uDOF2 = self.subsystem2.uDOF
#         local_uDOF1 = self.subsystem1.local_uDOF_P(self.xi1)
#         local_uDOF2 = self.subsystem2.local_uDOF_P(self.xi2)
#         self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
#         self._nu1 = len(local_uDOF1)
#         self._nu2 = len(local_uDOF2)
#         self._nu = self._nu1 + self._nu2
#         u01 = self.subsystem1.u0
#         u02 = self.subsystem2.u0
#         self.u0 = np.concatenate((u01[local_uDOF1], u02[local_uDOF2]))
#
#         self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
#             t, q[: self._nq1], self.xi1, self.B_r_CP1
#         )
#         self.A_IB1 = lambda t,y: self.subsystem1.A_IB(
#             t,q[: self._nq1], self.xi1, self.B_r_CP1
#         )
#         self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
#             t, q[: self._nq1], self.xi1, self.B_r_CP1
#         )
#         self.J_P1 = lambda t, q: self.subsystem1.J_P(
#             t, q[: self._nq1], self.xi1, self.B_r_CP1
#         )
#         self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
#             t, q[: self._nq1], self.xi1, self.B_r_CP1
#         )
#         self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
#             t, q[: self._nq1], u[: self._nu1], self.xi1, self.B_r_CP1
#         )
#         self.v_P1_q = lambda t, q, u: self.subsystem1.v_P_q(
#             t, q[: self._nq1], u[: self._nu1], self.xi1, self.B_r_CP1
#         )
#         self.= lambda t, q, u: self.sybsystem1.B_omega_IB1(
#             t, q[: self_nq1], u[: self._nu1], self.xi1, self.A_IB1
#         )
#
#         self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
#             t, q[self._nq1 :], self.xi2, self.B_r_CP2
#         )
#         self.A_IB2 = lambda t,y: self.subsystem2.A_IB(
#             t,q[: self._nq2], self.xi2, self.B_r_CP2
#         )
#         self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
#             t, q[self._nq1 :], self.xi2, self.B_r_CP2
#         )
#         self.J_P2 = lambda t, q: self.subsystem2.J_P(
#             t, q[self._nq1 :], self.xi2, self.B_r_CP2
#         )
#         self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
#             t, q[self._nq1 :], self.xi2, self.B_r_CP2
#         )
#         self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
#             t, q[self._nq1 :], u[self._nu1 :], self.xi2, self.B_r_CP2
#         )
#         self.v_P2_q = lambda t, q, u: self.subsystem2.v_P_q(
#             t, q[self._nq1 :], u[self._nu1 :], self.xi2, self.B_r_CP2
#         )
#         self.= lambda t, q, u: self.sybsystem2.B_omega_IB2(
#             t, q[: self_nq2], u[: self._nu2], self.xi2, self.A_IB2
#         )
#
#         l0 = norm(self.r_OP2(self.t0, self.q0) - self.r_OP1(self.t0, self.q0))
#         assert (
#             l0 > IS_CLOSE_ATOL
#         ), "Initial distance of two-point interaction is close to zero."
#
#     # auxiliary functions
#     # calculate angle between two axes
#     def phi(self, t, q):
#         column1 = self.A_IB1[:, self.axis]
#         column2 = self.A_IB2[:, self.axis]
#         absolute_product = np.absolute(column1) * np.absolute(column2)
#         return self.np.arccos(np.cross(column1,column2)/absolute_product)
#
#     # def l_q(self, t, q):
#     #     r_OP1_q = self.r_OP1_q(t, q)
#     #     r_OP2_q = self.r_OP2_q(t, q)
#     #
#     #     n = self._n(t, q)
#     #     return np.hstack((-n @ r_OP1_q, n @ r_OP2_q))
#
#     def l_dot(self, t, q, u):
#         rotation1 = (self.A_IB1 * self.B_omega_IB1(t,q,u))[self.axis]
#         rotation2 = (self.A_IB2 * self.B_omega_IB2(t,q,u))[self.axis]
#         return rotation2 - rotation1
#         # return self._n(t, q) @ (self.v_P2(t, q, u) - self.v_P1(t, q, u))
#
#     # def l_dot_q(self, t, q, u):
#     #     n_q1, n_q2 = self._n_q(t, q)
#     #     n = self._n(t, q)
#     #     v_P1 = self.v_P1(t, q, u)
#     #     v_P2 = self.v_P2(t, q, u)
#     #     v_P1P2 = v_P2 - v_P1
#
#     #     nq1 = self._nq1
#     #     gamma_q = np.zeros(self._nq)
#     #     gamma_q[:nq1] = -n @ self.v_P1_q(t, q, u) + v_P1P2 @ n_q1
#     #     gamma_q[nq1:] = n @ self.v_P2_q(t, q, u) + v_P1P2 @ n_q2
#     #     return gamma_q
#
#     # def l_dot_u(self, t, q, u):
#     #     n = self._n(t, q)
#
#     #     nu1 = self._nu1
#     #     l_dot_u = np.zeros(self._nu)
#     #     l_dot_u[:nu1] = -n @ self.J_P1(t, q)
#     #     l_dot_u[nu1:] = n @ self.J_P2(t, q)
#     #     return l_dot_u
#
#     # def _n(self, t, q):
#     #     r_OP1 = self.r_OP1(t, q)
#     #     r_OP2 = self.r_OP2(t, q)
#     #     return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)
#
#     # def _n_q(self, t, q):
#     #     r_OP1_q = self.r_OP1_q(t, q)
#     #     r_OP2_q = self.r_OP2_q(t, q)
#
#     #     r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
#     #     g = norm(r_P1P2)
#     #     n = r_P1P2 / g
#     #     P = (np.eye(3) - np.outer(n, n)) / g
#     #     n_q1 = -P @ r_OP1_q
#     #     n_q2 = P @ r_OP2_q
#     #     return n_q1, n_q2
#
#     # def W_l(self, t, q):
#     #     n = self._n(t, q)
#     #     J_P1 = self.J_P1(t, q)
#     #     J_P2 = self.J_P2(t, q)
#     #     return np.concatenate([-J_P1.T @ n, J_P2.T @ n])
#
#     # def W_l_q(self, t, q):
#     #     nq1 = self._nq1
#     #     nu1 = self._nu1
#     #     n = self._n(t, q)
#     #     n_q1, n_q2 = self._n_q(t, q)
#     #     J_P1 = self.J_P1(t, q)
#     #     J_P2 = self.J_P2(t, q)
#     #     J_P1_q = self.J_P1_q(t, q)
#     #     J_P2_q = self.J_P2_q(t, q)
#
#     #     # dense blocks
#     #     W_q = np.zeros((self._nu, self._nq))
#     #     W_q[:nu1, :nq1] = -J_P1.T @ n_q1 + np.einsum("i,ijk->jk", -n, J_P1_q)
#     #     W_q[:nu1, nq1:] = -J_P1.T @ n_q2
#     #     W_q[nu1:, :nq1] = J_P2.T @ n_q1
#     #     W_q[nu1:, nq1:] = J_P2.T @ n_q2 + np.einsum("i,ijk->jk", n, J_P2_q)
#     #     return W_q
#
#     # def export(self, sol_i, **kwargs):
#     #     points = [
#     #         self.r_OP1(sol_i.t, sol_i.q[self.qDOF]),
#     #         self.r_OP2(sol_i.t, sol_i.q[self.qDOF]),
#     #     ]
#     #     cells = [(VTK_LINE, [0, 1])]
#     #
#     #     return points, cells, None, None

