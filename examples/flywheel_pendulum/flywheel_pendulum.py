import numpy as np
from os import path
from scipy.integrate import odeint
from matplotlib import pyplot as plt

from cardillo import System
from cardillo.math import axis_angle2quat, cross3
from cardillo.discrete import RigidBody, Meshed
from cardillo.solver import ScipyIVP, ScipyDAE
from cardillo.constraints import Revolute
from cardillo.forces import Force


class GearTransmission:
    def __init__(
        self,
        subsystem1: Revolute,
        subsystem2: Revolute,
        radius1,
        radius2,
        reversed=False,
    ):
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.radius1 = radius1
        self.radius2 = radius2 if not reversed else -radius2
        self.nla_g = 1

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        self._nq1 = self.subsystem1._nq
        self._nq2 = self.subsystem2._nq
        self._nq = self._nq1 + self._nq2
        self.qDOF = np.concatenate((qDOF1, qDOF2))

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        self._nu1 = self.subsystem1._nu
        self._nu2 = self.subsystem2._nu
        self._nu = self._nu1 + self._nu2
        self.uDOF = np.concatenate((uDOF1, uDOF2))

    def g(self, t, q):
        nq1 = self._nq1
        return (
            self.subsystem1.l(t, q[:nq1]) - self.subsystem1.angle0
        ) * self.radius1 - (
            self.subsystem2.l(t, q[nq1:]) - self.subsystem2.angle0
        ) * self.radius2

    def g_q(self, t, q):
        nq1 = self._nq1
        g_q = np.zeros(self._nq, dtype=q.dtype)
        g_q[:nq1] = self.subsystem1.l_q(t, q[:nq1]) * self.radius1
        g_q[nq1:] = -self.subsystem2.l_q(t, q[nq1:]) * self.radius2
        return g_q

    def g_dot(self, t, q, u):
        nq1 = self._nq1
        nu1 = self._nu1
        return (
            self.subsystem1.l_dot(t, q[:nq1], u[:nu1]) * self.radius1
            - self.subsystem2.l_dot(t, q[nq1:], u[nu1:]) * self.radius2
        )

    def g_dot_q(self, t, q, u):
        nq1 = self._nq1
        nu1 = self._nu1
        g_dot_q = np.zeros(self._nq, dtype=q.dtype)
        g_dot_q[:nq1] = self.subsystem1.l_dot_q(t, q[:nq1], u[:nu1]) * self.radius1
        g_dot_q[nq1:] = -self.subsystem2.l_dot_q(t, q[nq1:], u[nu1:]) * self.radius2
        return g_dot_q

    def g_dot_u(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        g_dot_u = np.zeros(self._nu, dtype=q.dtype)
        g_dot_u[:nu1] = self.subsystem1.l_dot_u(t, q[:nq1], None) * self.radius1
        g_dot_u[nu1:] = -self.subsystem2.l_dot_u(t, q[nq1:], None) * self.radius2
        return g_dot_u

    def g_ddot(self, t, q, u, u_dot):
        nq1 = self._nq1
        nu1 = self._nu1
        e_c1_1 = self.subsystem1.A_IJ1(t, q[:nq1])[:, self.subsystem1.axis]
        e_c1_2 = self.subsystem2.A_IJ1(t, q[nq1:])[:, self.subsystem2.axis]
        g_ddot = (
            self.subsystem1.Psi2(t, q[:nq1], u[:nu1], u_dot[:nu1])
            - self.subsystem1.Psi1(t, q[:nq1], u[:nu1], u_dot[:nu1])
        ) @ e_c1_1 * self.radius1 - (
            self.subsystem2.Psi2(t, q[nq1:], u[nu1:], u_dot[nu1:])
            - self.subsystem2.Psi1(t, q[nq1:], u[nu1:], u_dot[nu1:])
        ) @ e_c1_2 * self.radius2
        return g_ddot

    # def g_ddot_q(self, t, q, u, u_dot):
    #     return

    # def g_ddot_u(self, t, q, u, u_dot):
    #     return

    def W_g(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        W_g = np.zeros((self._nu, 1), dtype=q.dtype)
        W_g[:nu1] = self.subsystem1.W_l(t, q[:nq1]) * self.radius1
        W_g[nu1:] = -self.subsystem2.W_l(t, q[nq1:]) * self.radius2
        return W_g

    def Wla_g_q(self, t, q, la_g):
        nq1 = self._nq1
        nu1 = self._nu1
        nq2 = self._nq2
        nu2 = self._nu2
        Wla_g_q = np.zeros((self._nu, self._nq), dtype=q.dtype)
        Wla_g_q[:nu1, :nq1] = (
            self.subsystem1.W_l_q(t, q[:nq1]).reshape((nu1, nq1)) * self.radius1 * la_g
        )
        Wla_g_q[nu1:, nq1:] = (
            -self.subsystem2.W_l_q(t, q[nq1:]).reshape((nu2, nq2)) * self.radius2 * la_g
        )
        return Wla_g_q


# parameter
m1 = (445 + 124) * 1e-3
theta_A = (29644.7665 * 445 / 409.3892 + 190.6349 * 124 / 78.5229) * 1e-7

m2 = (64 + 45) * 1e-3
theta_B = (27.0383 * 64 / 12.0674 + 181) * 1e-7

m3 = 537 * 1e-3
theta_C = (17462.29 * 537 / 458.7439) * 1e-7

l1 = 5.04e-3
l2 = 70e-3
l3 = 150e-3

radius_B = 13e-3
radius_C = 9e-3
eta = radius_B / radius_C

z_offset = -10e-3

g = 9.81

kp, kd = np.array([2.5261, 0.0032])*0, np.array([0.3712, 0.0070])*0

tend = 5
dt = 1e-3
alpha0 = np.deg2rad(170)

M = np.array(
    [
        [
            m1 * l1**2 + theta_A + m2 * l2**2 + theta_B + m3 * l3**2 + theta_C,
            theta_B + eta * theta_C,
        ],
        [theta_B + eta * theta_C, theta_B + eta**2 * theta_C],
    ]
)

c0 = (-m1 * l1 + m2 * l2 - m3 * l3) * g

# build system
system = System()

# pendulum
r_OS = np.array([l1 * np.sin(alpha0), -l1 * np.cos(alpha0), z_offset])
phi = alpha0
p = axis_angle2quat(np.array([0, 0, 1]), phi)
omega = np.array([0, 0, 0])
v_S = cross3(omega, r_OS)
q0_pendulum = np.concatenate([r_OS, p])
u0 = np.concatenate([v_S, omega])
pendulum = Meshed(RigidBody)(
    path.join(path.dirname(__file__), "stl", "Pendulum.STL"),
    scale=1e-3,
    B_r_CP=np.array([0, l1, 0]),
    mass=m1,
    B_Theta_C=np.diag([1, 1, theta_A]),
    q0=q0_pendulum,
    name="pendulum",
)
system.add(pendulum)

# rotor
r_OS = np.array([-l2 * np.sin(alpha0), l2 * np.cos(alpha0), z_offset])
phi = alpha0
p = axis_angle2quat(np.array([0, 0, 1]), phi)
omega = np.array([0, 0, 0])
v_S = cross3(omega, r_OS)
q0_rotor = np.concatenate([r_OS, p])
u0 = np.concatenate([v_S, omega])
rotor = Meshed(RigidBody)(
    path.join(path.dirname(__file__), "stl", "Rotor.STL"),
    scale=1e-3,
    B_r_CP=np.array([0, -l2, 0]),
    mass=m2,
    B_Theta_C=np.diag([1, 1, theta_B]),
    q0=q0_rotor,
    name="rotor",
)
system.add(rotor)

# flywheel
r_OS = np.array([l3 * np.sin(alpha0), -l3 * np.cos(alpha0), z_offset])
phi = alpha0
p = axis_angle2quat(np.array([0, 0, 1]), phi)
omega = np.array([0, 0, 0])
v_S = cross3(omega, r_OS)
q0_flywheel = np.concatenate([r_OS, p])
u0 = np.concatenate([v_S, omega])
flywheel = Meshed(RigidBody)(
    path.join(path.dirname(__file__), "stl", "Flywheel.STL"),
    scale=1e-3,
    B_r_CP=np.array([0, l3, 0]),
    mass=m3,
    B_Theta_C=np.diag([1, 1, theta_C]),
    q0=q0_flywheel,
    name="flywheel",
)

system.add(flywheel)

rj1 = Revolute(system.origin, pendulum, axis=2, angle0=alpha0)
system.add(rj1)

rj2 = Revolute(pendulum, rotor, axis=2, r_OJ0=q0_rotor[:3], angle0=0)
system.add(rj2)

rj3 = Revolute(pendulum, flywheel, axis=2, r_OJ0=q0_flywheel[:3], angle0=0)
system.add(rj3)

gear = GearTransmission(rj2, rj3, radius_B, radius_C)
system.add(gear)

for body in [pendulum, rotor, flywheel]:
    system.add(Force(np.array([0, -body.mass * g, 0]), body))

system.assemble()

solver = ScipyDAE(system, tend, dt, rtol=1.0e-6, atol=1.0e-9)
sol = solver.solve()

rj1.previous_quadrant = 1
rj1.n_full_rotations = 0
rj2.previous_quadrant = 1
rj2.n_full_rotations = 0
angle = np.zeros((len(sol.t), 2), sol.q.dtype)
for i, (ti, qi) in enumerate(zip(sol.t, sol.q)):
    angle[i] = rj1.angle(ti, qi[rj1.qDOF]), rj2.angle(ti, qi[rj2.qDOF])
g = np.array([gear.g(ti, qi[gear.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])


# analytical solution
def func(t, x):
    tau = 0
    return np.array([*x[2:], *np.linalg.solve(M, [c0 * np.sin(x[0]), -tau])])


t_ref = np.linspace(0, tend, int(tend / dt))
y0 = np.array([alpha0, 0, 0, 0])

from scipy.integrate import solve_ivp
sol_ref = solve_ivp(func, (0, tend), y0, t_eval=t_ref, method='RK45', max_step=dt)

sol_ref2 = solve_ivp(func, (0, tend), y0, t_eval=t_ref, method='Radau', max_step=dt)
# sol_ref = odeint(func=func, y0=y0, t=t_ref)
y_ref = sol_ref.y
y_ref2 = sol_ref2.y

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_ref, y_ref[0, :] * 180 / np.pi, "r", label="RK45")
plt.plot(t_ref, y_ref2[0, :] * 180 / np.pi, "b.", label="Radau")
plt.plot(sol.t, angle[:, 0] * 180 / np.pi, "--", label="cardillo")
plt.ylabel("alpha")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t_ref, y_ref[1, :] * 180 / np.pi, "r")
plt.plot(t_ref, y_ref2[1, :] * 180 / np.pi, "b.")
plt.plot(sol.t, angle[:, 1] * 180 / np.pi, "--")
plt.ylabel("beta")
plt.xlabel("time")
plt.show()
