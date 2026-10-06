r"""
Moving meshes and the geometric conservation law
================================================

A space map that moves in time makes the mass matrix a function of time and
adds a Lie derivative term to the equation of every form. This example marches
a one-dimensional element whose length changes and shows that a density which
the mesh merely carries along does not move in the reference frame, which is
the discrete geometric conservation law.
"""  # noqa: D205 D400

# %%
#
# The discretization is a single element on the reference interval
# :math:`[-1, 1]`, mapped affinely onto a physical interval whose half length
# varies in time. The density is the top form of the domain.
#

import matplotlib.pyplot as plt
import numpy as np
from fdg import (
    BasisSpecs,
    BasisType,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    MovingMesh,
    advection_operator,
    lie_derivative_operator,
    march,
    space_maps_from_geometry_dofs,
    stage_mass,
)
from numpy.polynomial import legendre as L

ORDER_BASIS = 8
ORDER_INTEGRATION = 2 * ORDER_BASIS

base_space = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, ORDER_BASIS))
geometry_space = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, ORDER_BASIS))
integration = IntegrationSpace(
    IntegrationSpecs(ORDER_INTEGRATION, IntegrationMethod.GAUSS)
)
reference = np.linspace(-1.0, 1.0, ORDER_BASIS + 1)
specs_density = KFormSpecs(1, base_space)
nodes = np.ascontiguousarray(integration.nodes()[0], np.double)

AMPLITUDE = 0.3
OMEGA = 1.0
SPEED = 0.7


def half_length(t: float) -> float:
    """Return the half length of the physical element at time ``t``."""
    return 1.0 + AMPLITUDE * np.sin(OMEGA * t)


def geometry_dofs(t: float) -> np.ndarray:
    """Return the geometry degrees of freedom of the element at time ``t``."""
    return (half_length(t) * reference).reshape(1, 1, -1)


def density_dofs(values) -> np.ndarray:
    """Return the modal degrees of freedom of a top form."""
    vandermonde = L.legvander(nodes, ORDER_BASIS - 1)
    scaling = (2.0 * np.arange(ORDER_BASIS) + 1.0) / 2.0
    return (vandermonde.T @ (integration.weights() * np.asarray(values))) * scaling


def stage_index(mesh: MovingMesh, t: float, n_steps: int, stages: int) -> tuple:
    """Return the step and stage index of a stage time of ``mesh``."""
    times = np.array(
        [
            [mesh.stage_time(step, stage) for stage in range(stages)]
            for step in range(n_steps)
        ]
    )
    index = np.unravel_index(np.argmin(np.abs(times - t)), times.shape)
    return int(index[0]), int(index[1])


# %%
#
# # The mesh velocity
# #
# # The velocity of a uniformly stretching element grows towards the ends of the
# # reference interval, and it is obtained as the exact derivative of the
# # geometry interpolant rather than as a finite difference.
#

STEP_SIZE = 0.01
N_STEPS = 200
STAGES = 2

mesh = MovingMesh(
    geometry_dofs,
    STEP_SIZE,
    N_STEPS,
    element_count=1,
    geometry_space=geometry_space,
    integration=integration,
    stages=STAGES,
)

velocity = mesh.velocity(0, 0)[0, 0]
exact_velocity = AMPLITUDE * OMEGA * np.cos(OMEGA * mesh.stage_time(0, 0)) * nodes

print("Mesh velocity of a stretching element")
print("--------------------------------------")
print(f"stage time                    : {mesh.stage_time(0, 0):.6f}")
print(f"velocity at the reference ends : {velocity[0]:+.6f}, {velocity[-1]:+.6f}")
print(
    f"analytic derivative           : {exact_velocity[0]:+.6f}, {exact_velocity[-1]:+.6f}"
)

fig, ax = plt.subplots()
ax.plot(nodes, velocity, label="mesh velocity")
ax.plot(nodes, exact_velocity, "--", label=r"$\partial_t x$")
ax.set(
    xlabel=r"$\xi$",
    ylabel=r"$w(\xi)$",
    title="Mesh velocity of a uniformly stretching element",
)
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()

# %%
#
# # The Lie derivative of the volume form
# #
# # The Lie derivative of a top form is the exterior derivative of its
# # contraction, which for a top form coincides with advection. That identity is
# # what turns the mesh correction into a relative velocity in the reference
# # frame, and it is exact here to the accuracy of the assembly.
#

print("\nLie derivative of the volume form")
print("---------------------------------")

smap = space_maps_from_geometry_dofs(geometry_space, integration, geometry_dofs(0.0))[0]
volume = density_dofs(np.full(nodes.size, half_length(0.0)))

for label, values in (
    ("constant velocity", np.full(nodes.size, SPEED)),
    ("linear in xi", 0.3 * nodes),
):
    field_velocity = np.ascontiguousarray(values.reshape(1, -1))
    lie = lie_derivative_operator(smap, specs_density, field_velocity)
    advection = advection_operator(smap, specs_density, field_velocity)
    print(
        f"{label:20s} : max |L_w vol| = {np.max(np.abs(lie @ volume)):.3e}"
        f"   max |advection - Lie| = {np.max(np.abs(advection - lie)):.3e}"
    )

print()
print("The Lie derivative equals advection exactly for a top form, and a")
print("constant velocity leaves the volume form invariant to the accuracy of")
print("the assembly.")

# %%
#
# # Free streaming
# #
# # A mesh that translates at a constant speed has a mesh velocity that is
# # constant along the element, and advection by it annihilates a top form with
# # a constant component. Marching such a density over the moving mass matrix
# # therefore leaves it unchanged, which is the discrete geometric conservation
# # law: a free stream is not polluted by the motion of the mesh.
#

print("\nFree streaming on a translating mesh")
print("------------------------------------")

translating = MovingMesh(
    lambda t: (SPEED * t + reference).reshape(1, 1, -1),
    STEP_SIZE,
    N_STEPS,
    element_count=1,
    geometry_space=geometry_space,
    integration=integration,
    stages=STAGES,
)


def streaming_residual(state: np.ndarray, t: float) -> np.ndarray:
    """Transport the density by the mesh velocity alone."""
    step, stage = stage_index(translating, t, N_STEPS, STAGES)
    element = translating.space_maps(step, stage)[0]
    field_velocity = translating.velocity(step, stage)[0]
    return advection_operator(element, specs_density, field_velocity) @ state


initial = density_dofs(np.ones(nodes.size))
streaming = march(
    streaming_residual,
    initial,
    STEP_SIZE,
    N_STEPS,
    stages=STAGES,
    tolerance=1e-13,
    mass=translating.mass_factory(specs_density),
)

deviation = np.abs(streaming.states - initial[None, :]).max(axis=1)
print(f"steps                         : {N_STEPS}")
print(f"maximum deviation             : {np.max(deviation):.3e}")
print(f"fixed-point iterations        : {sum(streaming.iterations)}")
print()
print("The density is preserved to the accuracy at which the advection operator")
print("assembles the contraction of a constant vector field, which is about")
print("1e-12 and does not improve with the polynomial order because it is a")
print("property of the contraction rather than of the quadrature.")

fig, ax = plt.subplots()
ax.plot(streaming.times, np.maximum(deviation, 1e-18))
ax.set_yscale("log")
ax.set(
    xlabel="$t$",
    ylabel=r"$\| \rho(t) - \rho(0) \|$",
    title="Free streaming on a translating mesh",
)
ax.grid()
fig.tight_layout()
plt.show()

# %%
#
# # The moving mass matrix
# #
# # The mass matrix is block diagonal over the elements and follows the element
# # length. That is the other half of what the motion of the mesh changes, and it
# # is what the marcher evaluates at every stage.
#

print("\nMass matrix of the moving element")
print("---------------------------------")

for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
    step = min(int(fraction * N_STEPS), N_STEPS - 1)
    time = mesh.stage_time(step, 0)
    mass = stage_mass(mesh.space_maps(step, 0)[0], specs_density)
    print(
        f"  t = {time:5.2f}   trace(M) = {np.trace(mass):8.6f}"
        f"   half length = {half_length(time):.6f}"
    )

fig, ax = plt.subplots()
times = np.array([mesh.stage_time(step, 0) for step in range(N_STEPS)])
traces = np.array(
    [
        np.trace(stage_mass(mesh.space_maps(step, 0)[0], specs_density))
        for step in range(N_STEPS)
    ]
)
ax.plot(times, 2.0 * traces, label=r"$\mathrm{tr}(M)$")
ax.plot(times, 4.0 * np.array([half_length(t) for t in times]), "--", label="reference")
ax.set(xlabel="$t$", ylabel=r"$\mathrm{tr}(M)$", title="Mass matrix of the element")
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()
