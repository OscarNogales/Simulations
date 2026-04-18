import numpy as np
import pytest
from nbody import *


def test_two_body_acceleration():
    G = 4 * np.pi**2
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    velocities = np.zeros((2, 3))
    masses = np.array([1.0, 1.0])

    sim = NBodySimulator(positions, velocities, masses, G=G, parallel=False)
    acc = sim.compute_accelerations()

    expected = np.array([
        [G, 0.0, 0.0],
        [-G, 0.0, 0.0]
    ])

    assert np.allclose(acc, expected, rtol=1e-12, atol=1e-12)


def test_center_of_mass():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    velocities = np.zeros((2, 3))
    masses = np.array([1.0, 3.0])

    sim = NBodySimulator(positions, velocities, masses)
    com = sim.center_of_mass()

    expected = np.array([1.5, 0.0, 0.0])
    assert np.allclose(com, expected)


def test_total_momentum():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [1.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0]
    ])
    masses = np.array([1.0, 2.0])

    sim = NBodySimulator(positions, velocities, masses)
    p = sim.total_momentum()

    expected = np.array([0.0, 0.0, 0.0])
    assert np.allclose(p, expected)


def test_step_preserves_shapes():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2*np.pi, 0.0]
    ])
    masses = np.array([1.0, 3e-6])

    sim = NBodySimulator(positions, velocities, masses)
    sim.step(0.001)

    assert sim.positions.shape == (2, 3)
    assert sim.velocities.shape == (2, 3)
    assert sim.accelerations.shape == (2, 3)


def test_simulate_output_shapes():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2*np.pi, 0.0]
    ])
    masses = np.array([1.0, 3e-6])

    sim = NBodySimulator(positions, velocities, masses)
    results = sim.simulate(num_steps=100, dt=0.001, store_velocities=True)

    assert results["positions"].shape == (100, 2, 3)
    assert results["velocities"].shape == (100, 2, 3)
    assert results["energies"].shape == (100,)
    assert results["momentum"].shape == (100, 3)
    assert results["center_of_mass"].shape == (100, 3)
    assert results["angular_momentum"].shape == (100, 3)


def test_energy_is_approximately_conserved_in_two_body_orbit():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2*np.pi, 0.0]
    ])
    masses = np.array([1.0, 3e-6])

    sim = NBodySimulator(positions, velocities, masses)
    results = sim.simulate(num_steps=5000, dt=1e-4)

    energy = results["energies"]
    relative_error = np.max(np.abs((energy - energy[0]) / energy[0]))

    assert relative_error < 1e-4


def test_momentum_is_approximately_conserved():
    positions = np.array([
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    masses = np.array([1.0, 1.0])

    sim = NBodySimulator(positions, velocities, masses)
    results = sim.simulate(num_steps=2000, dt=1e-4)

    momentum = results["momentum"]
    drift = np.max(np.linalg.norm(momentum - momentum[0], axis=1))

    assert drift < 1e-8


def test_center_of_mass_remains_nearly_fixed_when_initial_momentum_is_zero():
    positions = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    masses = np.array([1.0, 1.0])

    sim = NBodySimulator(positions, velocities, masses)
    results = sim.simulate(num_steps=2000, dt=1e-4)

    com = results["center_of_mass"]
    drift = np.max(np.linalg.norm(com - com[0], axis=1))

    assert drift < 1e-8


def test_invalid_shapes_raise_error():
    positions = np.array([0.0, 0.0, 0.0])
    velocities = np.array([[0.0, 0.0, 0.0]])
    masses = np.array([1.0])

    with pytest.raises(ValueError):
        NBodySimulator(positions, velocities, masses)