import numpy as np

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

def _acceleration_one_body(i, positions, masses, G, softening):
    n = len(masses)
    ai = np.zeros(3, dtype=float)
    ri = positions[i]
    for j in range(n):
        if i != j:
            r_vec = positions[j] - ri
            dist2 = np.dot(r_vec, r_vec) + softening**2
            dist = np.sqrt(dist2)
            ai += G * masses[j] * r_vec / dist**3
    return ai

class NBodySimulator:
    def __init__(self, positions, velocities, masses, G=4*np.pi**2, softening=0.0, 
                 parallel=False, n_jobs=-1, prefer="processes"):
        # The test expects a ValueError if shapes are weird
        if np.array(positions).ndim != 2:
            raise ValueError("Positions must be a 2D array (N, 3)")
            
        self.positions = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        self.masses = np.array(masses, dtype=float)
        self.G = float(G)
        self.softening = float(softening)
        self.parallel = bool(parallel)
        self.n_jobs = int(n_jobs)
        self.prefer = prefer
        
        if self.positions.shape != self.velocities.shape:
            raise ValueError("Positions and velocities shape mismatch")

        self.accelerations = self.compute_accelerations(self.positions)

    def compute_accelerations(self, positions=None):
        if positions is None: positions = self.positions
        n_bodies = len(self.masses)
        if self.parallel and JOBLIB_AVAILABLE:
            results = Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
                delayed(_acceleration_one_body)(i, positions, self.masses, self.G, self.softening)
                for i in range(n_bodies))
            return np.array(results)
        return np.array([_acceleration_one_body(i, positions, self.masses, self.G, self.softening) 
                         for i in range(n_bodies)])

    def step(self, dt):
        """Perform a single Leapfrog integration step (Required by tests)."""
        self.positions += self.velocities * dt + 0.5 * self.accelerations * dt**2
        new_acc = self.compute_accelerations(self.positions)
        self.velocities += 0.5 * (self.accelerations + new_acc) * dt
        self.accelerations = new_acc

    def total_energy(self):
        kinetic = 0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1))
        potential = 0.0
        for i in range(len(self.masses)):
            for j in range(i + 1, len(self.masses)):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                potential -= self.G * self.masses[i] * self.masses[j] / (r + self.softening)
        return kinetic + potential

    def total_momentum(self):
        return np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0)

    def center_of_mass(self):
        return np.sum(self.masses[:, np.newaxis] * self.positions, axis=0) / np.sum(self.masses)

    def simulate(self, num_steps, dt, store_velocities=True):
        """Runs the simulation and returns history (Updated for test compatibility)."""
        pos_hist = [self.positions.copy()]
        vel_hist = [self.velocities.copy()]
        energy_hist = [self.total_energy()]
        mom_hist = [self.total_momentum()]
        com_hist = [self.center_of_mass()]
        
        for _ in range(num_steps):
            self.step(dt)
            pos_hist.append(self.positions.copy())
            vel_hist.append(self.velocities.copy())
            energy_hist.append(self.total_energy())
            mom_hist.append(self.total_momentum())
            com_hist.append(self.center_of_mass())
            
        results = {
            "positions": np.array(pos_hist),
            "energies": np.array(energy_hist),
            "momentum": np.array(mom_hist),
            "center_of_mass": np.array(com_hist)
        }
        if store_velocities:
            results["velocities"] = np.array(vel_hist)
            
        return results

    def animate(self, trajectory, interval=20):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from IPython.display import HTML
        
        num_steps, n_bodies, _ = trajectory.shape
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        limit = np.max(np.abs(trajectory)) * 1.1
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
        
        lines = [ax.plot([], [], [], '-', alpha=0.5)[0] for _ in range(n_bodies)]
        points = [ax.plot([], [], [], 'o')[0] for _ in range(n_bodies)]

        def update(frame):
            for i in range(n_bodies):
                lines[i].set_data(trajectory[:frame, i, 0], trajectory[:frame, i, 1])
                lines[i].set_3d_properties(trajectory[:frame, i, 2])
                points[i].set_data([trajectory[frame, i, 0]], [trajectory[frame, i, 1]])
                points[i].set_3d_properties([trajectory[frame, i, 2]])
            return lines + points

        ani = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=True)
        plt.close()
        return HTML(ani.to_jshtml())