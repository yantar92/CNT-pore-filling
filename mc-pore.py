"""
Metropolis Monte Carlo Simulation for Hard Carbon Pore Filling.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time

class HardCarbonPoreModel:
    def __init__(
            self,
            pore_radius_angstrom=7.0,
            # 3.72A experimental
            # 3.59346 optB88-vdW from our data
            na_bond_length_angstrom=3.59346,
            grid_padding_angstrom=10.0,
            defect_probability=0.058,
            # Interaction Energies (eV)
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            temperature_k=298.0,
            voltage=1.0):  # voltage relative to bulk Na
        """
        Initialize 2D Triangular Lattice Model with Metropolis Dynamics.
        """
        # 1. Geometry Constants
        self.bond_length = na_bond_length_angstrom
        self.pore_radius = pore_radius_angstrom
        self.defect_probability = defect_probability

        # 2. Lattice Units
        self.radius_lattice_units = pore_radius_angstrom / self.bond_length
        padding_lattice = int(grid_padding_angstrom / self.bond_length)
        pore_span_lattice = int(self.radius_lattice_units * 2)
        self.grid_width = pore_span_lattice + 2 * padding_lattice + 4

        # 3. Energetics & Thermodynamics
        self.energies = {
            'Na_Na': energy_na_na,
            'Na_C': energy_na_c,
            'Na_Defect': energy_na_defect
        }

        # Boltzmann constant in eV/K
        self.kB = 8.617333262e-5 
        self.T = temperature_k
        self.beta = 1.0 / (self.kB * self.T)
        self.voltage = voltage

        # 4. State Grid Constants
        self.EMPTY = 0
        self.NA = 1
        self.CARBON = 2
        self.DEFECT = 3
        
        # Grid initialization
        self.grid = np.zeros((self.grid_width, self.grid_width), dtype=int)

        self._initialize_circular_pore()
        
        # 5. Pre-calculate Valid and Surface Sites for Efficiency
        self.valid_sites = []   # List of (r, c) inside the pore
        self.surface_sites = [] # List of (r, c) adjacent to carbon
        self._classify_sites()

        # 6. Carbon energy map – precompute carbon interaction for each site
        self.carbon_energy_map = np.zeros((self.grid_width, self.grid_width), dtype=float)
        self.n_defects_carbon = 0  # number of defective carbon atoms on pore wall
        self._compute_carbon_energy_map()

        # 7. Calculate default probabilities
        if len(self.valid_sites) > 0:
            self.default_p_gcmc = len(self.surface_sites) / len(self.valid_sites)
        else:
            self.default_p_gcmc = 0.0

        print(f"Model Initialized: {self.grid_width}x{self.grid_width} Grid")
        print(f"  Temp: {self.T} K, Beta: {self.beta:.2f} eV^-1")
        print(f"  Voltage: {self.voltage} V, Chem. pot: {-self.voltage} eV")
        print(f"  Valid Sites: {len(self.valid_sites)}")
        print(f"  Surface Sites: {len(self.surface_sites)}")
        print(f"  Defects: {self.defect_probability:.3f}")
        print(f"  Carbon‑wall defects (atomic): {self.n_defects_carbon}")
        print(f"  Default P_GCMC: {self.default_p_gcmc:.4f}")

    @property
    def mu(self):
        """Return chemical potential according to voltage and Na energies.
        """
        # We assume 2D, that's why 3
        return -self.voltage + 3 * self.energies['Na_Na']

    def _initialize_circular_pore(self):
        """Creates a circular pore centered in the grid.
        RADIUS is the pore radius in lattice units.
        Sets grid cells outside the pore to CARBON (wall).
        Also generates a realistic carbon ring along the pore circumference
        with atomic defects, stored for energy calculations and visualization."""
        center_r = self.grid_width // 2
        center_c = self.grid_width // 2
        sqrt3_half = np.sqrt(3) / 2.0

        # --- 1. Mark wall sites (outside pore) as CARBON in the grid ---
        for r in range(self.grid_width):
            for c in range(self.grid_width):
                dx = (c - center_c) + 0.5 * ((r % 2) - (center_r % 2))
                dy = sqrt3_half * (r - center_r)
                dist = np.sqrt(dx**2 + dy**2)
                # Offset to avoid Na close to C
                if dist >= self.radius_lattice_units - 0.35:
                    self.grid[r, c] = self.CARBON

        # --- 2. Generate carbon ring (realistic spacing, defects) ---
        # Physical pore radius (Å) and lattice‑unit radius
        R = self.pore_radius  # Å

        # Carbon‑carbon bond length (Å)
        a_CC = 1.42
        circumference = 2.0 * np.pi * R
        N_carbons = int(np.round(circumference / a_CC))
        assert N_carbons > 1

        # Angular positions of carbon atoms (evenly spaced)
        carbon_angles = np.linspace(0.0, 2.0 * np.pi, N_carbons, endpoint=False)
        # Defect status (atomic concentration)
        self.n_defects_carbon = round(N_carbons * self.defect_probability)
        if self.n_defects_carbon > 0:
            defect_indices = np.random.choice(
                N_carbons, size=self.n_defects_carbon, replace=False)
            defect_mask = np.zeros(N_carbons, dtype=bool)
            defect_mask[defect_indices] = True
        else:
            defect_mask = np.zeros(N_carbons, dtype=bool)

        # This would be more suitable for ensemble, but let's not.
        # defect_mask = np.random.rand(N_carbons) < self.defect_probability
        # self.n_defects_carbon = np.sum(defect_mask)

        # Store carbon data for later use (energy map & visualization)
        self.carbon_angles = carbon_angles
        self.carbon_defect_mask = defect_mask
        # Positions in lattice units (scale by 1/bond_length)
        scale = 1.0 / self.bond_length
        self.carbon_positions_lattice = []
        for theta in carbon_angles:
            x_lat = (R * np.cos(theta)) * scale
            y_lat = (R * np.sin(theta)) * scale
            self.carbon_positions_lattice.append((x_lat, y_lat))

    def _classify_sites(self):
        """Identifies valid pore sites and surface sites based on distance from pore center."""
        center_r = self.grid_width // 2
        center_c = self.grid_width // 2
        sqrt3_half = np.sqrt(3) / 2.0

        # Surface threshold: sites within this distance (lattice units) from the pore wall
        surface_threshold = 1.0  # one Na‑bond length

        for r in range(self.grid_width):
            for c in range(self.grid_width):
                # Convert grid indices to lattice coordinates
                dx = (c - center_c) + 0.5 * ((r % 2) - (center_r % 2))
                dy = sqrt3_half * (r - center_r)
                dist = np.sqrt(dx**2 + dy**2)

                # Valid sites are inside the pore (not a wall)
                if dist <= self.radius_lattice_units:
                    self.valid_sites.append((r, c))
                    # Surface sites are those close to the wall
                    if dist > self.radius_lattice_units - surface_threshold:
                        self.surface_sites.append((r, c))

    def _compute_carbon_energy_map(self):
        """
        Pre‑compute the carbon‑interaction energy for each lattice site.
        Uses the carbon ring generated in _initialize_circular_pore.
        Each carbon atom is assigned to the nearest surface Na site (by angle).
        The carbon interaction energy for a Na site is the sum of contributions
        from its assigned carbon atoms, with defective carbons contributing a
        stronger binding energy.
        """
        if not self.surface_sites:
            return

        # Carbon data already generated
        carbon_angles = self.carbon_angles
        defect_mask = self.carbon_defect_mask
        N_carbons = len(carbon_angles)

        # Angular positions of surface Na sites
        na_angles = []
        na_sites = []
        for r, c in self.surface_sites:
            x, y = self.get_triangular_coordinates(r, c)
            angle = np.arctan2(y, x)  # range [-π, π]
            if angle < 0.0:
                angle += 2.0 * np.pi  # map to [0, 2π)
            na_angles.append(angle)
            na_sites.append((r, c))

        na_angles = np.array(na_angles)  # shape (N_na,)
        N_na = len(na_angles)

        # Compute pairwise circular angular distances
        # diff[i,j] = angular distance between na_angles[i] and carbon_angles[j]
        diff = np.abs(na_angles[:, None] - carbon_angles[None, :])  # shape (N_na, N_carbons)
        diff = np.minimum(diff, 2.0 * np.pi - diff)  # circular distance
        assigned_na_idx = np.argmin(diff, axis=0)  # shape (N_carbons,)

        # Count carbon atoms per Na site
        carbon_counts = np.bincount(assigned_na_idx, minlength=N_na)
        # Count defective carbons per Na site
        defect_counts = np.bincount(assigned_na_idx[defect_mask], minlength=N_na)

        print("Number of Na-C bonds on surface: ", carbon_counts)
        print("Number of adjacent Na-defect bonds", defect_counts)
        # Compute carbon interaction energy for each surface site
        for i, (r, c) in enumerate(na_sites):
            n_c = carbon_counts[i]
            n_def = defect_counts[i]
            if n_c == 0:
                energy = 0.0
            else:
                # Each normal carbon contributes energy_na_c, each
                # defective carbon energy_na_defect
                energy = (n_c - n_def) * self.energies['Na_C']
                energy += n_def * self.energies['Na_Defect']
            self.carbon_energy_map[r, c] = energy

    def get_neighbors(self, r, c, include_walls=False):
        """
        Returns list of neighbor coordinates for triangular lattice.
        If include_walls=True, returns all grid neighbors (including walls).
        If include_walls=False, returns only accessible neighbors (EMPTY or NA).
        """
        candidates = [
            (r, c - 1), (r, c + 1),
            (r - 1, c), (r + 1, c)
        ]
        if r % 2 == 0:  # Even rows
            candidates.extend([(r - 1, c - 1), (r + 1, c - 1)])
        else:          # Odd rows
            candidates.extend([(r - 1, c + 1), (r + 1, c + 1)])

        valid_neighbors = []
        for nr, nc in candidates:
            if 0 <= nr < self.grid_width and 0 <= nc < self.grid_width:
                if include_walls:
                    valid_neighbors.append((nr, nc))
                else:
                    # Only return accessible sites (EMPTY or NA)
                    # Wall sites are CARBON or DEFECT
                    if self.grid[nr, nc] in (self.EMPTY, self.NA):
                        valid_neighbors.append((nr, nc))
        return valid_neighbors

    def _calculate_potential_energy_at_site(self, r, c, ignore_neighbor=None):
        """
        Calculates the potential energy of a Sodium atom if it were placed at (r,c).
        This sums interactions with existing Na neighbors and adds the
        pre-computed carbon interaction energy for this site.
        """
        e_sum = self.carbon_energy_map[r, c]  # contribution from carbon wall

        # Add Na‑Na interactions from neighbor Na atoms
        neighbors = self.get_neighbors(r, c, include_walls=True)
        for nr, nc in neighbors:
            if (nr, nc) == ignore_neighbor:
                continue
            if self.grid[nr, nc] == self.NA:
                e_sum += self.energies['Na_Na']
            # CARBON and DEFECT neighbors are already accounted for in
            # carbon_energy_map
        return e_sum

    def calculate_swap_energy(self, r1, c1, r2, c2):
        """Delta E for moving particle from (r1, c1) to (r2, c2)."""
        assert self.grid[r2, c2] == self.EMPTY
        # Energy cost to remove from r1, c1
        energy_removal = -self._calculate_potential_energy_at_site(r1, c1)
        # Energy gain to add to r2, c2 (ignoring the particle
        # currently at r1, c1)
        energy_addition = self._calculate_potential_energy_at_site(
            r2, c2, ignore_neighbor=(r1, c1))
        return energy_removal + energy_addition

    def attempt_diffusion(self):
        """Attempts to move a particle to an empty neighbor."""
        # Pick a random valid site to maintain detailed balance
        # relative to area.
        r, c = random.choice(self.valid_sites)

        # Only proceed if there is a particle to move
        if self.grid[r, c] != self.NA:
            return False

        # Find empty valid neighbors
        neighbors = self.get_neighbors(r, c)  # Returns non-carbon neighbors
        empty_neighbors = [n for n in neighbors if self.grid[n] == self.EMPTY]

        if not empty_neighbors:
            return False

        nr, nc = random.choice(empty_neighbors)

        # Calculate Delta E
        dE = self.calculate_swap_energy(r, c, nr, nc)

        # Metropolis Acceptance
        if dE <= 0 or np.random.random() < np.exp(-dE * self.beta):
            assert self.grid[r, c] == self.NA
            assert self.grid[nr, nc] == self.EMPTY
            self.grid[r, c] = self.EMPTY
            self.grid[nr, nc] = self.NA
            return True
        return False

    def attempt_gcmc(self):
        """Attempts to Insert or Remove a particle at a surface site."""
        if not self.surface_sites:
            return False

        r, c = random.choice(self.surface_sites)

        if self.grid[r, c] == self.EMPTY:
            # --- INSERTION ---
            # Delta E = E_interaction - mu
            interaction = self._calculate_potential_energy_at_site(r, c)
            dE = interaction - self.mu

            if dE <= 0 or np.random.random() < np.exp(-dE * self.beta):
                self.grid[r, c] = self.NA
                return True
        elif self.grid[r, c] == self.NA:
            # --- REMOVAL ---
            # Reverse of insertion: Delta E = -(E_interaction - mu)
            interaction = self._calculate_potential_energy_at_site(r, c)
            dE = -(interaction - self.mu)

            if dE <= 0 or np.random.random() < np.exp(-dE * self.beta):
                self.grid[r, c] = self.EMPTY
                return True
        return False

    def get_filling_fraction(self):
        total_valid = len(self.valid_sites)
        filled = np.sum(self.grid == self.NA)
        return filled / total_valid

    def run_step(self, p_gcmc=None):
        """
        Executes one Monte Carlo Step (MCS).
        Traditionally 1 MCS = N_sites attempts.
        We perform logic for a single event here.

        p_gcmc: Probability of attempting a GCMC move vs Diffusion move.
                If None, defaults to ratio of surface sites to valid sites.
        """
        prob = p_gcmc if p_gcmc is not None else self.default_p_gcmc

        if np.random.random() < prob:
            self.attempt_gcmc()
        else:
            self.attempt_diffusion()

    def get_triangular_coordinates(self, r, c):
        """Convert grid indices to triangular lattice Cartesian coordinates."""
        center_r = self.grid_width // 2
        center_c = self.grid_width // 2
        sqrt3_half = np.sqrt(3) / 2.0
        
        x = (c - center_c) + 0.5 * ((r % 2) - (center_r % 2))
        y = sqrt3_half * (r - center_r)
        return x, y

# --- Simulation & Visualization Wrapper ---

def run_simulation(voltage=None, steps=None, temp=None):
    # Simulation Parameters
    MC_STEPS = 20000 if steps is None else steps  # Total normalized steps (attempts per site)
    SNAPSHOT_INTERVAL = 400

    # Initialize Model
    model = HardCarbonPoreModel(
        pore_radius_angstrom=20.0,
        temperature_k=298 if temp is None else temp,
        voltage=1.0 if voltage is None else voltage,
        defect_probability=0.058,
        # defect_probability=0,
        energy_na_na=-0.35,
        energy_na_c=-0.32,
        energy_na_defect=-1.77,
    )

    total_sites = len(model.valid_sites)
    total_attempts = MC_STEPS * total_sites

    # Tracking
    filling_history = []
    time_points = []

    print(f"Starting Simulation: {total_attempts} attempts ({MC_STEPS} MCS)...")
    start_time = time.time()

    # Visualization Setup
    fig, (ax_grid, ax_stats) = plt.subplots(1, 2, figsize=(12, 6))
    plt.show(block=False)

    for attempt in range(total_attempts):
        # Use default p_gcmc calculated by the model
        model.run_step()

        # Record stats every 0.5 MCS (approx)
        if attempt % (total_sites // 2) == 0:
            filling_history.append(model.get_filling_fraction())
            time_points.append(attempt / total_sites)

        # Snapshots
        current_mcs = attempt / total_sites
        if attempt % (SNAPSHOT_INTERVAL * total_sites) == 0 or attempt == total_attempts - 1:
            print(f"  Step {int(current_mcs)}/{MC_STEPS}: Filling = {filling_history[-1]:.2%}")

            # Update Grid Plot with triangular lattice visualization
            ax_grid.clear()
            
            # Collect coordinates and colors for different site types
            empty_x, empty_y = [], []
            na_x, na_y = [], []
            carbon_x, carbon_y = [], []
            defect_x, defect_y = [], []
            
            # We'll visualize all sites within 1.5 times pore radius to see some walls
            max_vis_radius = model.radius_lattice_units * 1.5
            
            # Na lattice sites (empty / occupied)
            for r in range(model.grid_width):
                for c in range(model.grid_width):
                    x, y = model.get_triangular_coordinates(r, c)
                    dist = np.sqrt(x**2 + y**2)
                    
                    # Only plot sites within visualization radius
                    if dist > max_vis_radius:
                        continue
                    
                    site_type = model.grid[r, c]
                    if site_type == model.EMPTY:
                        empty_x.append(x)
                        empty_y.append(y)
                    elif site_type == model.NA:
                        na_x.append(x)
                        na_y.append(y)
                    # CARBON and DEFECT grid cells are not plotted (they are only
                    # coarse wall markers). Real carbon ring is plotted separately.
            
            # Carbon ring (from carbon‑wall model)
            for (x_lat, y_lat), is_defect in zip(model.carbon_positions_lattice,
                                                 model.carbon_defect_mask):
                # Carbon positions are already in lattice units
                # Optionally, skip if outside visualization radius (should not happen)
                dist = np.sqrt(x_lat**2 + y_lat**2)
                if dist > max_vis_radius:
                    continue
                if is_defect:
                    defect_x.append(x_lat)
                    defect_y.append(y_lat)
                else:
                    carbon_x.append(x_lat)
                    carbon_y.append(y_lat)
            
            # Plot sites with different markers/colors
            # Scale marker size based on lattice spacing
            marker_scale = 100.0 / (model.grid_width / 8)  # Adjust scaling
            
            if empty_x:
                ax_grid.scatter(empty_x, empty_y, s=marker_scale, c='lightblue', 
                               edgecolors='gray', linewidths=0.5, alpha=0.7, label='Empty')
            if na_x:
                ax_grid.scatter(na_x, na_y, s=marker_scale, c='red', 
                               edgecolors='darkred', linewidths=0.5, alpha=0.9, label='Na')
            if carbon_x:
                ax_grid.scatter(carbon_x, carbon_y, s=marker_scale, c='black', 
                               edgecolors='gray', linewidths=0.5, alpha=0.5, label='Carbon')
            if defect_x:
                ax_grid.scatter(defect_x, defect_y, s=marker_scale, c='orange', 
                               edgecolors='darkorange', linewidths=0.5, alpha=0.8, label='Defect')
            
            # Draw pore boundary circle
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = model.radius_lattice_units * np.cos(theta)
            circle_y = model.radius_lattice_units * np.sin(theta)
            ax_grid.plot(circle_x, circle_y, 'k--', linewidth=1, alpha=0.7, label='Pore Boundary')
            
            # Set equal aspect ratio and limits
            ax_grid.set_aspect('equal')
            ax_grid.set_xlim(-max_vis_radius, max_vis_radius)
            ax_grid.set_ylim(-max_vis_radius, max_vis_radius)
            ax_grid.set_title(f"Pore State (MCS: {int(current_mcs)})")
            ax_grid.legend(loc='upper right', fontsize='small')
            # ax_grid.grid(True, alpha=0.3)
            ax_grid.grid(False)

            # Update Stats Plot
            ax_stats.clear()
            ax_stats.plot(time_points, filling_history, label='Filling Fraction')
            ax_stats.set_ylim(0, 1.0)
            ax_stats.set_xlabel('Monte Carlo Steps')
            ax_stats.set_ylabel('Filling %')
            ax_stats.set_title(f"Filling Kinetics (P_GCMC={model.default_p_gcmc:.2f})")
            ax_stats.grid(True)
            
            # Add simulation parameters as text
            param_text = (f"T = {model.T} K\n"
                          f"V = {model.voltage:.2f} V\n"
                          f"R = {model.pore_radius} Å\n"
                          f"defects = {model.defect_probability:.3f}\n"
                          f"E_Na-Na = {model.energies['Na_Na']:.3f} eV\n"
                          f"E_Na-C = {model.energies['Na_C']:.3f} eV\n"
                          f"E_Na-def = {model.energies['Na_Defect']:.3f} eV")
            ax_stats.text(0.02, 0.98, param_text, transform=ax_stats.transAxes,
                         fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.draw()
            plt.pause(0.01)

    print(f"Simulation Complete in {time.time() - start_time:.2f}s")
    plt.show()

if __name__ == "__main__":
    run_simulation()
