"""
Metropolis Monte Carlo Simulation for Hard Carbon Pore Filling.

Command line usage:
    python mc-pore.py [--voltage 0.1] [--radius 10.0] [--file snapshots.pkl]
        [--steps 20000] [--visualize] [--energy_na_defect -1.77]
        [--temp 298] [--defect_placement surface] [--defect_probability 0.174]
        [--csv] [--quiet] [--seed INT]
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle
import copy
import argparse
import sys

class HardCarbonPoreModel:
    def __init__(
            self,
            pore_radius_angstrom=7.0,
            # 3.72A experimental
            # 3.59346 optB88-vdW from our data
            na_bond_length_angstrom=3.59346,
            grid_padding_angstrom=10.0,
            defect_probability=0.058,
            defect_placement='surface',  # 'random' or 'surface'
            # Interaction Energies (eV)
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            temperature_k=298.0,
            voltage=1.0,  # voltage relative to bulk Na
            eq_window=10000,
            eq_slope_threshold=1e-8,
            eq_min_mcs=10000,
            quiet=False):
        """
        Initialize 2D Triangular Lattice Model with Metropolis Dynamics.

        defect_placement: 'random' (Bernoulli per wall site) or
                          'surface' (exact fraction of pore‑surface wall sites).
        """
        # 1. Geometry Constants
        self.bond_length = na_bond_length_angstrom
        self.pore_radius = pore_radius_angstrom
        self.defect_probability = defect_probability
        self.defect_placement = defect_placement

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

        self._initialize_circular_pore(self.radius_lattice_units)

        # 5. Pre-calculate Valid and Surface Sites for Efficiency
        self.valid_sites = []   # List of (r, c) inside the pore
        self.surface_sites = [] # List of (r, c) adjacent to carbon
        self._classify_sites()

        # 6. Calculate default probabilities
        if len(self.valid_sites) > 0:
            self.default_p_gcmc = len(self.surface_sites) / len(self.valid_sites)
        else:
            self.default_p_gcmc = 0.0

        # 7. History
        self.steps = 0
        self.mcs_fill = None
        self.time_points = [0.0]
        self.filling_history = [0.0]

        # 8. Equilibrium detection
        self.equilibrium_reached = False
        self.eq_window = eq_window  # number of samples for equilibrium check
        self.eq_slope_threshold = eq_slope_threshold  # slope per MCS threshold
        self.eq_min_mcs = eq_min_mcs  # minimum MCS before checking
        self.quiet = quiet

        if not self.quiet:
            print(f"Model Initialized: {self.grid_width}x{self.grid_width} Grid")
            print(f"  Temp: {self.T} K, Beta: {self.beta:.2f} eV^-1")
            print(f"  Voltage: {self.voltage} V, Chem. pot: {-self.voltage} eV")
            print(f"  Valid Sites: {len(self.valid_sites)}")
            print(f"  Surface Sites: {len(self.surface_sites)}")
            print(f"  Defects: {self.defect_probability:.3f} ({self.defect_placement})")
            n_defects = 0
            for r, c in self.adjacent_wall_sites:
                n_defects += 1 if self.grid[r, c] == self.DEFECT else 0
            print(f"  Surface Carbons: {len(self.surface_sites)} ({n_defects} defects)")
            print(f"  Default P_GCMC: {self.default_p_gcmc:.4f}")

    @property
    def mu(self):
        """Return chemical potential according to voltage and Na energies.
        """
        # We assume 2D, that's why 3
        return -self.voltage + 3 * self.energies['Na_Na']

    def _initialize_circular_pore(self, radius):
        """Creates a circular pore centered in the grid.
        RADIUS is the pore radius in lattice units."""
        center_r = self.grid_width // 2
        center_c = self.grid_width // 2
        sqrt3_half = np.sqrt(3) / 2.0  # constant for triangular lattice geometry

        # Precompute distances and identify wall sites
        distances = [[0.0 for _ in range(self.grid_width)] for _ in range(self.grid_width)]
        self.wall_sites = []

        for r in range(self.grid_width):
            for c in range(self.grid_width):
                dx = (c - center_c) + 0.5 * ((r % 2) - (center_r % 2))
                dy = sqrt3_half * (r - center_r)
                dist = np.sqrt(dx**2 + dy**2)
                distances[r][c] = dist

                if dist >= radius:
                    self.wall_sites.append((r, c))

        self.adjacent_wall_sites = []
        for r, c in self.wall_sites:
            neighbors = self.get_neighbors(r, c, include_walls=True)
            is_adjacent = False
            for nr, nc in neighbors:
                if distances[nr][nc] < radius:
                    is_adjacent = True
                    break
            if is_adjacent:
                self.adjacent_wall_sites.append((r, c))

        # Initialize all wall sites as carbon
        for r, c in self.wall_sites:
            self.grid[r, c] = self.CARBON

        # Apply defect placement according to mode
        if self.defect_placement == 'random':
            # Bernoulli per wall site
            for r, c in self.wall_sites:
                if np.random.random() < self.defect_probability:
                    self.grid[r, c] = self.DEFECT

        elif self.defect_placement == 'surface':
            # Exact fraction of pore‑surface wall sites
            if self.adjacent_wall_sites:
                k = int(round(self.defect_probability * len(self.adjacent_wall_sites)))
                if k <= 0:
                    defect_set = set()
                elif k == len(self.adjacent_wall_sites):
                    defect_set = set(self.adjacent_wall_sites)
                else:
                    defect_set = set(random.sample(self.adjacent_wall_sites, k))
                for r, c in defect_set:
                    self.grid[r, c] = self.DEFECT

        else:
            raise ValueError(f"Unknown defect_placement: {self.defect_placement}")

    def _classify_sites(self):
        """Identifies valid pore sites and surface sites (adjacent to walls)."""
        for r in range(self.grid_width):
            for c in range(self.grid_width):
                # Valid sites are those that are not walls (CARBON or DEFECT)
                # Initially everything else is EMPTY (0)
                if self.grid[r, c] in (self.EMPTY, self.NA):
                    self.valid_sites.append((r, c))

                    # Check if it's a surface site (neighbor is carbon/defect)
                    neighbors = self.get_neighbors(r, c, include_walls=True)
                    is_surface = False
                    for nr, nc in neighbors:
                        if self.grid[nr, nc] in (self.CARBON, self.DEFECT):
                            is_surface = True
                            break
                    if is_surface:
                        self.surface_sites.append((r, c))

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
        This sums interactions with existing neighbors.
        """
        e_sum = 0.0
        # Get all grid neighbors to check for Carbon/Defects
        neighbors = self.get_neighbors(r, c, include_walls=True)

        for nr, nc in neighbors:
            if (nr, nc) == ignore_neighbor:
                continue

            neighbor_type = self.grid[nr, nc]

            if neighbor_type == self.CARBON:
                e_sum += self.energies['Na_C']
            elif neighbor_type == self.DEFECT:
                e_sum += self.energies['Na_Defect']
            elif neighbor_type == self.NA:
                e_sum += self.energies['Na_Na']

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

    @property
    def mcs(self) -> float:
        """Number of normalized MC steps.
        """
        return self.steps / len(self.valid_sites)

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

        self.steps += 1
        # Record stats every 0.5 MCS (approx)
        if self.steps % (len(self.valid_sites) // 2) == 0:
            self.time_points.append(self.mcs)
            self.filling_history.append(self.get_filling_fraction())
            # Snapshot time of full pore filling
            if self.mcs_fill is None and self.mcs == 1:
                self.mcs_fill = self.mcs
        if self.steps % (len(self.valid_sites) * 10) == 0:
            self._check_equilibrium()

    def _check_equilibrium(self):
        """Check if filling fraction has stabilized."""
        if self.equilibrium_reached:
            return True
        if self.mcs < self.eq_min_mcs:
            return False
        if len(self.filling_history) < self.eq_window:
            return False
        y = np.array(self.filling_history[-self.eq_window:])
        x = np.array(self.time_points[-self.eq_window:])
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        if abs(slope) < self.eq_slope_threshold:
            self.equilibrium_reached = True
        return self.equilibrium_reached

    def get_triangular_coordinates(self, r, c):
        """Convert grid indices to triangular lattice Cartesian coordinates."""
        center_r = self.grid_width // 2
        center_c = self.grid_width // 2
        sqrt3_half = np.sqrt(3) / 2.0
        x = (c - center_c) + 0.5 * ((r % 2) - (center_r % 2))
        y = sqrt3_half * (r - center_r)
        return x, y

    def take_snapshot(self):
        """Return a deep copy of the current model state."""
        return copy.deepcopy(self)

# --- Simulation & Visualization Wrapper ---

def save_model_svg(model, filename, scale=80):
    """
    Save pore model atomic grid to FILENAME as svg.
    SCALE is number of pixels per lattice unit in the svg.
    """

    # 1. Collect elements to draw
    atoms = []  # List of (x, y, type)
    xs, ys = [], []

    # Visualization radius limit
    vis_limit_lattice = model.radius_lattice_units + 1.8

    for r in range(model.grid_width):
        for c in range(model.grid_width):
            site_type = model.grid[r, c]
            if site_type == model.EMPTY:
                continue

            x, y = model.get_triangular_coordinates(r, c)
            dist = np.sqrt(x**2 + y**2)

            if dist > vis_limit_lattice:
                continue

            xs.append(x)
            ys.append(y)

            # Determine Color/Style Category
            style_type = 'carbon'
            if site_type == model.DEFECT:
                style_type = 'carbon_defect'
            elif site_type == model.NA:
                style_type = 'na_bulk'

            atoms.append((x, y, style_type))

    if not atoms:
        print("Warning: No atoms to visualize.")
        return

    # 2. Calculate ViewBox
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 1.0
    width_u = (max_x - min_x) + 2 * pad
    height_u = (max_y - min_y) + 2 * pad
    width_px = width_u * scale
    height_px = height_u * scale

    # 3. Calculate Blur (Proportional to radius)
    # Ref: stdDev=0.92, radius=7.97 -> ratio ~0.115
    atom_radius_px = 0.4 * scale
    blur_std_dev = atom_radius_px * 0.115

    # 4. Generate SVG Content
    svg_lines = []
    svg_lines.append(f'<svg width="{width_px:.2f}" height="{height_px:.2f}" '
                     f'viewBox="0 0 {width_px:.2f} {height_px:.2f}" '
                     'xmlns="http://www.w3.org/2000/svg" '
                     'xmlns:xlink="http://www.w3.org/1999/xlink">')

    # 5. Definitions: Radial Gradients (3D Spheres) & Filter
    svg_lines.append("<defs>")

    # Filter for soft sphere edge (mimicking SVG Gaussian blur)
    svg_lines.append(f'''
    <filter id="atom_blur" x="-0.2" y="-0.2" width="1.4" height="1.4">
      <feGaussianBlur stdDeviation="{blur_std_dev:.4f}" />
    </filter>
    ''')

    fill_na="#c28d14"
    # Na Bulk Gradient (Gold) - Focal point offset for 3D look
    svg_lines.append('''
    <radialGradient id="grad_na_bulk" cx="50%" cy="50%" r="50%" fx="30%" fy="30%">
        <stop offset="0%" style="stop-color:#f4b31c;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#c28d14;stop-opacity:1" />
    </radialGradient>
    ''')

    fill_defect="#ff0000"
    # Na Defect Gradient (Red) - Focal point offset for 3D look
    svg_lines.append('''
    <radialGradient id="grad_carbon_defect" cx="50%" cy="50%" r="50%" fx="30%" fy="30%">
        <stop offset="0%" style="stop-color:#ff5555;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#ff0000;stop-opacity:1" />
    </radialGradient>
    ''')

    fill_carbon="#6a6a6a"
    # Carbon Gradient (Grey) - Focal point offset for 3D look
    svg_lines.append('''
    <radialGradient id="grad_carbon" cx="50%" cy="50%" r="50%" fx="30%" fy="30%">
        <stop offset="0%" style="stop-color:#999999;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#6a6a6a;stop-opacity:1" />
    </radialGradient>
    ''')

    svg_lines.append("</defs>")

    # Background (White)
    # svg_lines.append(f'<rect width="100%" height="100%" fill="white"/>')

    # 6. Draw Atoms
    atoms.sort(key=lambda a: a[1])

    for x, y, atype in atoms:
        px = (x - min_x + pad) * scale
        py = (max_y - y + pad) * scale  # Inverted Y for drawing

        # fill_val = "url(#grad_carbon)"  # Carbon default
        fill_val = fill_carbon
        if atype == 'na_bulk':
            # fill_val = "url(#grad_na_bulk)"
            fill_val = fill_na
        elif atype == 'carbon_defect':
            # fill_val = "url(#grad_carbon_defect)"
            fill_val = fill_defect

        # All atoms get the blur filter + radial gradient fill
        # svg_lines.append(
        #     f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{atom_radius_px:.2f}" '
        #     f'style="fill:{fill_val};stroke:none;filter:url(#atom_blur)" />')
        svg_lines.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{atom_radius_px:.2f}" '
            f'style="fill:{fill_val};stroke:none" />')

    svg_lines.append('</svg>')

    with open(filename, 'w') as f:
        f.write("\n".join(svg_lines))
    print(f"Saved visualization to {filename}")


def visualize_model(model, ax_grid, ax_stats):
    """Visualize MODEL interactively.
    AX_GRID is axis to be used to plot the pore.
    AX_STATS is pore filling stats axis.
    """
    # Update Grid Plot with triangular lattice visualization
    ax_grid.clear()

    # Collect coordinates and colors for different site types
    empty_x, empty_y = [], []
    na_x, na_y = [], []
    carbon_x, carbon_y = [], []
    defect_x, defect_y = [], []

    # We'll visualize all sites within 1.5 times pore radius to see some walls
    max_vis_radius = model.radius_lattice_units * 1.5

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
            elif site_type == model.CARBON:
                carbon_x.append(x)
                carbon_y.append(y)
            elif site_type == model.DEFECT:
                defect_x.append(x)
                defect_y.append(y)

    # Plot sites with different markers/colors
    # Scale marker size based on lattice spacing
    marker_scale = 100.0 / (model.grid_width / 8)  # Adjust scaling

    if empty_x:
        ax_grid.scatter(
            empty_x, empty_y, s=marker_scale, c='lightblue',
            edgecolors='gray', linewidths=0.5, alpha=0.7,
            label='Empty')
    if na_x:
        ax_grid.scatter(
            na_x, na_y, s=marker_scale, c='red', edgecolors='darkred',
            linewidths=0.5, alpha=0.9, label='Na')
    if carbon_x:
        ax_grid.scatter(
            carbon_x, carbon_y, s=marker_scale, c='black',
            edgecolors='gray', linewidths=0.5, alpha=0.5,
            label='Carbon')
    if defect_x:
        ax_grid.scatter(
            defect_x, defect_y, s=marker_scale, c='orange',
            edgecolors='darkorange', linewidths=0.5, alpha=0.8,
            label='Defect')

    # Draw pore boundary circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = model.radius_lattice_units * np.cos(theta)
    circle_y = model.radius_lattice_units * np.sin(theta)
    ax_grid.plot(circle_x, circle_y, 'k--', linewidth=1, alpha=0.7, label='Pore Boundary')

    # Set equal aspect ratio and limits
    ax_grid.set_aspect('equal')
    ax_grid.set_xlim(-max_vis_radius, max_vis_radius)
    ax_grid.set_ylim(-max_vis_radius, max_vis_radius)
    current_mcs = 'N/A' if model.mcs is None else model.mcs
    ax_grid.set_title(f"Pore State (MCS: {int(current_mcs)})")
    ax_grid.legend(loc='upper right', fontsize='small')
    # ax_grid.grid(True, alpha=0.3)
    ax_grid.grid(False)

    # Update Stats Plot
    ax_stats.clear()
    ax_stats.plot(model.time_points, model.filling_history, label='Filling Fraction')
    ax_stats.set_ylim(0, 1.2)
    ax_stats.set_xlabel('Monte Carlo Steps')
    ax_stats.set_ylabel('Filling %')
    ax_stats.set_title(f"Filling Kinetics (P_GCMC={model.default_p_gcmc:.2f})")
    ax_stats.grid(True)

    # Add simulation parameters as text
    param_text = (
        f"T = {model.T} K\n"
        f"V = {model.voltage:.2f} V\n"
        f"R = {model.pore_radius} Å\n"
        f"defects = {model.defect_probability:.3f} ({model.defect_placement})\n"
        f"E_Na-Na = {model.energies['Na_Na']:.3f} eV\n"
        f"E_Na-C = {model.energies['Na_C']:.3f} eV\n"
        f"E_Na-def = {model.energies['Na_Defect']:.3f} eV")
    ax_stats.text(0.02, 0.98, param_text, transform=ax_stats.transAxes,
                  fontsize=8, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))



def run_simulation(
        voltage=0.1, steps=20000, temp=298, radius=10.0,
        defect_placement='surface',
        defect_probability=0.058 * 3,
        visualize=True,
        snapshot_file='snapshots.pkl',
        energy_na_na=-0.35,
        energy_na_c=-0.32,
        energy_na_defect=-1.77,
        csv_output=False,
        seed=None,
        quiet=False):
    """
    Run a Monte Carlo simulation of pore filling.
    
    Args:
        voltage: Voltage relative to bulk Na (V)
        steps: Number of normalized Monte Carlo steps (MCS)
        temp: Temperature (K)
        radius: Pore radius (Å)
        defect_placement: 'surface' or 'random'
        snapshot_file: If provided, save snapshots to this pickle file.
        energy_na_na: Na-Na interaction energy (eV)
        energy_na_c: Na-C interaction energy (eV)
        energy_na_defect: Na-defect interaction energy (eV)
        csv_output: If True, print a CSV line with results to stdout.
        seed: Random seed for reproducibility (None for random).
        quiet: Suppress progress output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    MC_STEPS = steps  # Total normalized steps (attempts per site)
    SNAPSHOT_INTERVAL = 400

    # Initialize Model
    model = HardCarbonPoreModel(
        pore_radius_angstrom=radius,
        temperature_k=temp,
        voltage=voltage,
        # defect_probability=0.058,
        # Defect density should be scaled by unknown factor to get 3d->2d mapping
        # The carbons are placed on Na lattice, so the number of C is different
        # here and thus need to adjust concentration.
        # defect_probability=0.058 * 3,
        defect_probability=defect_probability,
        defect_placement=defect_placement,
        energy_na_na=energy_na_na,
        energy_na_c=energy_na_c,
        energy_na_defect=energy_na_defect,
        quiet=quiet,
    )
    snapshots = []

    total_sites = len(model.valid_sites)
    total_attempts = MC_STEPS * total_sites

    if not quiet:
        print(f"Starting Simulation: {total_attempts} attempts ({MC_STEPS} MCS)...")
    start_time = time.time()

    # Visualization Setup
    if visualize:
        fig, (ax_grid, ax_stats) = plt.subplots(1, 2, figsize=(12, 6))
        plt.show(block=False)

    for attempt in range(total_attempts):
        model.run_step()

        if model.equilibrium_reached:
            if not quiet:
                print(f"Equilibrium reached at MCS {model.mcs:.2f}")
            if visualize and not quiet:
                print(f"  Final filling = {model.filling_history[-1]:.2%}")
                visualize_model(model, ax_grid, ax_stats)
                plt.draw()
                plt.pause(0.01)
            if snapshot_file is not None:
                snapshots.append(model.take_snapshot())
            break

        if attempt % (SNAPSHOT_INTERVAL * total_sites) == 0\
           or attempt == total_attempts - 1:
            if visualize and not quiet:
                print(f"  Step {int(model.mcs)}/{MC_STEPS}:"
                      f" Filling = {model.filling_history[-1]:.2%}")
                visualize_model(model, ax_grid, ax_stats)
                plt.draw()
                plt.pause(0.01)
            if snapshot_file is not None:
                snapshots.append(model.take_snapshot())

    elapsed = time.time() - start_time
    if not quiet:
        print(f"Simulation Complete in {elapsed:.2f}s")
    if snapshot_file is not None:
        with open(snapshot_file, 'wb') as f:
            pickle.dump(snapshots, f)
        if not quiet:
            print(f"Saved {len(snapshots)} snapshots to {snapshot_file}")
    
    # CSV output
    if csv_output:
        final_filling = model.filling_history[-1] if model.filling_history else 0.0
        row = [
            f"{voltage:.6f}",
            f"{radius:.1f}",
            f"{defect_probability:.6f}",
            defect_placement,
            f"{energy_na_defect:.6f}",
            f"{energy_na_na:.6f}",
            f"{energy_na_c:.6f}",
            f"{temp:.1f}",
            f"{steps}",
            str(seed) if seed is not None else '',
            f"{final_filling:.6f}",
            str(model.equilibrium_reached),
            f"{model.mcs:.2f}",
            f"{len(model.valid_sites)}",
            f"{len(model.surface_sites)}",
            f"{model.default_p_gcmc:.6f}",
            f"{model.mu:.6f}",
            f"{model.mcs_fill}",
        ]
        print(','.join(row))
    return model

def replay_simulation(snapshot_file, interval=0.01, every=1):
    """
    Load snapshots from SNAPSHOT_FILE and visualize them sequentially.
    INTERVAL is pause time between frames in seconds.
    EVERY X will only show every X's snapshot.
    """
    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots")

    fig, (ax_grid, ax_stats) = plt.subplots(1, 2, figsize=(12, 6))
    plt.show(block=False)

    for i, model in enumerate(snapshots):
        if i % every != 0:
            continue
        visualize_model(model, ax_grid, ax_stats)
        ax_grid.set_title(f"Pore State (MCS: {int(model.mcs)}) - Snapshot {i+1}/{len(snapshots)}")
        ax_stats.set_title(f"Filling Kinetics (P_GCMC={model.default_p_gcmc:.2f}) - Snapshot {i+1}/{len(snapshots)}")
        plt.draw()
        plt.pause(interval)

    plt.show()


def summarize_snapshots(pattern="*.pkl", output_csv="summary.csv"):
    """
    Process all .pkl files matching PATTERN, extract final snapshot data,
    and save summary to OUTPUT_CSV.
    """
    import glob
    import pandas as pd
    import traceback

    files = glob.glob(pattern)
    if not files:
        print(f"No files matching pattern '{pattern}'")
        return

    print(f"Found {len(files)} files")

    data_rows = []

    for fpath in sorted(files):
        try:
            with open(fpath, 'rb') as f:
                snapshots = pickle.load(f)

            if not snapshots:
                print(f"Warning: {fpath} contains no snapshots")
                continue

            # Take the last snapshot
            model = snapshots[-1]

            # Extract parameters
            row = {
                'filename': fpath,
                'pore_radius_A': model.pore_radius,
                'voltage_V': model.voltage,
                'defect_probability': model.defect_probability,
                'defect_placement': model.defect_placement,
                'temperature_K': model.T,
                'energy_na_na_eV': model.energies['Na_Na'],
                'energy_na_c_eV': model.energies['Na_C'],
                'energy_na_defect_eV': model.energies['Na_Defect'],
                'final_filling': model.filling_history[-1] if model.filling_history else 0.0,
                'final_mcs': model.mcs,
                # 'equilibrium_reached': model.equilibrium_reached,
                'n_snapshots': len(snapshots),
                'n_valid_sites': len(model.valid_sites),
                'n_surface_sites': len(model.surface_sites),
                'default_p_gcmc': model.default_p_gcmc,
                'mu_eV': model.mu,
                'fill_mcs': model.fill_mcs,
            }

            data_rows.append(row)
            print(f"Processed {fpath}: R={model.pore_radius:.1f}Å, V={model.voltage:.2f}V, filling={row['final_filling']:.3f}")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            traceback.print_exc()
            continue

    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_csv(output_csv, index=False)
        print(f"Saved summary to {output_csv} with {len(df)} rows")
        return df
    else:
        print("No valid data extracted")
        return None


# Example usage:
# run_simulation(snapshot_file='snapshots.pkl')
# replay_simulation('snapshots.pkl')
# summarize_snapshots("v2.snapshots.*.pkl", "summary.csv")

def main():
    parser = argparse.ArgumentParser(
        description='Metropolis Monte Carlo simulation of pore filling in hard carbon.'
    )
    parser.add_argument('--voltage', type=float, default=0.1,
                        help='Voltage relative to bulk Na (V)')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Pore radius (Å)')
    parser.add_argument('--file', type=str, default='snapshots.pkl',
                        help='Output snapshot pickle file')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='Number of normalized Monte Carlo steps (MCS)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable live visualization')
    parser.add_argument('--energy_na_defect', type=float, default=-1.77,
                        help='Na-defect interaction energy (eV)')
    parser.add_argument('--energy_na_na', type=float, default=-0.35,
                        help='Na-Na interaction energy (eV)')
    parser.add_argument('--energy_na_c', type=float, default=-0.32,
                        help='Na-C interaction energy (eV)')
    parser.add_argument('--temp', type=float, default=298.0,
                        help='Temperature (K)')
    parser.add_argument('--defect_placement', type=str, default='surface',
                        choices=['surface', 'random'],
                        help='Defect placement mode')
    parser.add_argument('--defect_probability', type=float, default=0.058*3,
                        help='Defect probability (fraction)')
    parser.add_argument('--csv', action='store_true',
                        help='Output a single CSV line with final results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress all progress output')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    run_simulation(
        voltage=args.voltage,
        steps=args.steps,
        temp=args.temp,
        radius=args.radius,
        defect_placement=args.defect_placement,
        defect_probability=args.defect_probability,
        visualize=args.visualize,
        snapshot_file=args.file,
        energy_na_na=args.energy_na_na,
        energy_na_c=args.energy_na_c,
        energy_na_defect=args.energy_na_defect,
        csv_output=args.csv,
        quiet=args.quiet,
        seed=args.seed)

if __name__ == "__main__":
    main()
