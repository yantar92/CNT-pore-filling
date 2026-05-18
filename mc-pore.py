"""
Metropolis Monte Carlo Simulation for Hard Carbon Pore Filling.

Command line usage:
    python mc-pore.py [--voltage 0.1 [0.1 ...]] [--radius 10.0] [--file snapshots.pkl]
        [--steps 20000] [--visualize] [--energy_na_defect -1.77]
        [--temp 298] [--defect_placement surface] [--defect_probability 0.174]
        [--csv] [--quiet] [--seed INT] [--converge] [--convergence_threshold 0.05]
        [--min_replicates 3] [--max_replicates 50]
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle
import copy
import argparse
import sys
import os
import pandas as pd

# Module-level constants
CSV_GZ_SUFFIX = '.csv.gz'

class HardCarbonPoreModel:
    def __init__(
            self,
            pore_radius_angstrom=15,
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
            eq_window=4000,
            eq_slope_threshold=1e-8,
            eq_min_mcs=5000,
            quiet=False,
            seed=None):
        """
        Initialize 2D Triangular Lattice Model with Metropolis Dynamics.

        defect_placement: 'random' (Bernoulli per wall site) or
                          'surface' (exact fraction of pore‑surface wall sites).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
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
        self.formation_energy_history = [-self.mu]
        self.fine_time_points_entry = []
        self.fine_time_points_exit = []
        self.dE_history_entry = []
        self.dE_history_exit = []

        # 8. Equilibrium detection
        self.equilibrium_reached = False
        self.eq_window = eq_window  # number of samples for equilibrium check
        self.eq_slope_threshold = eq_slope_threshold  # slope per MCS threshold
        self.eq_min_mcs = eq_min_mcs  # minimum MCS before checking
        self.quiet = quiet

        # if not self.quiet:
        #     print(f"Model Initialized: {self.grid_width}x{self.grid_width} Grid")
        #     print(f"  Temp: {self.T} K, Beta: {self.beta:.2f} eV^-1")
        #     print(f"  Voltage: {self.voltage} V, Chem. pot: {-self.voltage} eV")
        #     print(f"  Valid Sites: {len(self.valid_sites)}")
        #     print(f"  Surface Sites: {len(self.surface_sites)}")
        #     print(f"  Defects: {self.defect_probability:.3f} ({self.defect_placement})")
        #     n_defects = 0
        #     for r, c in self.adjacent_wall_sites:
        #         n_defects += 1 if self.grid[r, c] == self.DEFECT else 0
        #     print(f"  Surface Carbons: {len(self.surface_sites)} ({n_defects} defects)")
        #     print(f"  Default P_GCMC: {self.default_p_gcmc:.4f}")

    @property
    def beta(self) -> float:
        """Return beta (1/kT).
        """
        return 1.0 / (self.kB * self.T)

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
                else:
                    # IMPORTANT: Avoid sites inside pore adjacent to
                    # more than 4 wall sites. That skews the energies.
                    neighbors = self.get_neighbors(r, c, include_walls=True)
                    n_wall = 0
                    for nr, nc in neighbors:
                        if distances[nr][nc] >= radius:
                            n_wall += 1
                    if n_wall > 4:
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

    def formation_energy(self, norm='Na'):
        """Calculate formation energy of the system.
        NORM is normalization type. Allowed values:
        'Na' - normalize by number of Na
        'pore' - normalize by total number of sites inside the pore
        None - do not normalize.
        """
        energy = 0
        tot_na = 0
        for r, c in self.valid_sites:
            if self.grid[r, c] == self.NA:
                tot_na += 1
            else:
                continue
            for nr, nc in self.get_neighbors(r, c, include_walls=True):
                if self.grid[nr, nc] == self.NA:
                    energy += self.energies['Na_Na'] / 2.0
                elif self.grid[nr, nc] == self.CARBON:
                    energy += self.energies['Na_C']
                elif self.grid[nr, nc] == self.DEFECT:
                    energy += self.energies['Na_Defect']
        if tot_na == 0:
            return 0
        fenergy_abs = energy - self.mu*tot_na
        if norm is None:
            return fenergy_abs
        elif norm == 'Na':
            return (energy - self.mu*tot_na)/tot_na
        # norm == 'pore'
        return (energy - self.mu*tot_na)/len(self.valid_sites)

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
            # print(f"Insertion: {dE} = {interaction} - {self.mu}")
            self.fine_time_points_entry.append(self.mcs)
            self.dE_history_entry.append(dE)

            if dE <= 0 or np.random.random() < np.exp(-dE * self.beta):
                self.grid[r, c] = self.NA
                return True
        elif self.grid[r, c] == self.NA:
            # --- REMOVAL ---
            # Reverse of insertion: Delta E = -(E_interaction - mu)
            interaction = self._calculate_potential_energy_at_site(r, c)
            dE = -(interaction - self.mu)
            # print(f"Removal: {dE} = {self.mu} - {interaction}")
            self.fine_time_points_exit.append(self.mcs)
            self.dE_history_exit.append(dE)

            if dE <= 0 or np.random.random() < np.exp(-dE * self.beta):
                self.grid[r, c] = self.EMPTY
                return True
        return False

    def get_filling_fraction(self):
        total_valid = len(self.valid_sites)
        filled = np.sum(self.grid == self.NA)
        return filled / total_valid

    def get_final_filling_percent(self):
        """Return average filling ratio, in %.
        """
        if self.filling_history is None:
            return 0
        if len(self.filling_history) < self.eq_window:
            return np.mean(self.filling_history)
        return np.mean(self.filling_history[-self.eq_window:-1])

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
            self.filling_history.append(self.get_filling_fraction() * 100)
            self.formation_energy_history.append(self.formation_energy())
            # Snapshot time of full pore filling
            if self.mcs_fill is None and self.get_filling_fraction() == 1:
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

    def __repr__(self):
        """Brief representation of model state."""
        filled = np.sum(self.grid == self.NA)
        total = len(self.valid_sites)
        if total == 0:
            frac = 0.0
        else:
            frac = filled / total
        return (f"HardCarbonPoreModel(R={self.pore_radius:.1f}Å, "
                f"V={self.voltage:.2f}V, T={self.T}K, "
                f"filling={filled}/{total}={frac:.1%}, "
                f"MCS={self.mcs:.1f})")

    def __str__(self):
        """Detailed summary of model state."""
        filled = np.sum(self.grid == self.NA)
        total = len(self.valid_sites)
        if total == 0:
            frac = 0.0
        else:
            frac = filled / total
        lines = [
            "Hard Carbon Pore Model",
            "======================",
            f"Pore radius: {self.pore_radius} Å (lattice units: {self.radius_lattice_units:.2f})",
            f"Grid: {self.grid_width}x{self.grid_width}",
            f"Valid sites: {total}, Surface sites: {len(self.surface_sites)}",
            f"Defects: {self.defect_probability:.3f} ({self.defect_placement}), "
                f"Na-defect energy: {self.energies['Na_Defect']:.3f} eV",
            f"Temperature: {self.T} K, Beta: {self.beta:.2f} eV^-1",
            f"Voltage: {self.voltage} V, Chemical potential mu: {self.mu:.3f} eV",
            f"Interaction energies: Na-Na {self.energies['Na_Na']:.3f} eV, "
                f"Na-C {self.energies['Na_C']:.3f} eV",
            f"Default P_GCMC: {self.default_p_gcmc:.4f}",
            f"Current filling: {filled}/{total} ({frac:.1%})",
            f"Monte Carlo steps: {self.mcs:.1f} (steps={self.steps})",
            f"Equilibrium reached: {self.equilibrium_reached}",
        ]
        if self.mcs_fill is not None:
            lines.append(f"Pore filled at MCS: {self.mcs_fill:.1f}")
        return "\n".join(lines)

    def pretty_print(self, file=sys.stdout):
        """Print detailed summary of model state to FILE (default stdout)."""
        print(str(self), file=file)

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


def visualize_model(model, ax_grid, ax_stats, dE_axis=None, formation_axis=None):
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
    ax_stats.set_ylim(0, 120)
    ax_stats.set_xlabel('Monte Carlo Steps')
    ax_stats.set_ylabel('Filling %')
    ax_stats.set_title(f"Filling Kinetics (P_GCMC={model.default_p_gcmc:.2f})")
    ax_stats.grid(True)
    lines1, labels1 = ax_stats.get_legend_handles_labels()
    lines2, labels2 = [], []
    legend_axis = ax_stats
    if formation_axis is not None:
        formation_axis.clear()
        formation_axis.set_title("Formation energy history")
        formation_axis.plot(model.time_points[10::5], model.formation_energy_history[10::5], label='Formation energy', color='red')
        formation_axis.set_ylabel('Formation energy, eV/atom')
    if dE_axis is not None:
        legend_axis = dE_axis
        dE_axis.clear()
        dE_axis.set_title("Entry and exit of Na")
        dE_axis.set_ylabel('Entry/exit energy, eV/atom')
        # dE_axis.yaxis.set_label_position('right')
        window = 5
        window_entry = int(window * (1.0 - model.default_p_gcmc)/model.default_p_gcmc)
        dE_axis.plot(model.fine_time_points_entry[::5], pd.DataFrame(model.dE_history_entry).rolling(window_entry).mean()[::5], color='green', label='dE entry')
        dE_axis.plot(model.fine_time_points_exit[::5], pd.DataFrame(model.dE_history_exit).rolling(window).mean()[::5], color='red', label='dE exit')
        dE_axis.legend()
        # lines2, labels2 = dE_axis.get_legend_handles_labels()

    # legend_axis.legend(lines1 + lines2, labels1 + labels2, loc=0)

    # Add simulation parameters as text
    param_text = (
        f"T = {model.T} K\n"
        f"V = {model.voltage:.2f} V\n"
        f"R = {model.pore_radius} Å\n"
        f"defects = {model.defect_probability:.3f} ({model.defect_placement})\n"
        f"E_Na-Na = {model.energies['Na_Na']:.3f} eV\n"
        f"E_Na-C = {model.energies['Na_C']:.3f} eV\n"
        f"E_Na-def = {model.energies['Na_Defect']:.3f} eV")
    legend_axis.text(0.02, 0.98, param_text, transform=ax_stats.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def run_simulation(
        model=HardCarbonPoreModel(
            pore_radius_angstrom=10.0,
            temperature_k=298,
            voltage=0.1,
            # defect_probability=0.058,
            # Defect density should be scaled by unknown factor to get 3d->2d mapping
            # The carbons are placed on Na lattice, so the number of C is different
            # here and thus need to adjust concentration.
            # defect_probability=0.058 * 3,
            defect_probability=0.058 * 3,
            defect_placement='surface',
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            quiet=True,
        ),
        steps=20000,
        visualize=True,
        snapshot_file: str | None = 'snapshots.pkl',
        csv_output=False,
        seed=None,
        anneal0K=False,
        quiet=False):
    """
    Run a Monte Carlo simulation of pore filling.

    Args:
        model: HardCarbonPoreModel
        steps: Number of normalized Monte Carlo steps (MCS)
        snapshot_file: If provided, save output to this file.
            If the filename ends with '.csv' or '.csv.gz', writes the time series
            (MCS, filling %, formation energy) as CSV (optionally gzip-compressed)
            instead of pickle snapshots.
        csv_output: If True, print a CSV line with results to stdout.
        seed: Random seed for reproducibility (None for random).
        quiet: Suppress progress output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Determine output mode from filename extension
    is_csv_output = (snapshot_file is not None
                     and (snapshot_file.lower().endswith('.csv')
                          or snapshot_file.lower().endswith(CSV_GZ_SUFFIX)))

    MC_STEPS = steps  # Total normalized steps (attempts per site)
    SNAPSHOT_INTERVAL = 400

    # Initialize Model
    model.quiet = quiet
    model.equilibrium_reached = False
    snapshots = []
    BEGIN_MC = model.mcs

    total_sites = len(model.valid_sites)
    total_attempts = MC_STEPS * total_sites

    if not quiet:
        print(f"Starting Simulation: {total_attempts} attempts ({MC_STEPS} MCS)...")
    start_time = time.time()

    # Visualization Setup
    if visualize:
        fig, ((ax_grid, ax_stats), (dE_axis, formation_axis)) = plt.subplots(2, 2, figsize=(10, 10))
        # dE_axis = ax_stats.twinx()
        plt.show(block=False)

    for attempt in range(total_attempts):
        model.run_step()
        if model.equilibrium_reached and attempt < model.eq_min_mcs:
            model.equilibrium_reached = False

        if model.equilibrium_reached and anneal0K:
            model = run_0K_min(model, steps=steps)

        should_report = (model.equilibrium_reached
                         or attempt == total_attempts - 1
                         or attempt % (SNAPSHOT_INTERVAL * total_sites) == 0)

        if should_report:
            # Collect snapshots only when writing pickle output
            if snapshot_file is not None and not is_csv_output:
                snapshots.append(model.take_snapshot())
            if not quiet and model.equilibrium_reached:
                print(f"Equilibrium reached at MCS {model.mcs:.2f}")
            elif not quiet:
                print(f"  Step {int(model.mcs)}/{MC_STEPS + BEGIN_MC}:"
                      f" Filling = {model.filling_history[-1]:.2f}%")
            if not quiet and visualize:
                visualize_model(model, ax_grid, ax_stats, dE_axis, formation_axis)
                plt.draw()
                plt.pause(0.01)
        if model.equilibrium_reached:
            break

    elapsed = time.time() - start_time
    if not quiet:
        print(f"Simulation Complete in {elapsed:.2f}s")

    if snapshot_file is not None:
        if is_csv_output:
            # Write time series as CSV instead of pickle snapshots
            save_timeseries_csv(model, snapshot_file)
        else:
            with open(snapshot_file, 'wb') as f:
                pickle.dump(snapshots, f)
            if not quiet:
                print(f"Saved {len(snapshots)} snapshots to {snapshot_file}")

    # CSV output
    if csv_output:
        final_filling = model.get_final_filling_percent()
        row = [
            f"{model.voltage:.6f}",
            f"{model.pore_radius:.1f}",
            f"{model.defect_probability:.6f}",
            model.defect_placement,
            f"{model.energies['Na_Defect']:.6f}",
            f"{model.energies['Na_Na']:.6f}",
            f"{model.energies['Na_C']:.6f}",
            f"{model.T:.1f}",
            f"{steps}",
            str(seed),
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


def run_voltage_sweep_simulation(
        model=HardCarbonPoreModel(),
        voltages=np.arange(0.2, -1e-9, -0.01, ),
        steps=20000,
        visualize=True,
        converge=False,
        seed=None,
        anneal0K=False,
        quiet=True):
    """Run MODEL sweeping across VOLTAGES.
    For each voltage, hold up to STEPS or until MODEL stabilization.
    SEED is random seed.
    When VISUALIZE is True, visualize the model.
    When QUIET is True, avoid printing info.
    When CONVERGE is False, run simulation for each voltage once.
    Otherwise, CONVERGE should be a dict {'threshold': 0.01, 'max_runs': 50, 'min_runs': 3}
    When ANNEAL0K is True, anneal at 0K after each step.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if visualize:
        fig, ((ax_grid, ax_stats), (voltage_axis, formation_axis)) = plt.subplots(2, 2, figsize=(10, 10))
        # dE_axis = ax_stats.twinx()
        plt.show(block=False)
    filling_data = []
    for voltage in voltages:
        model.voltage = voltage
        if converge:
            tem = copy.deepcopy(model)
            run_convergence_simulation(
                tem, steps=steps,
                convergence_threshold=converge['threshold'],
                min_replicates=converge['min_runs'],
                max_replicates=converge['max_runs'],
                quiet=quiet,
                anneal0K=anneal0K,
                snapshot_file=None,
            )
            model = tem
        else:
            run_simulation(
                model,
                steps=steps,
                visualize=False,
                snapshot_file=None,
                csv_output=True,
                anneal0K=anneal0K,
                quiet=quiet)
        save_model_svg(model, f"snapshot_{voltage}.svg")
        filling_data.append(model.get_final_filling_percent())
        if visualize:
            visualize_model(model, ax_grid, ax_stats, formation_axis=formation_axis)
            voltage_axis.clear()
            voltage_axis.set_title('CE profile')
            voltage_axis.set_ylabel('Voltage, V')
            voltage_axis.set_xlabel('Filling ratio, %')
            voltage_axis.plot(filling_data, voltages[:len(filling_data)])
            plt.draw()
            plt.pause(0.01)


def run_convergence_simulation(
        model=HardCarbonPoreModel(
            pore_radius_angstrom=10.0,
            temperature_k=298,
            voltage=0.1,
            # defect_probability=0.058,
            # Defect density should be scaled by unknown factor to get 3d->2d mapping
            # The carbons are placed on Na lattice, so the number of C is different
            # here and thus need to adjust concentration.
            # defect_probability=0.058 * 3,
            defect_probability=0.058 * 3,
            defect_placement='surface',
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            quiet=True,
        ),
        steps=20000,
        convergence_threshold=0.01,
        min_replicates=3,
        max_replicates=50,
        seed=None,
        anneal0K=False,
        snapshot_file=None,
        quiet=False):
    """
    Run multiple simulations until statistics of final filling and pore filling time converge.
    Prints CSV line for each replicate.
    When SNAPSHOT_FILE is given, each replicate saves output using the file name
    as base with replicate index suffix (e.g. data_r0.csv, data_r1.csv).
    Returns list of (final_filling, mcs_fill) tuples.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    replicates = []
    fill_means = []
    fill_stds = []
    time_means = []
    time_stds = []

    for i in range(max_replicates):
        # Generate per-replicate filename if snapshot_file is given
        rep_snapshot = None
        if snapshot_file is not None:
            if snapshot_file.endswith(CSV_GZ_SUFFIX):
                base = snapshot_file[:-len(CSV_GZ_SUFFIX)]
                ext = CSV_GZ_SUFFIX
            else:
                base, ext = os.path.splitext(snapshot_file)
            rep_snapshot = f"{base}_r{i}{ext}"

        # Run simulation with csv_output=True (prints CSV line)
        model_tem = copy.deepcopy(model)
        model_tem.quiet = quiet
        model_tem = run_simulation(
            model=model_tem,
            steps=steps,
            visualize=False,
            snapshot_file=rep_snapshot,
            csv_output=True,
            seed=None,
            anneal0K=anneal0K,
            quiet=quiet)

        # Extract results
        final_filling = model_tem.get_final_filling_percent()
        mcs_fill = model_tem.mcs_fill if model_tem.mcs_fill is not None else 0.0

        replicates.append((final_filling, mcs_fill))

        # Check convergence after min_replicates
        if len(replicates) >= min_replicates:
            # Compute current statistics
            fill_array = np.array([r[0] for r in replicates])
            time_array = np.array([r[1] for r in replicates])
            fill_mean = fill_array.mean()
            fill_std = fill_array.std(ddof=1) if len(replicates) > 1 else 0.0
            time_mean = time_array.mean()
            time_std = time_array.std(ddof=1) if len(replicates) > 1 else 0.0

            if np.isclose(fill_mean, 0):
                fill_mean = 0
            if np.isclose(fill_std, 0):
                fill_std = 0
            if np.isclose(time_mean, 0):
                time_mean = 0
            if np.isclose(time_std, 0):
                time_std = 0

            # Compute relative changes (handle zero denominators)
            fill_mean_change = 0.0
            fill_std_change = 0.0
            time_mean_change = 0.0
            time_std_change = 0.0

            if fill_means:
                prev_fill_mean = fill_means[-1]
                if abs(prev_fill_mean) > 1e-12:
                    fill_mean_change = abs(fill_mean - prev_fill_mean) / prev_fill_mean
                else:
                    fill_mean_change = abs(fill_mean - prev_fill_mean)
                prev_fill_std = fill_stds[-1]
                if prev_fill_std > 1e-12:
                    fill_std_change = abs(fill_std - prev_fill_std) / prev_fill_std
                else:
                    fill_std_change = abs(fill_std - prev_fill_std)

            if time_means:
                prev_time_mean = time_means[-1]
                if abs(prev_time_mean) > 1e-12:
                    time_mean_change = abs(time_mean - prev_time_mean) / prev_time_mean
                else:
                    time_mean_change = abs(time_mean - prev_time_mean)
                prev_time_std = time_stds[-1]
                if prev_time_std > 1e-12:
                    time_std_change = abs(time_std - prev_time_std) / prev_time_std
                else:
                    time_std_change = abs(time_std - prev_time_std)

            fill_means.append(fill_mean)
            fill_stds.append(fill_std)
            time_means.append(time_mean)
            time_stds.append(time_std)

            if not quiet:
                print(f"fill_mean: Δ{fill_mean_change}, fill_std: Δ{fill_std_change}")
                print(f"time_mean: Δ{time_mean_change}, time_std: Δ{time_std_change}")
            # Check if all changes below threshold
            if (fill_mean_change <= convergence_threshold and
                fill_std_change <= convergence_threshold and
                time_mean_change <= convergence_threshold and
                time_std_change <= convergence_threshold):
                if not quiet:
                    print(f"Convergence reached after {i+1} replicates", file=sys.stderr)
                break
        else:
            # Not enough replicates yet, still store stats for future comparison
            fill_array = np.array([r[0] for r in replicates])
            time_array = np.array([r[1] for r in replicates])
            fill_mean = fill_array.mean()
            fill_std = fill_array.std(ddof=1) if len(replicates) > 1 else 0.0
            time_mean = time_array.mean()
            time_std = time_array.std(ddof=1) if len(replicates) > 1 else 0.0
            fill_means.append(fill_mean)
            fill_stds.append(fill_std)
            time_means.append(time_mean)
            time_stds.append(time_std)

    return replicates


def run_0K_min(
        model=HardCarbonPoreModel(
            pore_radius_angstrom=10.0,
            temperature_k=0,
            voltage=0.1,
            # defect_probability=0.058,
            # Defect density should be scaled by unknown factor to get 3d->2d mapping
            # The carbons are placed on Na lattice, so the number of C is different
            # here and thus need to adjust concentration.
            # defect_probability=0.058 * 3,
            defect_probability=0.058 * 3,
            defect_placement='surface',
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            quiet=True,
        ),
        steps=20000,
        seed=None):
    """
    Minimize energy in MODEL at 0K, while keeping the number of Na constant.

    Args:
        model: HardCarbonPoreModel
        steps: Number of normalized Monte Carlo steps (MCS)
        snapshot_file: If provided, save snapshots to this pickle file.
        csv_output: If True, print a CSV line with results to stdout.
        seed: Random seed for reproducibility (None for random).
        quiet: Suppress progress output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    MC_STEPS = steps  # Total normalized steps (attempts per site)

    # Initialize Model
    old_T = model.T
    old_quiet = model.quiet
    model.quiet = True

    total_sites = len(model.valid_sites)
    total_attempts = MC_STEPS * total_sites

    N_temps = 5

    # Down to 1K, we cannot use literally 0K.
    for new_T in np.logspace(np.log10(old_T), np.log10(1), N_temps):
        model.T = new_T

        for attempt in range(int(total_attempts/N_temps)):
            # Run step, while disallowing Na exiting or entering.
            model.run_step(p_gcmc=0)

    model.quiet = old_quiet
    model.T = old_T

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

    fig, ((ax_grid, ax_stats), (dE_axis, formation_axis)) = plt.subplots(2, 2, figsize=(10, 10))
    # dE_axis = ax_stats.twinx()
    plt.show(block=False)

    for i, model in enumerate(snapshots):
        if i % every != 0:
            continue
        visualize_model(model, ax_grid, ax_stats, dE_axis, formation_axis)
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
                'final_filling': model.get_final_filling_percent(),
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
# run_simulation(snapshot_file='data.csv')      # writes CSV time series
# run_simulation(snapshot_file='data.csv.gz')   # writes gzip-compressed CSV time series
# replay_simulation('snapshots.pkl')
# summarize_snapshots("v2.snapshots.*.pkl", "summary.csv")


def save_timeseries_csv(model, csv_path):
    """Save simulation time series data to a CSV file.

    The CSV contains columns: mcs, filling_pct, formation_energy.
    If csv_path ends with '.csv.gz', writes gzip-compressed CSV.
    """
    data = {
        'mcs': model.time_points,
        'filling_pct': model.filling_history,
        'formation_energy': model.formation_energy_history,
    }
    df = pd.DataFrame(data)
    compression = 'gzip' if csv_path.endswith(CSV_GZ_SUFFIX) else None
    df.to_csv(csv_path, index=False, compression=compression)
    if not model.quiet:
        print(f"Saved time series to {csv_path}")

    # Events data
    # base, _ = os.path.splitext(csv_path)
    # events_path = base + '_events.csv'

    # events = []
    # for mcs, dE in zip(model.fine_time_points_entry, model.dE_history_entry):
    #     events.append({'mcs': mcs, 'type': 'entry', 'dE': dE})
    # for mcs, dE in zip(model.fine_time_points_exit, model.dE_history_exit):
    #     events.append({'mcs': mcs, 'type': 'exit', 'dE': dE})

    # if events:
    #     df_events = pd.DataFrame(events)
    #     df_events = df_events.sort_values('mcs')
    #     df_events.to_csv(events_path, index=False)
    #     if not model.quiet:
    #         print(f"Saved event data to {events_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Metropolis Monte Carlo simulation of pore filling in hard carbon.'
    )
    parser.add_argument('--voltage', type=float, default=[0.1], nargs='*',
                        help='Voltage relative to bulk Na (V); single value or multiple for sweep')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Pore radius (Å)')
    parser.add_argument('--file', type=str, default='snapshots.pkl',
                        help='Output file. If the filename ends with .csv, writes time series'
                        ' CSV instead of pickle snapshots.'
                        ' In convergence mode, appends _rN suffix per replicate.')
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
    parser.add_argument('--converge', action='store_true',
                        help='Enable convergence loop: run replicates until statistics stabilize')
    parser.add_argument('--convergence_threshold', type=float, default=0.05,
                        help='Relative change threshold for mean and std (default: 0.05)')
    parser.add_argument('--min_replicates', type=int, default=3,
                        help='Minimum number of replicates before checking convergence (default: 3)')
    parser.add_argument('--max_replicates', type=int, default=50,
                        help='Maximum number of replicates (default: 50)')
    parser.add_argument('--csv', action='store_true',
                        help='Output a single CSV line with final results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress all progress output')
    parser.add_argument('--anneal', action='store_true',
                        help='Anneal the model at 0K after each step')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Normalize voltage to a list
    if isinstance(args.voltage, list):
        voltages = args.voltage
    else:
        voltages = [args.voltage]

    # Prepare converge dict if converge flag is set
    converge_dict = False
    if args.converge:
        converge_dict = {
            'threshold': args.convergence_threshold,
            'max_runs': args.max_replicates,
            'min_runs': args.min_replicates
        }

    model = HardCarbonPoreModel(
        voltage=voltages[0],
        temperature_k=args.temp,
        pore_radius_angstrom=args.radius,
        defect_placement=args.defect_placement,
        defect_probability=args.defect_probability,
        energy_na_na=args.energy_na_na,
        energy_na_c=args.energy_na_c,
        energy_na_defect=args.energy_na_defect,
        quiet=args.quiet,
        seed=args.seed
        )

    if len(voltages) == 1:
        # Single voltage mode
        if args.converge:
            run_convergence_simulation(
                model,
                steps=args.steps,
                convergence_threshold=args.convergence_threshold,
                min_replicates=args.min_replicates,
                max_replicates=args.max_replicates,
                seed=args.seed,
                anneal0K=args.anneal,
                snapshot_file=args.file,
                quiet=args.quiet)
        else:
            run_simulation(
                model,
                visualize=args.visualize,
                snapshot_file=args.file,
                steps=args.steps,
                csv_output=args.csv,
                quiet=args.quiet,
                anneal0K=args.anneal,
                seed=args.seed)
    else:
        # Multiple voltages: run voltage sweep
        # Create base model with first voltage (will be overwritten)
        run_voltage_sweep_simulation(
            model,
            voltages=voltages,
            steps=args.steps,
            visualize=args.visualize,
            converge=converge_dict,
            anneal0K=args.anneal,
            seed=args.seed,
            quiet=args.quiet)


def plot_filled_pore_energy():
    """Plot formation energy of the pore vs. pore radius.
    Consider the pore to be fully filled.
    """
    energy_na_na = -0.35
    formation_energies = []
    formation_energies2 = []
    formation_energies3 = []
    formation_energies4 = []
    radiuses = []
    inv_radiuses = []
    for radius in np.arange(5, 100, 1, dtype=float):
        model = HardCarbonPoreModel(
            radius,
            defect_probability=0,
            energy_na_c=energy_na_na,
            energy_na_na=energy_na_na,
            voltage=0)
        model2 = HardCarbonPoreModel(
            radius,
            defect_probability=0,
            energy_na_c=-0.33,
            energy_na_na=energy_na_na,
            voltage=0)
        model3 = HardCarbonPoreModel(
            radius,
            defect_probability=0.174,
            energy_na_c=-0.33,
            energy_na_na=energy_na_na,
            voltage=0)
        model4 = HardCarbonPoreModel(
            radius,
            defect_probability=0,
            energy_na_c=0,
            energy_na_na=energy_na_na,
            voltage=0)
        # Fill the pore
        for m in [model, model2, model3, model4]:
            for r, c in m.valid_sites:
                m.grid[r, c] = model.NA
        formation_energies.append(model.formation_energy())
        formation_energies2.append(model2.formation_energy())
        formation_energies3.append(model3.formation_energy())
        formation_energies4.append(model4.formation_energy())
        radiuses.append(radius)
        inv_radiuses.append(1.0/radius)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(inv_radiuses, formation_energies, 'o-', label='Na in Na')
    ax.plot(inv_radiuses, formation_energies2, 'o-', label='Na in C')
    ax.plot(inv_radiuses, formation_energies3, 'o-', label='Na in C (with defects)')
    ax.plot(inv_radiuses, formation_energies4, 'o-', label='Na in C (Na-C = 0eV)')
    # ax.set_xlabel('Radius, Å')
    ax.set_xlabel('Reciprocal radius, 1/Å')
    ax.set_ylabel('Formation energy, eV/atom')
    ax.set_title('Formation energy of fully filled pore (Na in Na)')
    ax.legend()
    ax.grid()
    plt.show()

def get_formation_energies(radius, defect_probability=0.058*3, norm='Na'):
    """Get formation energy vs. concentration for pore with RADIUS.
    Return (filling_ratios, energies).
    """
    model = HardCarbonPoreModel(
        pore_radius_angstrom=radius,
        temperature_k=4000,
        defect_probability=defect_probability,
        energy_na_c=-0.32,
        energy_na_na=-0.35,
        energy_na_defect=-1.77,
        # energy_na_defect=-0.77,
        voltage=0)
    filling_ratios = [0]
    energies = [model.formation_energy(norm)]
    for idx in range(len(model.valid_sites)):
        min_energy = 1E100
        min_loc = None
        for r, c in model.valid_sites:
            if model.grid[r, c] == model.EMPTY:
                model.grid[r, c] = model.NA
                new_energy = model.formation_energy(norm)
                model.grid[r, c] = model.EMPTY
                if new_energy < min_energy:
                    min_energy = new_energy
                    min_loc = (r, c)
        assert min_loc is not None
        model.grid[min_loc] = model.NA
        # model = run_0K_min(model)
        filling_ratios.append(model.get_filling_fraction())
        energies.append(model.formation_energy(norm))
        save_model_svg(model, f'test_{radius}_{defect_probability}_{idx}.svg')
    return filling_ratios, energies

def plot_formation_energies(radii=[5, 6, 10, 16, 20, 24, 30], defect_probability=0.058*3): # 
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for radius in radii:
        filling_ratios, energies = get_formation_energies(radius, defect_probability, norm='pore')
        # deltas = []
        prev_en = None
        # for en in energies:
        #     if prev_en is None:
        #         deltas.append(0)
        #         prev_en = en
        #     else:
        #         deltas.append(en - prev_en)
        #         prev_en = en
        ax.plot(filling_ratios, energies, 'o-', label=f'{radius}Å')
    ax.set_xlabel('Filling ratio')
    ax.set_ylabel('Formation energy, eV/atom')
    ax.set_title(f'Formation energies of gradually filled pore (defects: {defect_probability})')
    ax.legend()
    ax.grid()
    # plt.show()
    name = f"formation_energy_vs_filling_{defect_probability:.2f}"
    plt.savefig(f'{name}.svg')
    plt.savefig(f'{name}.png')

def plot_voltages(radii=[5, 7, 8, 10, 16, 20, 24, 30], defect_probability=0.058*3):
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.apps.battery.insertion_battery import InsertionElectrode
    from pymatgen.apps.battery.plotter import VoltageProfilePlotter
    from pymatgen.core import Composition
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for radius in radii:
        filling_ratios, energies = get_formation_energies(radius, defect_probability, norm=None)
        voltages = []
        entries = []
        print(energies)
        na_entry = ComputedEntry(Composition("Na"), 0)
        na_entry.data["volume"] = 1
        c_entry = ComputedEntry(Composition("C"), 0)
        c_entry.data["volume"] = 1
        entries.append(c_entry)
        # entries.append(na_entry)
        for n, e in enumerate(energies):
            if n == 0:
                continue
            # print(f"Na{n}C")
            entry = ComputedEntry(Composition(f"Na{n}C"), e)
            entry.data["volume"] = 1
            entries.append(entry)
        electrode = InsertionElectrode.from_entries(
            entries, working_ion_entry=na_entry, strip_structures=False)
        plotter = VoltageProfilePlotter(xaxis='x_form')
        x, voltage = plotter.get_plot_data(electrode, term_zero=False)
        ax.plot(np.array(x) / (len(energies) - 1), voltage, 'o-', label=f'{radius}Å')
    ax.set_xlabel('Filling ratio')
    ax.set_ylabel('Voltage, V')
    ax.set_title(f'Voltages for gradually filled pore (defects: {defect_probability})')
    ax.legend()
    ax.grid()
    # plt.show()
    name = f"voltage_{defect_probability:.2f}"
    plt.savefig(f'{name}.svg')
    plt.savefig(f'{name}.png')

if __name__ == "__main__":
    main()
