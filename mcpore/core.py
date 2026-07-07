"""Core model and visualization for hard-carbon pore-filling MC simulation."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import copy
import sys
import pandas as pd

# Module-level constants
CSV_GZ_SUFFIX = '.csv.gz'


class HardCarbonPoreModel:
    """2D triangular-lattice Monte Carlo model of Na filling a hard-carbon pore."""

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
            energy_na_c=-0.26,
            energy_na_defect=-1.53,
            temperature_k=298.0,
            voltage=1.0,  # voltage relative to bulk Na
            initial_na_layers=0,  # pre-fill surface-adjacent layers with Na
            eq_window=4000,
            eq_slope_threshold=1e-8,
            eq_min_mcs=1E9,
            quiet=False,
            seed=None):
        """
        Initialize 2D Triangular Lattice Model with Metropolis Dynamics.

        defect_placement: 'random' (Bernoulli per wall site) or
                          'surface' (exact fraction of pore-surface wall sites).
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
        self.grid_width = int(1.1 * self.grid_width)

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

        # 5a. Compute the real pore radius from the furthest valid site
        self.real_radius_angstrom = self._compute_real_radius()

        # 5b. Initialize Na surface layers (pre-fill from non-empty state)
        self.initial_na_layers = initial_na_layers
        self._initialize_na_layers(initial_na_layers)

        # 6. Calculate default probabilities
        if len(self.valid_sites) > 0:
            self.default_p_gcmc = len(self.surface_sites) / (len(self.valid_sites) + len(self.surface_sites))
        else:
            self.default_p_gcmc = 0.0

        # 7. History
        self.steps = 0
        self.mcs_fill = 0.0 if self.get_filling_fraction() >= 0.9999 else None
        self.time_points = [0.0]
        self.filling_history = [self.get_filling_fraction() * 100]
        self.formation_energy_history = [self.formation_energy()]
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

    @property
    def beta(self) -> float:
        """Return beta (1/kT)."""
        return 1.0 / (self.kB * self.T)

    @property
    def mu(self):
        """Return chemical potential according to voltage and Na energies.

        We assume 2D, that's why 3.
        """
        return -self.voltage + 3 * self.energies['Na_Na']

    def _initialize_circular_pore(self, radius):
        """Creates a circular pore centered in the grid.

        RADIUS is the pore radius in lattice units.
        """
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
            # Exact fraction of pore-surface wall sites
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

    def _compute_wall_distance(self):
        """Compute distance in lattice hops from each valid site to nearest wall.

        Uses BFS from surface sites (layer 0) outward through valid sites.
        Returns a dict mapping (r, c) -> distance (int).
        """
        from collections import deque

        distance = {}
        queue = deque()

        # Layer 0: sites directly adjacent to carbon/defect walls
        for r, c in self.valid_sites:
            neighbors = self.get_neighbors(r, c, include_walls=True)
            for nr, nc in neighbors:
                if self.grid[nr, nc] in (self.CARBON, self.DEFECT):
                    distance[(r, c)] = 0
                    queue.append((r, c))
                    break

        # BFS outward through valid (non-wall) sites
        while queue:
            r, c = queue.popleft()
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) not in distance:
                    distance[(nr, nc)] = distance[(r, c)] + 1
                    queue.append((nr, nc))

        return distance

    def _initialize_na_layers(self, n_layers):
        """Pre-fill the first N_LAYERS of wall-adjacent sites with Na atoms.

        n_layers=0: empty pore (no change).
        n_layers=1: fill surface sites (adjacent to wall).
        n_layers=2: fill surface + first subsurface layer.
        Returns the number of Na atoms placed.
        """
        if n_layers <= 0:
            return 0

        distance = self._compute_wall_distance()
        count = 0
        for (r, c), d in distance.items():
            if d < n_layers:
                self.grid[r, c] = self.NA
                count += 1
        return count

    def get_neighbors(self, r, c, include_walls=False):
        """Returns list of neighbor coords for triangular lattice.

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
        """Calculates the potential energy of a Na atom if placed at (r,c).

        Sums interactions with existing neighbors.
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

        # Find all neighbors
        neighbors = self.get_neighbors(r, c, include_walls=True)

        nr, nc = random.choice(neighbors)

        if self.grid[nr, nc] != self.EMPTY:
            return False

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
            self.fine_time_points_exit.append(self.mcs)
            self.dE_history_exit.append(dE)

            if dE <= 0 or np.random.random() < np.exp(-dE * self.beta):
                self.grid[r, c] = self.EMPTY
                return True
        return False

    def attempt_swap(self):
        """Non-local Na hop: pick random Na, random empty site, Metropolis accept.

        Symmetric proposal preserves detailed balance.  This move accelerates
        equilibration across free-energy barriers (e.g. the surface ring) but
        is not physically meaningful for time.  Use only when equilibrium
        properties (filling fraction, voltage) are the goal.
        """
        na_sites = [(r, c) for r, c in self.valid_sites
                    if self.grid[r, c] == self.NA]
        empty_sites = [(r, c) for r, c in self.valid_sites
                       if self.grid[r, c] == self.EMPTY]
        if not na_sites or not empty_sites:
            return False
        r1, c1 = random.choice(na_sites)
        r2, c2 = random.choice(empty_sites)
        dE = self.calculate_swap_energy(r1, c1, r2, c2)
        if dE <= 0 or np.random.random() < np.exp(-dE * self.beta):
            self.grid[r1, c1] = self.EMPTY
            self.grid[r2, c2] = self.NA
            return True
        return False

    def get_filling_fraction(self):
        total_valid = len(self.valid_sites)
        filled = np.sum(self.grid == self.NA)
        return filled / total_valid

    def get_final_filling_percent(self):
        """Return average filling ratio, in %."""
        if self.filling_history is None:
            return 0
        if len(self.filling_history) < self.eq_window:
            return np.mean(self.filling_history)
        return np.mean(self.filling_history[-self.eq_window:-1])

    @property
    def mcs(self) -> float:
        """Number of normalized MC steps."""
        return self.steps / len(self.valid_sites)

    def run_step(self, p_gcmc=None, p_swap=0.0):
        """Executes one Monte Carlo Step (MCS).

        p_gcmc: Probability of attempting a GCMC move vs Diffusion move.
                If None, defaults to ratio of surface sites to valid sites.
        p_swap: Probability of attempting a non-local swap move.
                Non-local swaps are a computational acceleration for
                equilibration; they are not physical dynamics.
                Use p_swap=0.0 when measuring kinetics/time.
        """
        prob_gcmc = p_gcmc if p_gcmc is not None else self.default_p_gcmc

        if np.random.random() < prob_gcmc:
            if np.random.random() < p_swap:
                self.attempt_swap()
            else:
                self.attempt_gcmc()
        else:
            self.attempt_diffusion()

        self.steps += 1
        # Snapshot time of full pore filling
        if self.mcs_fill is None and self.get_filling_fraction() >= 0.99:
            self.mcs_fill = self.mcs
        # Record stats every 0.5 MCS (approx)
        if self.steps % (len(self.valid_sites) // 2) == 0:
            self.time_points.append(self.mcs)
            self.filling_history.append(self.get_filling_fraction() * 100)
            self.formation_energy_history.append(self.formation_energy())
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

    def _compute_real_radius(self):
        """Compute the actual pore radius from the closest wall site.

        Because of the discrete triangular grid, the user-specified
        pore_radius does not necessarily correspond to a real pore
        shape. This method finds the minimum distance from the pore
        center among all wall sites and returns it in angstroms.
        """
        if not self.adjacent_wall_sites:
            return 0.0
        min_dist = 1E100
        for r, c in self.adjacent_wall_sites:
            x, y = self.get_triangular_coordinates(r, c)
            dist = np.sqrt(x**2 + y**2)
            if dist < min_dist:
                min_dist = dist
        # Convert from lattice units to angstroms
        return min_dist * self.bond_length

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
            f"Real pore radius: {self.real_radius_angstrom:.3f} Å (from furthest valid site)",
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


# --- Visualization ---

def save_model_svg(model, filename, scale=80):
    """Save pore model atomic grid to FILENAME as SVG.

    SCALE is number of pixels per lattice unit in the SVG.
    """

    # 1. Collect elements to draw
    atoms = []  # List of (x, y, type)
    xs, ys = [], []

    # Visualization radius limit
    vis_limit_lattice = min(
        model.radius_lattice_units + 1.8,
        model.radius_lattice_units * 1.1)

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

    # 6. Draw Atoms
    atoms.sort(key=lambda a: a[1])

    for x, y, atype in atoms:
        px = (x - min_x + pad) * scale
        py = (max_y - y + pad) * scale  # Inverted Y for drawing

        fill_val = fill_carbon
        if atype == 'na_bulk':
            fill_val = fill_na
        elif atype == 'carbon_defect':
            fill_val = fill_defect

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
        formation_axis.plot(model.time_points[10::5], model.formation_energy_history[10::5],
                            label='Formation energy', color='red')
        formation_axis.set_ylabel('Formation energy, eV/atom')
    if dE_axis is not None:
        legend_axis = dE_axis
        dE_axis.clear()
        dE_axis.set_title("Entry and exit of Na")
        dE_axis.set_ylabel('Entry/exit energy, eV/atom')
        window = 5
        window_entry = int(window * (1.0 - model.default_p_gcmc)/model.default_p_gcmc)
        dE_axis.plot(model.fine_time_points_entry[::5],
                     pd.DataFrame(model.dE_history_entry).rolling(window_entry).mean()[::5],
                     color='green', label='dE entry')
        dE_axis.plot(model.fine_time_points_exit[::5],
                     pd.DataFrame(model.dE_history_exit).rolling(window).mean()[::5],
                     color='red', label='dE exit')
        dE_axis.legend()

    # Add simulation parameters as text
    param_text = (
        f"T = {model.T} K\n"
        f"V = {model.voltage:.2f} V\n"
        f"R = {model.pore_radius} Å\n"
        f"defects = {model.defect_probability:.3f} ({model.defect_placement})\n"
        f"initial Na layers = {model.initial_na_layers}\n"
        f"E_Na-Na = {model.energies['Na_Na']:.3f} eV\n"
        f"E_Na-C = {model.energies['Na_C']:.3f} eV\n"
        f"E_Na-def = {model.energies['Na_Defect']:.3f} eV")
    legend_axis.text(0.02, 0.98, param_text, transform=ax_stats.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


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
