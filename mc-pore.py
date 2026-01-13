"""This file implements Monte-Carlo model of pore filing in hard carbon.
"""
import numpy as np


class HardCarbonPoreModel:
    def __init__(
            self,
            pore_radius_angstrom=10.0,
            # 3.72A experimental
            # 3.59346 optB88-vdW from our data
            na_bond_length_angstrom=3.59346, 
            grid_padding_angstrom=10.0,
            defect_probability=0.1,
            # eV
            energy_na_na=-1.0,
            energy_na_c=-0.5,
            energy_na_defect=-1.5):
        """
        Initialize 2D Triangular Lattice Model with Real Units.

        Args:
            pore_radius_angstrom (float): Real diameter of the pore (Å).
            na_bond_length_angstrom (float): NN distance for Na BCC (Å).
            grid_padding_angstrom (float): Extra space around pore (Å).
            defect_probability (float): Probability for carbon to be defective.
            energy_na_na, energy_na_c, energy_na_defect:
              Interaction energies (eV).
        """
        # 1. Geometry Constants
        self.bond_length = na_bond_length_angstrom
        self.pore_radius = pore_radius_angstrom

        self.defect_probability = defect_probability

        # 2. Convert to Lattice Units
        # In a triangular grid, the distance between neighbor centers
        # is 1 lattice unit.
        # So, 1 lattice unit = 1 Bond Length.
        self.radius_lattice_units = pore_radius_angstrom / self.bond_length

        # Determine Grid Size (Pore + Padding)
        padding_lattice = int(grid_padding_angstrom / self.bond_length)
        pore_span_lattice = int(self.radius_lattice_units * 2)
        self.grid_width = pore_span_lattice + 2 * padding_lattice + 4  # +4 buffer

        # 3. Energetics
        self.energies = {
            'Na_Na': energy_na_na,
            'Na_C': energy_na_c,
            'Na_Defect': energy_na_defect
        }

        # 4. State Grid: 0 = Empty, 1 = Na filled
        self.EMPTY = 0
        self.FILLED = 1
        self.grid = np.zeros((self.grid_width, self.grid_width), dtype=int)

        # 5. Structure Maps
        self.is_carbon = np.zeros((self.grid_width, self.grid_width), dtype=bool)
        self.is_defect = np.zeros((self.grid_width, self.grid_width), dtype=bool)

        self._initialize_circular_pore(self.radius_lattice_units)

        print(f"Model Initialized: {self.grid_width}x{self.grid_width} Grid")
        print(f"  Real Pore Diameter: {self.pore_radius * 2:.2f} Å")
        print(f"  Na-Na Bond Length:  {self.bond_length:.5f} Å")
        print(f"  Lattice Radius:     {self.radius_lattice_units:.2f} units")

    def _initialize_circular_pore(self, radius):
        """Creates a circular pore centered in the grid."""
        center = self.grid_width // 2
        for r in range(self.grid_width):
            for c in range(self.grid_width):
                # Calculate distance in offset coordinates (Euclidean approx)
                dist = np.sqrt((r - center)**2 + (c - center)**2)

                # If outside radius, it's the carbon wall
                if dist >= radius:
                    self.is_carbon[r, c] = True
                    # Random defects (adjustable prob)
                    if np.random.random() < self.defect_probability:
                        self.is_defect[r, c] = True

    def get_neighbors(self, r, c):
        """Returns list of valid neighbor coordinates for triangular lattice.
        Return a list of (nr, nc)."""
        candidates = [
            (r, c - 1), (r, c + 1),  # Same row
            (r - 1, c), (r + 1, c)   # Vertical
        ]
        if r % 2 == 0:  # Even rows
            candidates.extend([(r - 1, c - 1), (r + 1, c - 1)])
        else:          # Odd rows
            candidates.extend([(r - 1, c + 1), (r + 1, c + 1)])

        valid_neighbors = []
        for nr, nc in candidates:
            if 0 <= nr < self.grid_width and 0 <= nc < self.grid_width:
                valid_neighbors.append((nr, nc))
        return valid_neighbors

    def calculate_swap_energy(self, r1, c1, r2, c2):
        """
        Calculates Delta E for moving a particle from (r1, c1) to (r2, c2).
        Assumes (r1, c1) is currently FILLED and (r2, c2) is EMPTY.
        """
        assert self.grid[r1, c1] == self.FILLED
        assert self.grid[r2, c2] == self.EMPTY
        # 1. Calculate energy cost of REMOVING from r1, c1
        # Note: We compute this based on the CURRENT grid where r2,c2 is 0.
        # So we don't worry about the 1-2 bond yet because it doesn't exist.
        energy_removal = -self._get_site_interaction_energy(r1, c1)

        # 2. Calculate energy gain of ADDING to r2, c2
        # Critically: We must ignore r1, c1 in this calculation
        # because it is effectively empty during this move.
        energy_addition = self._get_site_interaction_energy(
            r2, c2, ignore_neighbor=(r1, c1))

        return energy_removal + energy_addition

    def calculate_local_energy_change(self, r, c, new_state):
        """Calculates Delta E for changing state at single site (r,c).
        NEW_STATE is either FILLED or EMPTY."""
        assert new_state in (self.FILLED, self.EMPTY)
        current_state = self.grid[r, c]
        if current_state == new_state:
            return 0.0

        # Energy of the particle if it exists
        interaction_energy = self._get_site_interaction_energy(r, c)

        if new_state == self.FILLED:
            return interaction_energy  # Energy to add particle
        return -interaction_energy  # Energy to remove particle

    def _get_site_interaction_energy(self, r, c, ignore_neighbor=None):
        """Helper to sum interactions for a FILLED particle at r, c.
        IGNORE_NEIGHBOR can be a tuple (rn, cn) for the neighbor to be
        ignored."""
        e_sum = 0.0
        for nr, nc in self.get_neighbors(r, c):
            if (nr, nc) == ignore_neighbor:
                continue

            if self.grid[nr, nc] == self.FILLED:
                e_sum += self.energies['Na_Na']
            elif self.is_carbon[nr, nc]:
                if self.is_defect[nr, nc]:
                    e_sum += self.energies['Na_Defect']
                else:
                    e_sum += self.energies['Na_C']
        return e_sum


# --- Verification Run ---
# Example: 20 Angstrom pore (typical HC nanopore size)
model = HardCarbonPoreModel(pore_radius_angstrom=10.0,
                            na_bond_length_angstrom=3.59346)

# Test Geometry
center_idx = model.grid_width // 2
# Check a point roughly 10 Å away from center (Radius ~ 2.78 lattice units)
# At 3 units away, it should be carbon.
print(f"Center ({center_idx},{center_idx}) is Carbon? {model.is_carbon[center_idx, center_idx]}")
