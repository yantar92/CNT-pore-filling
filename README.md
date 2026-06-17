

# Hard‑Carbon Pore‑Filling Monte Carlo Simulation – Software Documentation

This package provides a Metropolis Monte Carlo (MC) simulator for
studying sodium‑ion filling of nanopores in hard‑carbon anodes for
sodium‑ion batteries. It implements a 2‑dimensional
triangular‑lattice model with circular pores, Na‑Na and Na‑C
interactions, and explicit carbon‑wall defects. The code is designed
for high‑throughput parameter scanning (pore radius, voltage, defect
density, defect‑adsorption energy) and includes tools for automated
scan generation, parallel execution, and publication‑ready analysis.


## Overview of scripts

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Script</th>
<th scope="col" class="org-left">Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>mc‑pore.py</code></td>
<td class="org-left">Core simulation engine. Runs a single MC simulation for a given parameter set.</td>
</tr>

<tr>
<td class="org-left"><code>generate_scan.py</code></td>
<td class="org-left">Generates a command list for a full parameter scan (voltages, radii, defect probabilities, Na‑defect energies).</td>
</tr>

<tr>
<td class="org-left"><code>run.sh</code></td>
<td class="org-left">Example wrapper that calls <code>generate_scan.py</code> and executes the scan with GNU Parallel.</td>
</tr>

<tr>
<td class="org-left"><code>analyze.py</code></td>
<td class="org-left">Loads the aggregated results (<code>results.csv</code>), averages replicates, and produces a set of plots (filling fraction vs. radius, coloured by voltage) for each defect‑probability / Na‑defect‑energy combination.</td>
</tr>
</tbody>
</table>


## Installation & dependencies

The simulation requires Python 3 with `numpy`, `matplotlib`,
`pandas`, and `seaborn` (for plot styling). GNU Parallel is needed
for running the parameter scan in parallel.

    # Install Python dependencies
    pip install numpy matplotlib pandas seaborn
    
    # On a typical Linux system, install GNU Parallel via your package manager
    # (e.g., on Gentoo: emerge sys‑process/parallel)

All scripts are standalone and can be placed in the same directory.


## Quick start: running a single simulation

    python mc‑pore.py --voltage 0.1 --radius 10.0 --defect_probability 0.174 \
                        --energy_na_defect -1.53 --steps 1000000 --csv --quiet --seed 12345

This will output a single CSV line to stdout containing all simulation parameters and the final filling fraction. To see live visualisation, omit `--csv` and `--quiet` and add `--visualize`.


## Parameter‑scan workflow

1.  **Generate the command list** (adjust `--replicates` and `--steps` as needed):
    
        python generate_scan.py --replicates 15 --steps 1000000 --output commands.txt
    
    The script prints an estimate of the total number of simulations and wall‑time.

2.  **Run the scan with GNU Parallel** (here using 96 cores):
    
        cat commands.txt | parallel -j 96 --line‑buffer > results.csv
    
    Each simulation outputs one CSV line; the `--line‑buffer` ensures lines are flushed immediately.

3.  **After completion**, the file `results.csv` contains all simulation results, one row per run.


## Output format (`results.csv`)

The CSV file has the following columns (in order):

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Column</th>
<th scope="col" class="org-left">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>voltage</code></td>
<td class="org-left">Applied voltage (V) relative to bulk Na.</td>
</tr>

<tr>
<td class="org-left"><code>radius</code></td>
<td class="org-left">Pore radius (Å).</td>
</tr>

<tr>
<td class="org-left"><code>defect_probability</code></td>
<td class="org-left">Atomic defect concentration (fraction of carbon sites that are defective).</td>
</tr>

<tr>
<td class="org-left"><code>defect_placement</code></td>
<td class="org-left">Always <code>"surface"</code> (defects placed only on pore‑surface carbon sites).</td>
</tr>

<tr>
<td class="org-left"><code>energy_na_defect</code></td>
<td class="org-left">Na‑defect bond energy (eV, vs. vacuum).</td>
</tr>

<tr>
<td class="org-left"><code>energy_na_na</code></td>
<td class="org-left">Na‑Na bond energy (eV).</td>
</tr>

<tr>
<td class="org-left"><code>energy_na_c</code></td>
<td class="org-left">Na‑C bond energy (eV).</td>
</tr>

<tr>
<td class="org-left"><code>temperature</code></td>
<td class="org-left">Temperature (K).</td>
</tr>

<tr>
<td class="org-left"><code>steps</code></td>
<td class="org-left">Total MC steps requested.</td>
</tr>

<tr>
<td class="org-left"><code>seed</code></td>
<td class="org-left">Random seed used.</td>
</tr>

<tr>
<td class="org-left"><code>final_filling</code></td>
<td class="org-left">Filling fraction (0–1) at the end of the simulation.</td>
</tr>

<tr>
<td class="org-left"><code>equilibrium_reached</code></td>
<td class="org-left">Boolean (<code>True=/=False</code>) indicating whether the filling fraction stabilised before the step limit.</td>
</tr>

<tr>
<td class="org-left"><code>mcs</code></td>
<td class="org-left">Actual MC steps executed (normalised by number of lattice sites).</td>
</tr>

<tr>
<td class="org-left"><code>n_valid_sites</code></td>
<td class="org-left">Number of lattice sites inside the pore.</td>
</tr>

<tr>
<td class="org-left"><code>n_surface_sites</code></td>
<td class="org-left">Number of pore‑surface sites (where GCMC insertion/removal is allowed).</td>
</tr>

<tr>
<td class="org-left"><code>default_p_gcmc</code></td>
<td class="org-left">Default probability of attempting a GCMC move (surface‑sites / valid‑sites).</td>
</tr>

<tr>
<td class="org-left"><code>mu</code></td>
<td class="org-left">Chemical potential (eV) derived from voltage and Na‑Na energy.</td>
</tr>
</tbody>
</table>


## Analysis and visualisation

Run the analysis script to average replicates and generate publication‑style plots:

    python analyze.py results.csv --output‑dir ./plots

This will:

-   Load `results.csv` and exclude any simulations that did not reach equilibrium.
-   Group results by (`voltage`, `radius`, `defect_probability`, `energy_na_defect`) and compute mean ± standard deviation across replicates.
-   Create one PNG plot per (`defect_probability`, `energy_na_defect`) combination, showing filling fraction vs. pore radius with separate lines (coloured by voltage). Each plot includes a colour‑bar for voltage and a text box listing constant simulation parameters.
-   Save all plots to the specified output directory (default: current directory).

Command‑line options:

-   `--output‑dir DIR` – directory for PNG files (created if missing).
-   `--show‑plots` – display plots interactively (instead of only saving).
-   `--quiet` – suppress progress messages.


## Model description (summary)

The simulation implements a 2D triangular lattice of Na sites with a circular pore. Carbon‑wall sites occupy lattice positions outside the pore radius; a fraction of these wall sites are marked as defects (`DEFECT`) with a stronger Na‑adsorption energy. The remaining wall sites are ordinary carbon (`CARBON`). Sites inside the pore are either empty (`EMPTY`) or occupied by Na (`NA`).

**Monte Carlo moves:**

-   **Diffusion**: a Na atom swaps with an adjacent empty site (Metropolis acceptance based on energy change).
-   **Grand‑canonical Monte Carlo (GCMC)**: insertion or removal of a Na atom at a pore‑surface site (adjacent to a carbon/defect). Acceptance depends on the interaction energy of the new/adatom and the chemical potential `μ = –voltage + 3·E_Na‑Na`.

**Key energy parameters (default values from optB88‑vdW DFT):**

-   `E_Na‑Na` = –0.35 eV/bond
-   `E_Na‑C` = –0.32 eV/bond
-   `E_Na‑defect` = –1.77 eV/bond (vs. vacuum)

**Defect placement:** The `defect_probability` is interpreted as the atomic concentration of defective carbon atoms. Defects are placed exactly on the pore‑surface carbon sites (i.e., those adjacent to at least one empty pore site). This ensures defects are always accessible to Na.

**Equilibrium detection:** The simulation monitors the filling‑fraction slope over a moving window (default 10 000 samples). When the slope falls below `1×10⁻⁵` per MC step and at least 10 000 MC steps have been performed, the run is considered to have reached equilibrium and stops early.


# License and citation

This software is provided under the MIT License. If you use this
code in your research, please acknowledge the authors.

