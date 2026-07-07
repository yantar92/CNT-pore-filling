

# Hard‑Carbon Pore‑Filling Monte Carlo Simulation

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


## Implementation details

See [1.8](#org783cd52) for the semi‑grand‑canonical ensemble,
move set, Metropolis acceptance, and kinetic interpretation.

**Default energy parameters (optB88‑vdW DFT):**

-   `E_Na‑Na` = –0.35 eV/bond
-   `E_Na‑C` = –0.32 eV/bond
-   `E_Na‑defect` = –1.77 eV/bond (vs. vacuum)

**Defect placement:** The `defect_probability` is interpreted as the
atomic concentration of defective carbon atoms. With
`defect_placement`'surface'= (the default), an exact fraction of
pore‑surface carbon sites (those adjacent to at least one empty
pore site) is randomly marked as `DEFECT`. This ensures defects are
always accessible to Na.

**Equilibrium detection:** The simulation monitors the
filling‑fraction slope over a moving window (default 10 000
samples). When the slope falls below `1×10⁻⁵` per MC step and at
least 10 000 MC steps have been performed, the run is considered to
have reached equilibrium and stops early.


## Physical background


### Semi-grand-canonical ensemble

Na filling in a hard‑carbon pore is modeled as a lattice gas in
contact with a metallic Na electrode reservoir. The pore contains `M`
valid sites on a triangular lattice, with binary occupation
variables. A subset of `M_s` pore‑surface sites, those adjacent to
carbon‑wall atoms, exchanges Na with the electrode. The reservoir
fixes the chemical potential `mu`, which is related to the applied
voltage `V` (relative to bulk Na) by

    mu = -V + 3 E_Na‑Na

assuming each Na in the pore has up to 3 in‑plane Na‑Na bonds on
average (2D coordination).

The target distribution is the semi‑grand‑canonical ensemble

    pi(n) ∝ exp[-beta (E(n) - mu N(n))]

where `E(n)` is the pore interaction energy, `N(n)` is the number of
Na atoms in configuration `n`, and `beta = 1/(k_B T)`. The energy
comprises three pairwise terms summed over nearest neighbors:

    E(n) = E_Na‑Na(n) + E_Na‑C(n) + E_Na‑defect(n)


### Move set and Metropolis acceptance

The physical move set consists of two classes:

-   ****Diffusion**:** A random valid pore site is selected, followed by a
    random nearest‑neighbor direction on the triangular lattice. If the
    selected site contains Na and the target is an empty valid pore
    site, a local hop is proposed. Otherwise the attempt is a null
    move. Diffusion conserves `N`.

-   ****Surface insertion/deletion (GCMC)**:** A random surface‑exchange
    site is selected. If the site is empty, Na insertion is proposed;
    if occupied, Na deletion is proposed.

Both proposal schemes are symmetric because they sample from fixed
geometrical site sets, not from the set of currently possible moves.
The Metropolis acceptance rule is

    A = min[1, exp(-beta (Delta E - mu Delta N))]

which gives `A_diff = min[1, exp(-beta Delta E)]` for diffusion,
`A_ins = min[1, exp(-beta(Delta E - mu))]` for insertion, and
`A_del = min[1, exp(-beta(Delta E + mu))]` for deletion.

Null moves leave the configuration unchanged and do not affect the
stationary distribution. The algorithm therefore satisfies detailed
balance with respect to the semi‑grand distribution and samples
equilibrium pore‑filling statistics.


### Diffusion/GCMC move ratio

Each MC step selects a move class with probability `p_gcmc` (GCMC) or
`1-p_gcmc` (diffusion). The move‑class probability multiplies both
forward and reverse proposal probabilities for that class and cancels
in the Metropolis‑Hastings ratio. Consequently, `p_gcmc` does *not*
affect the equilibrium distribution. It changes how the Markov chain
explores configuration space but not the ensemble to which it
converges.

For equilibrium calculations, the ratio can be optimized for sampling
efficiency (e.g., by minimizing autocorrelation times of slow
observables such as `N` or filling fraction). Too small a `p_gcmc`
slows exploration of different filling levels; too large a `p_gcmc`
gives insufficient internal relaxation by diffusion.


### Kinetic interpretation

The same algorithm admits a kinetic interpretation when
`p_gcmc` follows from the relative attempt frequencies of
microscopic clocks. Assign one diffusion attempt clock to each valid
pore site (frequency `nu_diff`) and one exchange attempt clock to
each surface site (frequency `nu_gcmc`). The probability that the
next attempted event is a surface‑exchange event is

    p_gcmc = (r M_s) / (M + r M_s) ,  r = nu_gcmc / nu_diff

The default setting `p_gcmc = M_s / (M + M_s)` corresponds to `r=1`,
i.e., equal elementary attempt frequencies per diffusion site and per
surface exchange site. This isolates the effects of pore geometry,
surface/volume scaling (`p_gcmc ~ 1/R` for large pores), thermodynamic
driving force, and lattice crowding under a controlled reference
assumption.

Under this interpretation, accepted moves follow Metropolis rates
proportional to `min[1, exp(-beta(Delta E - mu Delta N))]`, which obey
local detailed balance and relax to the correct semi‑grand equilibrium
distribution.


### Limitations

The Metropolis kinetic model is not a replacement for
transition‑state‑theory (TST) kinetics when absolute physical times
or barrier‑controlled rates are required:

-   ****No explicit barriers**:** The acceptance rule uses only initial and
    final state energies. It does not include migration barriers,
    charge‑transfer barriers, desolvation barriers, or
    interface‑crossing barriers.

-   ****Downhill moves**:** All downhill moves in `Delta E - mu Delta N`
    occur at the same attempt‑limited rate, independent of how strongly
    downhill they are. A TST model would assign different rates based on
    saddle‑point barriers.

-   ****Absolute time**:** Absolute physical time requires calibration of
    `nu_diff`, `nu_gcmc`, or `r` against independent data (diffusion
    coefficients, measured rates, atomistic barriers).

-   ****Size‑dependent barriers**:** The model assumes the same elementary
    attempt‑frequency ratio for all pore sizes. If activation barriers
    vary systematically with pore size (e.g., radius‑dependent diffusion
    barriers near curved walls), the kinetic size dependence may be
    incomplete.

Despite these limitations, the model provides internally consistent
and physically interpretable pore‑size trends when those trends are
dominated by geometry, thermodynamics, and lattice crowding rather
than by unknown changes in activation barriers. The size dependence
of `p_gcmc` (~1/R) captures the intuitive surface/volume scaling:
larger pores have proportionally fewer exchange sites per interior
diffusion site. The model is rigorous as an equilibrium
semi‑grand‑canonical sampler and a well‑defined Metropolis kinetic
model satisfying local detailed balance.


# License and citation

This software is provided under the MIT License. If you use this
code in your research, please acknowledge the authors.

