"""Monte Carlo simulation package for hard-carbon pore filling in Na-ion batteries.

Provides:
- HardCarbonPoreModel: 2D triangular-lattice MC model
- run_simulation / run_convergence_simulation / run_voltage_sweep_simulation
- get_formation_energies: sequential pore filling energetics
- analyze_pore_filling: parameter scan result analysis
- plot_* functions: publication-quality figures
- generate_voltage_points / generate_scan_commands: scan setup
"""

from mcpore.core import (
    HardCarbonPoreModel,
    save_model_svg,
    visualize_model,
    save_timeseries_csv,
    CSV_GZ_SUFFIX,
)

from mcpore.simulation import (
    run_simulation,
    run_voltage_sweep_simulation,
    run_convergence_simulation,
    run_0K_min,
    replay_simulation,
    summarize_snapshots,
)

from mcpore.analysis import (
    get_formation_energies,
    plot_filled_pore_energy,
    plot_filling_barriers,
    plot_formation_energies,
    plot_filling_voltages,
    plot_voltages,
    analyze_pore_filling,
)

from mcpore.scan import (
    generate_voltage_points,
    generate_scan_commands,
)
