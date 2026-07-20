import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

TEMPERATURE = 2000
# DEFECT_CONCENTRATION = 0.5
DEFECT_CONCENTRATION = 0
# DEFECT_CONCENTRATION = 0.174

# --- Publication style (good for ~half A4 width figure) ---
mpl.rcParams.update({
    "figure.figsize": (4.13, 3.10),   # half A4 width, 4:3 ratio
    "figure.dpi": 300,
    "savefig.dpi": 600,

    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    "lines.linewidth": 1.2,
    "lines.markersize": 3.2,

    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})


os.chdir('/home/yantar92/helios/0000.lumi/2026.Na.pore-filling.MC/11.voltage_sweep')

df = pd.read_csv(
    'results.csv',
    names=[
        "voltage", "radius",
        "defect_probability", "defect_placement",
        "energy_na_defect", "energy_na_na", "energy_na_c",
        "temperature", "steps", "seed",
        "final_filling", "equilibrium_reached", "mcs",
        "n_valid_sites", "n_surface_sites", "default_p_gcmc",
        "mu", "fill_mcs"
    ],
)

# Only per voltage simulations are averaged out
df = df[:90000]
# df = df[90000:]

# This is limiting to calculations where we sample over fixed pore defect distribution
df = df[np.isclose(df['temperature'], TEMPERATURE)]
df = df[np.isclose(df['defect_probability'], DEFECT_CONCENTRATION)]
radii = df['radius'].unique()


cmap = mpl.colormaps.get_cmap('tab20')
# cmap = distinctipy.get_colormap(distinctipy.get_colors(len(radii)))
# norm = mpl.colors.Normalize(vmin=min(radii), vmax=max(radii), )
norm = mpl.colors.LogNorm(vmin=min(radii), vmax=max(radii))


fig, ax = plt.subplots(constrained_layout=True)

radii = np.array(sorted(df["radius"].unique()))

# Perceptually-uniform colormap + log scaling across radii
cmap = mpl.colormaps["tab20"]
# norm = mpl.colors.LogNorm(vmin=radii.min(), vmax=radii.max())

sel = np.r_[radii[radii < 10.0], radii[radii >= 10.0][::2]]
# sel = radii[np.isclose(radii, 20)]

for r, color in zip(sel, cmap.colors):
    tem = df[np.isclose(df["radius"], r)]

    g = tem.groupby("voltage")["final_filling"]
    avg = g.median()
    avg_min = g.min()
    avg_max = g.max()

    volts = tem.groupby("voltage")["voltage"].median()

    # ax.plot(
    #     tem['final_filling'], tem['voltage'],
    #     marker="o",
    #     color=color,
    #     markerfacecolor="white",
    #     markeredgewidth=0.8,
    #     label=f'{r:.0f}Å'
    # )
    ax.plot(
        avg, volts,
        marker="o",
        color=color,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label=f'{r:.0f}Å'
    )
    ax.fill_betweenx(
        volts, avg_min, avg_max,
        color=color,
        alpha=0.2
    )

# Axes styling
ax.set_title(f"CV plots (via MC) for T={TEMPERATURE}K, defects={DEFECT_CONCENTRATION}")
ax.set_xlabel("Filling ratio (%)")
ax.set_ylabel("Voltage (V)")
ax.set_xlim(-5, 105)
# ax.set_ylim(0, 0.1)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.minorticks_on()

# Light grid (optional, but helps readability at small size)
ax.grid(True, which="major", alpha=0.18, linewidth=0.6)

ax.legend(ncols=4)

# sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.06)
# cbar.set_label("Pore radius (Å)")
# cbar.ax.tick_params(direction="out")

fig.savefig(f"voltage_vs_filling_T{TEMPERATURE}_c{DEFECT_CONCENTRATION}.svg")
fig.savefig(f"voltage_vs_filling_T{TEMPERATURE}_c{DEFECT_CONCENTRATION}.png")
plt.show()


# fig, ax = plt.subplots(1, 1)
# radii = sorted(radii)
# radii = np.array(radii)
# for r in list(radii[radii < 10.0]) + list(radii[radii >= 10.0])[::3]:
#     tem = df[np.isclose(df['radius'], r)]
#     avg_filling = tem.groupby('voltage')['final_filling'].median()
#     avg_filling_min = tem.groupby('voltage')['final_filling'].min()
#     avg_filling_max = tem.groupby('voltage')['final_filling'].max()
#     voltages = tem.groupby('voltage')['voltage'].mean()
#     # ax.plot(tem['final_filling'], tem['voltage'], 'o-', label=str(r))
#     ax.plot(avg_filling, voltages, 'o-', label=str(r), color=cmap(norm(r)))
#     # ax.fill_betweenx(voltages, avg_filling_min, avg_filling_max, color=cmap(norm(r)), alpha=0.2)
# ax.legend()
# ax.set_xlabel('Filling ratio, %')
# ax.set_ylabel('Voltage, V')
# plt.show()
