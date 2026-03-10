import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

FONT_SIZE = 40
LINE_STYLES = ['-', '--', '-.', ':']
LINE_WIDTH  = 8

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": 40,
    "figure.titlesize": FONT_SIZE,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.8,
    "grid.linestyle": "--",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.ymargin": 0.1
})

def plot_flexible(
    x_val: np.ndarray,
    y_vals: list,
    labels: list,
    x_label: str,
    y_units: list,
    save_name: str,
    ylims=None,
    xlims=None,
    fig_size=32,
    show_plot=False):
    """
    Plotting function that handles multiple subplots and multiple lines per subplot.

    Parameters:
    - x_val:      1D array of x values (shared across all subplots)
    - y_vals:     list of lists of y values, one list per subplot
    - labels:     list of lists of labels, matching the structure of y_vals
    - x_label:    label for the x axis (shared across all subplots)
    - y_units:    list of y axis labels, one per subplot
    - save_name:  base name for saving the plot (without extension)
    - ylims:      list of (min, max) tuples for y axis limits, one per subplot
    - xlims:      (min, max) tuple for x axis limits
                  [{"x": 1.0, "color": "gray", "linestyle": "--", "label": "t=1s"}]
    - fig_size:   figure width in inches
    - show_plot:  whether to display the plot after saving
    """
    # --- auto-wrap flat lists into nested lists ---
    if not isinstance(y_vals[0], (list, np.ndarray)) or (
        isinstance(y_vals[0], np.ndarray) and y_vals[0].ndim == 1
    ):
        y_vals = [[y] for y in y_vals]

    if not isinstance(labels[0], list):
        labels = [[l] for l in labels]

    subplots = len(y_vals)

    # --- validate inputs ---
    assert len(y_vals)  == subplots, "y_vals must have one list per subplot"
    assert len(labels)  == subplots, "labels must have one list per subplot"
    assert len(y_units) == subplots, "y_units must have one entry per subplot"

    if ylims is None:
        ylims = [None] * subplots
    assert len(ylims) == subplots, "ylims must have one entry per subplot (or None)"

    # --- create folder ---
    save_path = Path("plots")
    save_path.mkdir(exist_ok=True)

    # --- plot ---
    fig, axes = plt.subplots(subplots, 1, figsize=(fig_size, 9 * subplots), sharex=True)
    if subplots == 1:
        axes = [axes]

    for ax, y_list, label_list, y_unit, ylim in zip(axes, y_vals, labels, y_units, ylims):
        for i, (y, label) in enumerate(zip(y_list, label_list)):
            ax.plot(x_val, y, label=label,
                    linewidth=LINE_WIDTH,
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)])
        ax.set_ylabel(y_unit)
        ax.legend()
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=1, linestyle='--')
        ax.grid(True, which='minor', alpha=0.5, linestyle=':')
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    if xlims is not None:
        axes[0].set_xlim(xlims[0], xlims[1])

    axes[-1].set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig(save_path / f"{save_name}.png")
    if show_plot:
        plt.show()
    plt.close()