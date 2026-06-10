import numpy as np
import matplotlib.pyplot as plt

def plot_barcode(intervals, max_limit=None, ax=None, show=True):
    """
    Plots the persistence intervals as a barcode.

    Parameters:
    -----------
    intervals : dict
        Dictionary of intervals, mapping dim -> list of (birth, death).
    max_limit : float, optional
        The value to use for plotting infinite death times. If None, it will be 
        calculated automatically based on finite death times.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure is created.
    show : bool, default=True
        Whether to show the plot immediately.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Determine a good max limit for infinite intervals
    finite_deaths = []
    for dim, d_intervals in intervals.items():
        for birth, death in d_intervals:
            if death != float('inf'):
                finite_deaths.append(death)
    
    if not finite_deaths:
        auto_max_limit = 1.0
    else:
        auto_max_limit = max(finite_deaths) * 1.2

    if max_limit is None:
        max_limit = auto_max_limit

    # Setup colors for different dimensions (harmonious modern palette)
    colors = {
        0: '#FF6B6B',  # Coral
        1: '#4D96FF',  # Sky Blue
        2: '#6BCB77',  # Emerald Green
        3: '#FFD93D',  # Warm Yellow
    }
    default_color = '#9E72C3'  # Lavender for higher dimensions

    # Count total number of bars to position them on y-axis
    total_bars = sum(len(d_intervals) for d_intervals in intervals.values())
    
    y_idx = 0
    y_ticks = []
    y_labels = []
    
    # Grid lines & styling
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E2E8F0', zorder=0)

    # Plot dimensions sequentially (high dimension at the top)
    for dim in sorted(intervals.keys()):
        d_intervals = intervals[dim]
        if not d_intervals:
            continue
            
        color = colors.get(dim, default_color)
        dim_start_y = y_idx
        
        for birth, death in d_intervals:
            is_inf = (death == float('inf'))
            display_death = max_limit if is_inf else death
            
            # Plot the bar
            ax.plot([birth, display_death], [y_idx, y_idx], 
                    color=color, linewidth=2.5, solid_capstyle='round', zorder=3)
            
            # If infinite, plot an arrowhead or marker at the end
            if is_inf:
                ax.scatter(display_death, y_idx, color=color, marker='>', s=40, zorder=4)
                
            y_idx += 1
            
        # Place a tick label in the middle of the block for this dimension
        dim_end_y = y_idx - 1
        y_ticks.append((dim_start_y + dim_end_y) / 2)
        y_labels.append(f"$H_{dim}$")

    # Add a horizontal line representing the infinity limit if there are infinite bars
    has_infinite = any(death == float('inf') for d_intervals in intervals.values() for _, death in d_intervals)
    if has_infinite:
        ax.axvline(x=max_limit, color='#A0AEC0', linestyle=':', linewidth=1.5, label='Infinity threshold')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='semibold')
    ax.set_xlabel("Filtration Value ($\epsilon$)", fontsize=11, fontweight='medium')
    ax.set_title("Persistence Barcode", fontsize=14, fontweight='bold', pad=15)
    
    # Hide top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors.get(dim, default_color), lw=3, label=f"$H_{dim}$ features")
        for dim in sorted(intervals.keys()) if len(intervals[dim]) > 0
    ]
    if has_infinite:
        legend_elements.append(Line2D([0], [0], color='#A0AEC0', linestyle=':', lw=1.5, label='$\infty$ marker'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='best', frameon=True, facecolor='white', edgecolor='#E2E8F0')

    # Buffer for y-axis
    ax.set_ylim(-0.5, total_bars - 0.5)
    ax.set_xlim(0, max_limit * 1.05)

    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_persistence_diagram(intervals, max_limit=None, ax=None, show=True):
    """
    Plots the persistence intervals as a persistence diagram (birth vs death).

    Parameters:
    -----------
    intervals : dict
        Dictionary of intervals, mapping dim -> list of (birth, death).
    max_limit : float, optional
        The value to use for plotting infinite death times. If None, it will be 
        calculated automatically based on finite death times.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure is created.
    show : bool, default=True
        Whether to show the plot immediately.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Determine a good max limit for infinite intervals
    finite_deaths = []
    for dim, d_intervals in intervals.items():
        for birth, death in d_intervals:
            if death != float('inf'):
                finite_deaths.append(death)
    
    if not finite_deaths:
        auto_max_limit = 1.0
    else:
        auto_max_limit = max(finite_deaths) * 1.2

    if max_limit is None:
        max_limit = auto_max_limit

    # Setup colors and markers for different dimensions
    colors = {
        0: '#FF6B6B',  # Coral
        1: '#4D96FF',  # Sky Blue
        2: '#6BCB77',  # Emerald Green
        3: '#FFD93D',  # Warm Yellow
    }
    markers = {
        0: 'o',  # Circle
        1: '^',  # Triangle
        2: 's',  # Square
        3: 'D',  # Diamond
    }
    default_color = '#9E72C3'
    default_marker = 'p'

    # Background styling
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E2E8F0', zorder=0)

    # Plot the diagonal line y = x
    ax.plot([0, max_limit * 1.05], [0, max_limit * 1.05], color='#A0AEC0', linestyle='--', linewidth=1.5, zorder=1)

    # Plot a horizontal dashed line for infinity
    has_infinite = any(death == float('inf') for d_intervals in intervals.values() for _, death in d_intervals)
    if has_infinite:
        ax.axhline(y=max_limit, color='#A0AEC0', linestyle=':', linewidth=1.5, zorder=1)

    # Plot points for each dimension
    for dim in sorted(intervals.keys()):
        d_intervals = intervals[dim]
        if not d_intervals:
            continue
            
        color = colors.get(dim, default_color)
        marker = markers.get(dim, default_marker)
        
        births = []
        deaths = []
        inf_births = []
        
        for birth, death in d_intervals:
            if death == float('inf'):
                inf_births.append(birth)
            else:
                births.append(birth)
                deaths.append(death)
                
        # Plot finite persistence points
        if births:
            ax.scatter(births, deaths, color=color, marker=marker, s=60, 
                       alpha=0.8, edgecolors='none', label=f"$H_{dim}$", zorder=3)
            
        # Plot infinite persistence points on the infinity line
        if inf_births:
            ax.scatter(inf_births, [max_limit] * len(inf_births), color=color, 
                       marker=marker, s=80, alpha=0.9, edgecolors='black', 
                       linewidth=1.0, label=f"$H_{dim}$ ($\infty$)", zorder=4)

    ax.set_xlabel("Birth", fontsize=11, fontweight='medium')
    ax.set_ylabel("Death", fontsize=11, fontweight='medium')
    ax.set_title("Persistence Diagram", fontsize=14, fontweight='bold', pad=15)
    
    # Set limits
    ax.set_xlim(-max_limit*0.02, max_limit * 1.05)
    ax.set_ylim(-max_limit*0.02, max_limit * 1.05)
    
    # Custom y ticks to show infinity label if applicable
    if has_infinite:
        current_ticks = ax.get_yticks()
        # Filter ticks that are near the infinity limit to prevent overlaps
        ticks = [t for t in current_ticks if t < max_limit * 0.95]
        ticks.append(max_limit)
        labels = [f"{t:.2f}" if t != max_limit else "$\infty$" for t in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

    # Hide top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')

    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#E2E8F0')

    # Force square aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if show:
        plt.show()
    return ax
