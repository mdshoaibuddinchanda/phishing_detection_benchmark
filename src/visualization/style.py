"""Publication-quality matplotlib styling for research figures."""

import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    """
    Apply publication-quality styling to all matplotlib figures.
    
    This ensures figures meet journal requirements:
    - Serif typography (Times-like)
    - Consistent, explicit font sizes
    - Visible line widths and marker sizes
    - Grayscale compatible
    - Professional appearance (not default/generated-looking)
    
    Call once at module import or at the start of a plotting script.
    """
    plt.rcParams.update({
        # Typography for research papers
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        
        # Line and marker visibility
        "lines.linewidth": 2,
        "lines.markersize": 6,
        
        # Grid and axes
        "axes.grid": False,         # explicit, no automatic grid
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Colors (minimal, professional)
        "axes.edgecolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        
        # Legend
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 0.9,
    })
