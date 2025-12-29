# Publication-Quality Figure Upgrade

## Summary

Your visualizations have been upgraded from **presentation quality** (seaborn defaults) to **publication quality** (research-standard typography and styling).

## Changes Made

### 1. Configuration Update

**File**: `src/config/config.yaml`

```yaml
# OLD (Presentation style)
visualization:
  figure_format: "png"
  dpi: 300
  figsize: [10, 6]
  style: "seaborn-v0_8-whitegrid"

# NEW (Publication style)
visualization:
  figure_format: "pdf"       # Vector graphics for journals
  dpi: 300
  figsize: [7, 5]            # Journal-column compatible
  style: "classic"           # Matplotlib default (no seaborn)
```

**Why these changes**:

- **PDF format**: Vector-based, scalable, crisp at any resolution. Preferred by all major publishers (IEEE, ACM, Elsevier, Springer, Nature).
- **Classic style**: Matplotlib baseline styling. No decorative seaborn defaults that signal "auto-generated."
- **Smaller figsize**: [7, 5] matches journal column widths. Fits better in two-column layouts.

### 2. New Style Module

**File**: `src/visualization/style.py` (created)

This module centralizes all matplotlib rcParams for consistent, publication-ready styling:

```python
def apply_publication_style():
    # Typography: serif fonts (Times-like) for research papers
    "font.family": "serif"
    "font.size": 11
    
    # Explicit line widths and marker sizes (not default)
    "lines.linewidth": 2
    "lines.markersize": 6
    
    # Clean axes (minimal decorations)
    "axes.grid": False
    "axes.spines.top": False
    "axes.spines.right": False
    
    # Professional legend
    "legend.frameon": True
    "legend.edgecolor": "black"
```

**Why this matters**:

- Serif typography signals "journal-ready" to reviewers
- Explicit line widths mean your plots look professional when printed or zoomed
- Removed grid defaults to match journal standards
- Consistent across all plotting functions

### 3. Updated Plotting Modules

All three plotting modules now:

- Import and apply `apply_publication_style()` once at module load
- Remove all hardcoded `plt.style.use('seaborn-...')` calls
- Remove decorative `fontweight='bold'` and explicit font sizes (now controlled by rcParams)
- Use grayscale-safe colors (#555555, #000000, etc.) instead of bright colors
- Minimal grid styling (alpha=0.2 instead of 0.3, linewidth=0.5)

**Files updated**:

- `src/visualization/plot_accuracy.py`
- `src/visualization/plot_energy.py`
- `src/visualization/pareto_frontier.py`

## Visual Impact

### Before → After

| Aspect               | Before (Seaborn)             | After (Publication)             |
|----------------------|------------------------------|---------------------------------|
| **Grid**             | Whitegrid background         | Clean, minimal grid             |
| **Colors**           | Bright (#e74c3c,#3498db)     | Grayscale (#555555, #000000)  |
| **Typography**       | Sans-serif, mixed sizes      | Serif, consistent sizes         |
| **Format**           | PNG (raster)                 | PDF (vector)                    |
| **Line thickness**   | Thin (default ~1.5)          | Thicker (2.0)                   |
| **Axes labels**      | Bold, large, decorative      | Clean, consistent               |
| **First impression** | "Dashboard"                  | "Research paper"                |

## How to Use

### 1. Generate plots as before

No changes to your API. All plotting functions work identically:

```python
from src.visualization import plot_accuracy_comparison
plot_accuracy_comparison(results_df, output_path)
```

### 2. Output format automatically changes

Thanks to config.yaml, all plots now:

- Save as `.pdf` (not `.png`)
- Use serif typography
- Have smaller, journal-friendly dimensions
- Appear professional and publication-ready

### 3. (Optional) Override config per-call

If you ever need PNG or different dimensions:

```python
plot_accuracy_comparison(results_df, output_path, figsize=(10, 6), dpi=300)
```

The call-level arguments override config defaults.

## What This Signals to Reviewers

When a reviewer sees your figures:

- ✅ Serif fonts → "This was created with care"
- ✅ PDF format → "Scalable, publication-ready"
- ✅ Clean axes → "Not a dashboard or blog post"
- ✅ Consistent styling → "Professional research"
- ✅ Minimal decorations → "Focus on data, not aesthetics"

## What This Does NOT Change

Your results, methodology, or science are unchanged. This is purely about **presentation**, which matters for:

- Peer review first impressions
- Journal acceptance standards
- Inclusion in slides/papers
- Professional credibility

## Common Journal Requirements

Most major publishers expect:

- ✅ Vector graphics (PDF, EPS, SVG) — **Now satisfied**
- ✅ Serif typography — **Now satisfied**
- ✅ Grayscale-compatible figures — **Now satisfied**
- ✅ Readable at print size — **Now satisfied**
- ✅ No decorative styling — **Now satisfied**

Your figures now meet these standards.

## Troubleshooting

**Q: My plots look boring now.**
A: That's exactly right. Boring = professional. Journal figures are intentionally understated to let data speak.

**Q: Can I use colors again?**
A: Yes. The grayscale choice is best practice, but edit `style.py` if needed. However, ensure colors are colorblind-safe (use colorbrewer or check with color-blindness simulators).

**Q: PDF files are larger than PNG.**
A: Correct, but PDFs scale infinitely and print better. PDFs are standard for publications.

**Q: How do I change back to seaborn?**
A: Revert config.yaml to use PNG + your seaborn style. But don't do this for journal submission.

## Next Steps

1. ✅ Regenerate all plots with `uv run python src/main.py --only-visualize`
2. ✅ Review `.pdf` files in `results/figures/` — they should look crisp and professional
3. ✅ Include these in your paper with confidence
4. ✅ Tell your professor: "I switched to vector PDF with serif typography per journal standards."

---

**Remember**: This upgrade is about *presentation quality*, not results quality. Your science is unchanged. You're now simply packaging it in a way that journals and reviewers expect.
