# Research Benchmark Tables Guide

## What Gets Generated

After running `uv run python src/main.py`, you'll have **3 versions of your benchmark table**:

### 1. **benchmark_table.csv** (Main table - for viewing)
Easy-to-read CSV with formatted numbers. Use this to:
- Share with your professor
- Include in presentations
- Quick reference

### 2. **benchmark_table.md** (GitHub/documentation)
Markdown format for GitHub README or documentation.

```
| Model           | Accuracy ↑ | F1 ↑   | Latency (ms) ↓ | Energy (kWh) ↓ | CO₂ (g) ↓ | Size (MB) ↓ |
|-----------------|------------|--------|----------------|-----------------|-----------|-------------|
| DistilBERT      | 0.9234     | 0.9145 | 12.3           | 0.0015          | 2.3       | 268        |
| MobileBERT      | 0.9112     | 0.9045 | 8.2            | 0.0008          | 1.2       | 93         |
```

### 3. **benchmark_table.tex** (LaTeX - for papers)
Ready to copy-paste into your paper:

```latex
\begin{table}[ht]
\centering
\caption{Benchmark results: Phishing email detection (test set).}
\label{tab:benchmark}
\small
\begin{tabular}{lrrrrrr}
\toprule
Model & Accuracy ↑ & F1 ↑ & Latency (ms) ↓ & ... \\
\midrule
DistilBERT & 0.9234 & 0.9145 & 12.3 & ... \\
...
\bottomrule
\end{tabular}
\end{table}
```

## How to Use in Your Paper

### In Main Text:
```latex
\documentclass{article}
...
\begin{document}

% Copy the benchmark_table.tex content directly here:
\input{tables/benchmark_table.tex}

Table~\ref{tab:benchmark} shows the benchmark results...

\end{document}
```

### Citation in Caption:
```
"Table 1. Benchmark results on phishing email detection (test set).
Metrics marked with ↑ indicate higher is better;
↓ indicates lower is better."
```

## Supplementary Tables (in Appendix)

Three additional tables are generated for detailed breakdown:

### supplementary_full_metrics.csv
Complete raw results with all columns. Use in appendix if needed.

### supplementary_performance.csv
Performance metrics only (Accuracy, Precision, Recall, F1).

### supplementary_efficiency.csv
Efficiency metrics only (Latency, Energy, CO₂, Model Size).

## What Your Professor Will See

✅ **One consolidated benchmark table** (not a data dump)
✅ **Clear metric grouping** (performance vs efficiency)
✅ **Visual clarity** (↑↓ indicators show what's "better")
✅ **Professional formatting** (publication-ready)
✅ **Easy to reference** (one table = all trade-offs)

## Key Features

### ↑ vs ↓ indicators
- **↑** = higher is better (Accuracy, F1, Precision, Recall)
- **↓** = lower is better (Latency, Energy, CO₂, Size)

### Metric Organization
**Performance Section:**
- Accuracy (classification accuracy)
- F1 (balanced performance metric)
- Precision (false positive control)
- Recall (false negative control)

**Efficiency Section:**
- Latency (inference speed)
- Energy (power consumption)
- CO₂ (carbon footprint)
- Size (model weight)

### How to Bold Best Values (optional)
If you want to bold the best value in each column, edit the LaTeX:

```latex
% Before:
DistilBERT & 0.9234 & 0.9145 & 12.3 & ... \\

% After (best accuracy bolded):
DistilBERT & \textbf{0.9234} & 0.9145 & 12.3 & ... \\
```

## Talking Points for Your Professor

"I've created a publication-quality benchmark table with:
- Clear performance vs efficiency trade-offs
- Industry-standard direction indicators (↑↓)
- Professional formatting suitable for journals
- Detailed supplementary tables in the appendix for full transparency

The main table gives a one-glance overview; supplementary tables provide reproducibility."

## Common Questions

**Q: Why are there multiple formats?**
A: Different contexts use different formats. LaTeX is for your paper, CSV is for sharing/data, Markdown is for GitHub.

**Q: Can I modify the table?**
A: Yes! The script is designed to be modified. Adjust metric selection, formatting, or grouping as needed.

**Q: What if I only want certain models?**
A: Edit `generate_benchmark_table.py` to filter rows before formatting.

**Q: Should I include all metrics in the main table?**
A: Best practice is to show the most important ones (Accuracy + F1 + one efficiency metric). Put the rest in supplementary.
