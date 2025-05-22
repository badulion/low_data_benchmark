import pandas as pd
import os
import natsort

# Load CSV
csv_file = 'results/aggregated_data.csv'  # Replace with your actual CSV path

output_dir = 'results/latex_tables'  # Directory to save separate tables



equation_map = {
    'advection': 'A',
    'burgers': 'B',
    'gasdynamics': 'GD',
    'kuramotosivashinsky': 'KS',
    'reactiondiffusion': 'RD',
    'wave': 'W',
}

metrics = ['test_loss-1', 'test_loss-16', 'test_time', 'train_step_ms_mean']

# Load and clean data
df = pd.read_csv(csv_file)
df['equation'] = df['equation'].map(lambda x: equation_map.get(x.lower(), x))
df['domain'] = df['domain']

# Filter required columns
df_filtered = df[['model', 'domain', 'equation'] + metrics].copy()

# Helper to format values
def format_val(val, metric):
    if pd.isna(val):
        return "-"
    if 'loss' in metric and val > 2:
        return r"\text{div}"
    try:
        return rf"\num{{{val:.2e}}}"
    except:
        return str(val)

for metric in metrics:
    # Pivot table
    pivot = df_filtered.pivot(index=['model', 'domain'], columns='equation', values=metric)
    pivot = pivot.sort_values(by='domain', ascending=False)

    # Create mask for underlining
    underline_mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for eq in pivot.columns:
        col = pivot[eq]
        if 'loss' in metric:
            valid = col[col <= 2].dropna()
        else:
            valid = col.dropna()
        if not valid.empty:
            min_val = valid.min()
            underline_mask.loc[col == min_val, eq] = True

    # Format all entries
    formatted = pivot.copy()
    for eq in pivot.columns:
        for idx in pivot.index:
            val = pivot.at[idx, eq]
            fval = format_val(val, metric)
            if underline_mask.at[idx, eq] and fval not in [r"\text{div}", "-"]:
                fval = rf"\underline{{{fval}}}"
            formatted.at[idx, eq] = fval

    # Build LaTeX table
    col_format = "ll" + rf"c@{{\hskip 0.3cm}}" * len(pivot.columns)  # model + domain + equations
    header_cols = ['model', 'domain'] + list(pivot.columns)
    header_line = ' & '.join(header_cols) + r" \\"

    rows = []
    for (model, domain), row in formatted.iterrows():
        row_vals = [model, domain] + [row[eq] for eq in pivot.columns]
        rows.append(' & '.join(row_vals) + r" \\")

    latex = rf"""
\begin{{table}}[ht]
\centering
\caption{{Values of metric \texttt{{{metric}}} for all models and equations. Best values per equation are \underline{{underlined}}. Divergent losses (> 2) are \text{{div}}.}}
\begin{{tabular}}{{{col_format}}}
\toprule
{header_line}
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\label{tab:all_models_all_eq_""" + metric + r"""}
\end{table}
"""

    # Save to .txt file
    filename = f"{output_dir}/table_all_models_all_eq_{metric}.txt"
    with open(filename, "w") as f:
        f.write(latex.strip())

    print(f"Saved LaTeX table to {filename}")
