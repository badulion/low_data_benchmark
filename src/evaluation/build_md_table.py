import pandas as pd
import os
import natsort

RES = "low"

metrics = ['test_loss-1', 'test_loss-16', 'test_time', 'train_step_ms_mean', 'batch_size', 'params']
UNDERLINE = [True,True,True,True,False,False]  # Whether to underline the minimum value in each column
# Load CSV
csv_file = 'results/aggregated_data.csv'  # Replace with your actual CSV path

output_dir = 'results/md_tables'  # Directory to save separate tables



equation_map = {
    'advection': 'A',
    'burgers': 'B',
    'gasdynamics': 'GD',
    'kuramotosivashinsky': 'KS',
    'reactiondiffusion': 'RD',
    'wave': 'W',
}

# Load and clean data
df = pd.read_csv(csv_file)
df['equation'] = df['equation'].map(lambda x: equation_map.get(x.lower(), x))
df['domain'] = df['domain']

# Filter required columns
df_filtered = df[['model', 'domain', 'equation'] + metrics].copy()

df_filtered = df_filtered[df['resolution'] == RES]

# Helper to format values
def format_val(val, metric):
    if pd.isna(val):
        return "-"
    if val > 10000:
        return f"{val}"
    else:
        return f"{val:.2e}"

for metric, ul in zip(metrics, UNDERLINE):
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
            formatted.at[idx, eq] = fval

    # Build LaTeX table
    col_format = "ll" + rf"c@{{\hskip 0.3cm}}" * len(pivot.columns)  # model + domain + equations
    header_cols = ['model', 'domain'] + list(pivot.columns)
    header_line = ' | '.join(header_cols)

    rows = []
    for (model, domain), row in formatted.iterrows():
        row_vals = [model, domain] + [row[eq] for eq in pivot.columns]
        rows.append("|" + ' | '.join(row_vals) + "|")

    latex = header_line + "\n" + "|---"*8 + "|\n" + "\n".join(rows)

    # Save to .txt file
    filename = f"{output_dir}/table_all_models_all_eq_{metric}_{RES}.txt"
    with open(filename, "w") as f:
        f.write(latex.strip())

    print(f"Saved LaTeX table to {filename}")
