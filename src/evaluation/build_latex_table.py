import pandas as pd
import os
import natsort
import numpy as np

def build_latex_table(csv_file, output_dir, res, metrics, underline):
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

    df_filtered = df_filtered[df['resolution'] == res]
    
    # Helper to format values
    def format_val(val, metric) -> str:
        if pd.isna(val):
            return "-"
        if 'loss' in metric and val > 2:
            return r"\text{div}"
        try:
            if val > 10.0:
                return rf"\num{{{np.float64(val)}}}"
            else:
                return rf"\num{{{np.float64(val):.2e}}}"
        except:
            return str(val)

    for metric, ul in zip(metrics, underline):

        if metric in ["params", "batch_size", "train_step_ms_mean", "test_time"]:
            df_filtered = df_filtered[df['model'] != 'persistance_cloud']
            df_filtered = df_filtered[df['model'] != 'persistance_grid']
            df_filtered = df_filtered[df['model'] != 'zero_cloud']
            df_filtered = df_filtered[df['model'] != 'zero_grid']

        # Pivot table
        pivot = df_filtered.pivot(index=['model', 'domain'], columns='equation', values=metric)
        pivot = pivot.sort_values(by='domain', ascending=False)

        # Create masks for underlining and underwaving
        underline_mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
        underwave_mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
        for eq in pivot.columns:
            for domain in ['grid', 'cloud']:
                col = pivot.xs(domain, level='domain')[eq] if (domain in pivot.index.get_level_values('domain')) else pd.Series(dtype=float)
                if 'loss' in metric:
                    valid = col[col <= 2].dropna()
                else:
                    valid = col.dropna()
                if not valid.empty:
                    min_val = valid.min()
                    mask = col == min_val
                    for model in col[mask].index:
                        if domain == 'grid':
                            underline_mask.loc[(model, domain), eq] = True
                        elif domain == 'cloud':
                            underwave_mask.loc[(model, domain), eq] = True

        # Format all entries
        formatted = pivot.copy()
        for eq in pivot.columns:
            for idx in pivot.index:
                val = pivot.at[idx, eq]
                fval = format_val(val, metric)
                if ul:
                    if underline_mask.at[idx, eq] and fval not in [r"\text{div}", "-"]:
                        fval = rf"\underline{{{fval}}}"
                    elif underwave_mask.at[idx, eq] and fval not in [r"\text{div}", "-"]:
                        fval = rf"\uwave{{{fval}}}"
                formatted.at[idx, eq] = fval


        # Build LaTeX table
        col_format = "ll" + rf"c@{{\hskip 0.3cm}}" * len(pivot.columns)  # model + domain + equations
        header_cols = ['model', 'domain'] + list(pivot.columns)
        header_line = ' & '.join(header_cols) + r" \\"

        rows = []
        for (model, domain), row in formatted.iterrows():
            row_vals = [model, domain] + [row[eq] for eq in pivot.columns]
            rows.append(' & '.join(row_vals) + r" \\")

        latex = "\\toprule \n" + header_line + "\n\\midrule \n" + "\n".join(rows) + "\n\\bottomrule"

        # Save to .txt file
        filename = f"{output_dir}/table_all_models_all_eq_{metric}_{res}.txt"
        with open(filename, "w") as f:
            f.write(latex.strip())

        print(f"Saved LaTeX table to {filename}")


if __name__ == "__main__":

    RES = ["low", "medium", "high", "full"]  # Resolutions to process

    metrics = ['test_loss-1', 'test_loss-16', 'test_time', 'train_step_ms_mean', 'batch_size', 'params']
    UNDERLINE = [True,True,True,True,False,False]  # Whether to underline the minimum value in each column
    
    # Load CSV
    csv_file = 'results/aggregated_data.csv'  # Replace with your actual CSV path
    output_dir = 'results/latex_tables'  # Directory to save separate tables

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for res in RES:
        build_latex_table(csv_file, output_dir, res=res, metrics=metrics, underline=UNDERLINE)
        print(f"Latex tables for resolution '{RES}' have been built and saved in '{output_dir}'.")