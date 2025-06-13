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

    # df_filtered = df_filtered[df['resolution'] == res]
    
    # Helper to format values
    def format_val(val, metric) -> str:
        if pd.isna(val):
            return "-"
        if 'loss' in metric and val > 2:
            return r"\text{div}"
        if metric in ['test_time', 'train_step_ms_mean']:
            return rf"\num{{{np.float64(val):.1f}}}"
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

def build_summary_table(csv_file, output_dir, res_list, metric):
    df = pd.read_csv(csv_file)
    equation_map = {
        'advection': 'A',
        'burgers': 'B',
        'gasdynamics': 'GD',
        'kuramotosivashinsky': 'KS',
        'reactiondiffusion': 'RD',
        'wave': 'W',
    }
    df['equation'] = df['equation'].map(lambda x: equation_map.get(x.lower(), x))
    df['domain'] = df['domain']

    summary = {}
    for res in res_list:
        # Filter by resolution if you have a 'resolution' column
        df_res = df[df['resolution'] == res]
        df_res = df_res.copy()  # If you don't have a 'resolution' column, remove this line and filtering

        # Remove unwanted models
        if metric in ["params", "batch_size", "train_step_ms_mean", "test_time"]:
            df_res = df_res[~df_res['model'].isin(['persistance_cloud', 'persistance_grid', 'zero_cloud', 'zero_grid'])]

        # Group by model, average over all equations and domains
        avg = df_res.groupby(['model', 'domain'])[metric].mean()
        summary[res] = avg

    # Build DataFrame: rows=models, columns=resolutions
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    summary_df = summary_df.loc[sorted(summary_df.index)]  # Sort models alphabetically

    # Format values for LaTeX
    def format_val(val):
        if pd.isna(val):
            return "-"
        try:
            if val > 10.0:
                return rf"\num{{{np.float64(val):.1f}}}"
            else:
                return rf"\num{{{np.float64(val):.2e}}}"
        except:
            return str(val)

    formatted_df = summary_df.applymap(format_val)

    # Build LaTeX table
    header_cols = ['model'] + list(formatted_df.columns)
    header_line = ' & '.join(header_cols) + r" \\"
    rows = []
    for (model, domain), row in formatted_df.iterrows():
        row_vals = [model, domain] + [row[res] for res in formatted_df.columns]
        rows.append(' & '.join(row_vals) + r" \\")
    latex = "\\toprule \n" + header_line + "\n\\midrule \n" + "\n".join(rows) + "\n\\bottomrule"

    # Save to .txt file
    filename = f"{output_dir}/summary_table_{metric}.txt"
    with open(filename, "w") as f:
        f.write(latex.strip())
    print(f"Saved summary LaTeX table to {filename}")


if __name__ == "__main__":
    RES = ["low", "medium", "high", "full"]
    metrics = ['test_time']
    UNDERLINE = [True,True,True,True,False,False]
    csv_file = 'results/aggregated_data.csv'
    output_dir = 'results/latex_tables'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build summary tables for each metric
    for metric in metrics:
        build_summary_table(csv_file, output_dir, RES, metric)