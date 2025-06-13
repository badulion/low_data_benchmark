from dynabench.utils.colors import NEURALPDE_COLORMAP
import numpy as np



import pandas as pd
import os
import natsort
import numpy as np


model_map = {
    'fno': 'FNO',
    'geo_fno': 'GeoFNO',
    'cnn': 'CNN',
    'resnet': 'ResNet',
    'neuralpde': 'NeuralPDE',
    'grind': 'GrINd',
    'ptv1': 'PTv1',
    'ptv3': 'PTv3',
    'pointnet': 'PointNet',
    'pointgnn': 'PointGNN',
    'graphpde': 'GraphPDE',
    'gat': 'GAT',
    'gcn': 'GCN',
    'feast': 'FeaStNet',
    'zero_cloud': 'zero',
    'persistance_cloud': 'persistence'
}

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
    df['model'] = df['model'].map(lambda x: model_map.get(x.lower(), x))
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
        
    def rgb2hex(r,g,b):
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        return "{:02x}{:02x}{:02x}".format(r,g,b)
    
    ranks = []

    for metric, ul in zip(metrics, underline):

        df_filtered = df_filtered[df_filtered['model'] != 'zero_grid']
        df_filtered = df_filtered[df_filtered['model'] != 'persistance_grid']

        # replace 'persistance' with 'persistence'
        # df_filtered['model'] = df_filtered['model'].str.replace('persistance_cloud', 'persistence', regex=False)
        # df_filtered['model'] = df_filtered['model'].str.replace('zero_cloud', 'zero', regex=False)
        # df_filtered['model'] = df_filtered['model'].str.replace('geo_fno', 'GeoFNO', regex=False)

        # change domain of persistence and zero models to 'baseline'
        df_filtered.loc[df_filtered['model'] == 'persistence', 'domain'] = 'baseline'
        df_filtered.loc[df_filtered['model'] == 'zero', 'domain'] = 'baseline'

        if metric in ["params", "batch_size", "train_step_ms_mean", "test_time"]:
            df_filtered = df_filtered[df['model'] != 'persistence']
            df_filtered = df_filtered[df['model'] != 'zero']

        # Pivot table
        pivot = df_filtered.pivot(index=['model', 'domain'], columns='equation', values=metric)
        pivot = pivot.sort_values(by='domain', ascending=False)

        # define colored
        colours = NEURALPDE_COLORMAP(np.linspace(0, 1, len(pivot)))
        latex_colour_definitions = ["\definecolor{rank%d}{rgb}{%f,%f,%f}" % (i+1, *col[:3]) for i, col in enumerate(colours)]
        latex_colour_definitions = "\n".join(latex_colour_definitions)
        # Save to .txt file
        filename = f"{output_dir}/color_definitions_{metric}_{res}.txt"
        with open(filename, "w") as f:
            f.write(latex_colour_definitions.strip())


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
        
        rank_table = df_filtered.pivot(index=['model', 'domain'], columns='equation', values=metric)

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
                if True:
                    try:
                        rank = int(pivot[eq].rank(ascending=True, method='min').at[idx])-1
                        color = colours[rank][:-1]  # Exclude alpha channel
                        hex_color = rgb2hex(*color)
                        # fval = rf"\textcolor[HTML]{{{hex_color}}}{{{fval}}}"
                        fval = "\cellcolor{rank%d!50}%s" % (rank + 1, fval)
                        rank_table.at[idx, eq] = rank
                    except ValueError:
                        # Handle case where rank cannot be determined
                        pass
                        rank_table.at[idx, eq] = 1000
                  # Store formatted value in rank table
                formatted.at[idx, eq] = fval

        rank_table[f'mean_rank_{res}_{metric}'] = rank_table.iloc[:].mean(axis=1)
        if metric in ['test_loss-1', 'test_loss-16']:
            ranks.append(rank_table)

        # # Build LaTeX table
        # col_format = "ll" + rf"c@{{\hskip 0.3cm}}" * len(pivot.columns)  # model + domain + equations
        # header_cols = ['model', 'domain'] + list(pivot.columns)
        # header_line = ' & '.join(header_cols) + r" \\"

        # rows = []
        # for (model, domain), row in formatted.iterrows():
        #     row_vals = [model, domain] + [row[eq] for eq in pivot.columns]
        #     rows.append(' & '.join(row_vals) + r" \\")

        # Define desired model and domain order
        model_order = [
            'CNN', 'ResNet', 'FNO', 'NeuralPDE', 'GrINd', 'PTv1', 'PTv3', 'GeoFNO',
            'PointNet', 'PointGNN', 'GraphPDE', 'GAT', 'GCN', 'FeaStNet', 'persistence', 'zero'
        ]
        domain_order = ['grid', 'cloud', 'baseline']

        # Convert index to DataFrame for sorting
        formatted_reset = formatted.reset_index()
        formatted_reset['domain'] = pd.Categorical(formatted_reset['domain'], categories=domain_order, ordered=True)
        formatted_reset['model'] = pd.Categorical(formatted_reset['model'], categories=model_order, ordered=True)
        formatted_sorted = formatted_reset.sort_values(['domain', 'model'])

        # Build LaTeX table
        col_format = rf"\begin{{tabular}}{{" + rf"l@{{\hskip 0.25cm}}" * (len(pivot.columns) + 1) + "l}" # model + domain + equations
        header_cols = ['model', 'domain'] + list(pivot.columns)
        header_line = ' & '.join(header_cols) + r" \\"

        rows = []
        past_dom = 'grid'
        for _, row in formatted_sorted.iterrows():
            if row['domain'] != past_dom:
                rows.append(r"\midrule")
            past_dom = row['domain']
            row_vals = [row['model'], row['domain']] + [str(row[eq]) for eq in pivot.columns]
            rows.append(' & '.join(row_vals) + r" \\")



        latex = col_format + "\n" + "\\toprule \n" + header_line + "\n\\midrule \n" + "\n".join(rows) + "\n\\bottomrule" + "\n\\end{tabular}"

        # Save to .txt file
        filename = f"{output_dir}/table_all_models_all_eq_{metric}_{res}.txt"
        with open(filename, "w") as f:
            f.write(latex.strip())

        print(f"Saved LaTeX table to {filename}")

    return ranks


if __name__ == "__main__":

    RES = ["low", "medium", "high", "full"]  # Resolutions to process

    metrics = ['test_loss-1', 'test_loss-16', 'test_time', 'train_step_ms_mean', 'batch_size', 'params']
    UNDERLINE = [True,True,True,True,False,False]  # Whether to underline the minimum value in each column
    
    # Load CSV
    csv_file = 'results/aggregated_data.csv'
    #csv_file = 'results/low_data_benchmark_results.csv'  # Replace with your actual CSV path
    output_dir = 'results/latex_tables'  # Directory to save separate tables

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ranks_1 = []
    ranks_16 = []
    
    for res in RES:
        rank_table = build_latex_table(csv_file, output_dir, res=res, metrics=metrics, underline=UNDERLINE)
        ranks_1 = ranks_1 + [rank_table[0]]
        ranks_16 = ranks_16 + [rank_table[1]]
        print(f"Latex tables for resolution '{RES}' have been built and saved in '{output_dir}'.")

    # Combine all rank tables into a single DataFrame for each metric
    def combine_rank_tables(rank_tables, metric_name):
        # Each rank_table is a DataFrame with MultiIndex (model, domain) and columns for equations + mean_rank
        combined = pd.concat(
            [df[f'mean_rank_{res}_{metric_name}'] for df, res in zip(rank_tables, RES)],
            axis=1,
            keys=RES
        )
        combined.columns = [f"{res}" for res in RES]
        combined = combined.reset_index()  # model, domain as columns
        return combined
    
    def save_combined_rank_table(df, output_dir, file_name):
        model_order = [
            'CNN', 'ResNet', 'FNO', 'NeuralPDE', 'GrINd', 'PTv1', 'PTv3', 'GeoFNO',
            'PointNet', 'PointGNN', 'GraphPDE', 'GAT', 'GCN', 'FeaStNet', 'persistence', 'zero'
        ]
        domain_order = ['grid', 'cloud', 'baseline']

        # Convert index to DataFrame for sorting
        formatted_reset = df #.reset_index()
        formatted_reset['domain'] = pd.Categorical(formatted_reset['domain'], categories=domain_order, ordered=True)
        formatted_reset['model'] = pd.Categorical(formatted_reset['model'], categories=model_order, ordered=True)
        formatted_sorted = formatted_reset.sort_values(['domain', 'model'])
        # Reorder columns as: low, medium, mean, full (keeping model and domain first)
        cols = ['model', 'domain', 'low', 'medium', 'high', 'mean', 'full']
        formatted_sorted = formatted_sorted[cols]

        # Build LaTeX table
        col_format = rf"\begin{{tabular}}{{" + "l@{\hskip 0.25cm}l@{\hskip 0.25cm}" + rf"r@{{\hskip 0.25cm}}" * (len(formatted_sorted.columns) -2) + "r}" # model + domain + equations
        header_cols = list(formatted_sorted.columns)
        header_line = ' & '.join(header_cols) + r" \\"

##
        formatted_sorted = formatted_sorted.sort_values('mean', ascending=True)
        ranked = formatted_sorted.copy()
        for c in formatted_sorted.columns:
            for idx in formatted_sorted.index:
                val = formatted_sorted.at[idx, c]
                if pd.isna(val) or (isinstance(val, np.float64) and np.isnan(val)):
                    fval = "-"
                elif type(val) == np.float64:
                    fval = rf"\num{{{np.float64(val):.1f}}}"
                else:
                    fval = str(val)

                try:
                    rank = int(formatted_sorted[c].rank(ascending=True, method='min').at[idx])-1
                    #color = colours[rank][:-1]  # Exclude alpha channel
                    #hex_color = rgb2hex(*color)
                    # fval = rf"\textcolor[HTML]{{{hex_color}}}{{{fval}}}"
                    fval = "\cellcolor{rank%d!50}%s" % (rank + 1, fval)
                except ValueError:
                    # Handle case where rank cannot be determined
                    pass
                
                if type(formatted_sorted[c].at[idx]) != str:
                    ranked.at[idx, c] = fval
##
        formatted_sorted = ranked
        print(formatted_sorted)

        rows = []
        for _, row in formatted_sorted.iterrows():
            row_vals = [str(row[eq]) for eq in formatted_sorted.columns]
            rows.append(' & '.join(row_vals) + r" \\")

        latex = col_format + "\n" + "\\toprule \n" + header_line + "\n\\midrule \n" + "\n".join(rows) + "\n\\bottomrule" + "\n\\end{tabular}"

        # Save to .txt file
        filename = f"{output_dir}/{file_name}.txt"
        with open(filename, "w") as f:
            f.write(latex.strip())


    # Example: combine for test_loss-1 and test_loss-16
    if ranks_1:
        combined_1 = combine_rank_tables(ranks_1, 'test_loss-1')
        #combined_1.fillna(100, inplace=True)
        numeric_cols = [col for col in combined_1.columns if col in ['low', 'medium', 'high']]
        combined_1['mean'] = combined_1[numeric_cols].mean(axis=1)
        print("Combined mean ranks for test_loss-1:")
        #print(combined_1)
        save_combined_rank_table(combined_1, output_dir, 'combined_mean_ranks_test_loss-1')

    if ranks_16:
        combined_16 = combine_rank_tables(ranks_16, 'test_loss-16')
        #combined_16.fillna(100, inplace=True)
        numeric_cols = [col for col in combined_16.columns if col in ['low', 'medium', 'high']]
        combined_16['mean'] = combined_16[numeric_cols].mean(axis=1)
        print("Combined mean ranks for test_loss-16:")
        #print(combined_16)
        save_combined_rank_table(combined_16, output_dir, 'combined_mean_ranks_test_loss-16')
        