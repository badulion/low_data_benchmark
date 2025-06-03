import pandas as pd
import os
import natsort
import numpy as np
import matplotlib.pyplot as plt
import itertools

RES = ["low","medium","full","high"]  # Low, Medium, High, Full Resolution
EQ = ["A","B","GD","KS","RD","W"]  # Advection, Burgers, Gas Dynamics, Kuramoto-Sivashinsky, Reaction-Diffusion, Wave

metrics = [f'test_loss-{i}' for i in range(1, 17)]

csv_file = 'results/low_data_benchmark_results.csv'  # Replace with your actual CSV path

output_dir = 'results/plots'  # Directory to save separate tables


equation_map = {
    'advection': 'A',
    'burgers': 'B',
    'gasdynamics': 'GD',
    'kuramotosivashinsky': 'KS',
    'reactiondiffusion': 'RD',
    'wave': 'W',
}

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
    'persistance_cloud': 'persistance'
}

# Load and clean data
df = pd.read_csv(csv_file)
df['equation'] = df['equation'].map(lambda x: equation_map.get(x.lower(), x))
df['model'] = df['model'].map(lambda x: model_map.get(x.lower(), x))
df['domain'] = df['domain']

# Filter required columns
DF_filtered = df[['model', 'domain', 'equation'] + metrics].copy()

for res in RES:
    for eq in EQ:
        df_filtered = DF_filtered

        df_filtered = df_filtered[df['resolution'] == res]
        df_filtered = df_filtered[df['equation'] == eq]

        df_filtered = df_filtered[df['model'] != 'zero_grid']
        df_filtered = df_filtered[df['model'] != 'persistance_grid']

        #print(df_filtered)

        plt.figure(figsize=(10, 5.7))

        # Filter only the step columns (assuming they are named like "step1", "step2", ..., or similar)
        step_columns = natsort.natsorted([col for col in df_filtered.columns if col.startswith('test_loss-')])

        # Extract rows with labels and final step value
        rows_with_labels = []
        for _, row in df_filtered.iterrows():
            # if row['model'] in ['GrINd', 'PTv1', 'PTv3', 'PointNet', 'PointGNN', 'GraphPDE', 'GAT', 'GCN', 'FeaStNet', 'GeoFNO'] and row['domain'] == 'grid':
            #     continue
            if row['model'] in ['zero', 'persistance']:
                label = f"{row['model']} - basline"
            else:
                label = f"{row['model']} - {row['domain']}"
            values = row[step_columns].values.astype(float)  # Ensure numeric
            final_value = values[0]
            rows_with_labels.append((label, values, final_value))

        # Sort by final step value descending
        #rows_with_labels.sort(key=lambda x: x[2], reverse=True)
        rows_with_labels = natsort.natsorted(rows_with_labels, key=lambda x: x[0], reverse=False)

        # Plotting
        min_v = min([min(values) for _, values, _ in rows_with_labels])
        max_v = max([max(values) for _, values, _ in rows_with_labels])

        from collections import defaultdict

        grouped = defaultdict(list)
        for label, values, final_value in rows_with_labels:
            group_key = label[:6]
            grouped[group_key].append((label, values, final_value))

        # Assign a color to each group
        color_cycle = itertools.cycle(plt.cm.tab20.colors)  # or use another colormap if you prefer
        group_colors = {}
        for group_key in grouped:
            group_colors[group_key] = next(color_cycle)

        # Define a list of marker styles
        marker_styles = ['s', '*', 's', '^', 'v', 'D', 'X', 'P', '<', '>', 'H', 'd', 'p', '|'] #, '_']

        for group_key, group_rows in grouped.items():
            color = group_colors[group_key]
            marker_cycle = itertools.cycle(marker_styles)
            for label, values, _ in group_rows:
                marker = next(marker_cycle)
                plt.plot(
                    np.arange(1, len(step_columns) + 1),
                    values,
                    label=label,
                    marker=marker,
                    linewidth=2,
                    color=color
                )

        # # Create an infinite cycle of markers if there are more lines than markers
        # marker_cycle = itertools.cycle(marker_styles)

        # for label, values, _ in rows_with_labels:
        #     marker = next(marker_cycle)
        #     plt.plot(np.arange(1, len(step_columns) + 1), values, label=label, marker=marker, linewidth=2)


        # Axis labels, title, legend
        # ToDo 
        # use gitter to pull lines apart in plot
        # bigger stars
        plt.xlabel("Number of Steps")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.xticks(np.arange(1, len(step_columns) + 1))
        plt.ylim(bottom=min_v - 1e-7, top=2.5)
        plt.xlim(left=0.8, right=len(step_columns)+0.2)
        plt.title(rf"Test Loss vs. Number of Steps for $\bf{{{eq}}}$ Equation", fontsize=14)
        plt.legend(fontsize='small', bbox_to_anchor=(1.02, 1.005), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'test_loss_vs_steps_{eq}_{res}.png'), bbox_inches='tight', dpi=300)
        #plt.show()