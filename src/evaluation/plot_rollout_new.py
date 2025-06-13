import pandas as pd
import os
import natsort
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.gridspec as gridspec

RES = ["low", "medium", "high", "full"]  # Low, Medium, High, Full Resolution
#EQ = ["A","B","GD","KS","RD","W"]  # Advection, Burgers, Gas Dynamics, Kuramoto-Sivashinsky, Reaction-Diffusion, Wave

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
    'persistance_cloud': 'persistence'
}

# Load and clean data
df = pd.read_csv(csv_file)
df['equation'] = df['equation'].map(lambda x: equation_map.get(x.lower(), x))
df['model'] = df['model'].map(lambda x: model_map.get(x.lower(), x))
df['domain'] = df['domain']

# Filter required columns
DF_filtered = df[['model', 'domain', 'equation', 'resolution'] + metrics].copy()

# Choose 3 equations to plot
EQS_TO_PLOT = [["A", "B", "GD"], ["KS","RD","W"]]  # Change as needed



for res in RES:
    for eq2plot in EQS_TO_PLOT:

        fig, axes = plt.subplots(len(eq2plot), 1, figsize=(10, 16), sharex=True)
        all_handles = []
        all_labels = []

        for idx, eq in enumerate(eq2plot):
            ax = axes[idx]
            df_filtered = DF_filtered
            df_filtered = df_filtered[df_filtered['resolution'] == res]
            df_filtered = df_filtered[df_filtered['equation'] == eq]
            df_filtered = df_filtered[df_filtered['model'] != 'zero_grid']
            df_filtered = df_filtered[df_filtered['model'] != 'persistance_grid']

            step_columns = natsort.natsorted([col for col in df_filtered.columns if col.startswith('test_loss-')])

            rows_with_labels = []
            for _, row in df_filtered.iterrows():
                if row['model'] in ['zero', 'persistence']:
                    label = f"{row['model']} - baseline"
                else:
                    label = f"{row['model']} - {row['domain']}"
                values = row[step_columns].values.astype(float)
                final_value = values[0]
                rows_with_labels.append((label, values, final_value))

            rows_with_labels = natsort.natsorted(rows_with_labels, key=lambda x: x[0], reverse=False)

            min_v = min([min(values) for _, values, _ in rows_with_labels])
            max_v = max([max(values) for _, values, _ in rows_with_labels])

            from collections import defaultdict
            grouped = defaultdict(list)
            for label, values, final_value in rows_with_labels:
                group_key = label[:6]
                grouped[group_key].append((label, values, final_value))

            color_cycle = itertools.cycle(plt.cm.tab20.colors)
            group_colors = {}
            for group_key in grouped:
                if group_key[:4] in ["zero", "pers"]:
                    group_colors[group_key] = 'black'
                else:
                    group_colors[group_key] = next(color_cycle)

            marker_styles = ['s', '*', 'p', 'D', 'H', '^', 'v', 'X', 'P', '<', '>', 'd', '|']
            for group_key, group_rows in grouped.items():
                color = group_colors[group_key]
                marker_cycle = itertools.cycle(marker_styles)
                for label, values, _ in group_rows:
                    if group_key[:4] == "zero":
                        marker = '|'
                    elif group_key[:4] == "pers":
                        marker = 'o'
                    else:
                        marker = next(marker_cycle)
                    if marker == '*':
                        handle, = ax.plot(
                            np.arange(1, len(step_columns) + 1),
                            values,
                            label=label,
                            marker=marker,
                            markersize=8,
                            linewidth=2,
                            color=color,
                            alpha=1.0
                        )
                    else:
                        handle, = ax.plot(
                            np.arange(1, len(step_columns) + 1),
                            values,
                            label=label,
                            marker=marker,
                            markersize=5,
                            linewidth=2,
                            color=color,
                            alpha=1.0
                        )
                    if label not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(label)

                ax.set_yscale("log")
                ax.set_ylabel("MSE", fontsize=13)
                ax.set_ylim(bottom=min_v - 1e-7, top=20)
                ax.set_xlim(left=0.8, right=len(step_columns)+0.2)
                ax.set_title(rf"Test Loss vs. Number of Steps for $\bf{{{eq}}}$ Equation", fontsize=14)
                ax.grid(True, which='major', axis='y')

        axes[-1].set_xlabel("Number of Steps", fontsize=14)
        plt.xticks(np.arange(1, len(step_columns) + 1))

        # Place the legend to the right of the plots
        fig.legend(all_handles, all_labels, ncol=1, fontsize=12, bbox_to_anchor=(1.02, 0.5), loc='center left')
        plt.tight_layout(pad=3)  # Leave space for legend
        print(f"Saving {f'test_loss_vs_steps_{res}_{'_'.join(eq2plot)}.png'} ...")
        plt.savefig(os.path.join(output_dir, f'test_loss_vs_steps_{res}_{'_'.join(eq2plot)}.png'), bbox_inches='tight', dpi=300)