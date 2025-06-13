import pandas as pd
import os
import natsort
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

RES = ["low","medium","high","full"]  # Low, Medium, High, Full Resolution
EQ = ["A","B","GD","KS","RD","W"]  # Advection, Burgers, Gas Dynamics, Kuramoto-Sivashinsky, Reaction-Diffusion, Wave
SINGLE_EQ = False
n = 1
DIFF_DOMAIN = SINGLE_EQ

csv_file = 'results/aggregated_data.csv'  # Replace with your actual CSV path
output_dir = 'results/plots'  # Directory to save separate tables

plt.figure(figsize=(10, 5.7))

def gen_plot_list(n):

    metric = f'test_loss-{n}'

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
        'zero_grid': 'zero',
        'persistance_grid': 'persistence'
    }

    # Load and clean data
    df = pd.read_csv(csv_file)
    df = df[df['model'] != 'zero_cloud']
    df = df[df['model'] != 'persistance_cloud']
    df = df[df['model'] != 'zero_grid']
    df = df[df['model'] != 'persistance_grid']
    df['equation'] = df['equation'].map(lambda x: equation_map.get(x.lower(), x))
    df['model'] = df['model'].map(lambda x: model_map.get(x.lower(), x))
    df['domain'] = df['domain']

    # Filter required columns
    DF_filtered = df[['model', 'domain', 'equation', 'resolution'] + [metric]].copy()

    plot_list = []

    #for eq in EQ:

    results = []

    for res in RES:

        df_filtered = DF_filtered

        df_filtered = df_filtered[df_filtered['resolution'] == res]

        df_filtered = df_filtered.pivot(index=['model', 'domain'], columns='equation', values=metric)

        if SINGLE_EQ:
            avg_df = df_filtered[eq]
        else:
            # Group by model and domain, then average test_loss over all equations
            if DIFF_DOMAIN:
                avg_df = df_filtered.median(axis=1)
            else:
                avg_df = df_filtered.groupby('model').mean(numeric_only=True).median(axis=1)

        avg_df.name = res  # Name the column by resolution
        results.append(avg_df)

    summary = pd.concat(results, axis=1)
    if DIFF_DOMAIN:
        summary.index.names = ['model', 'domain']
    else:
        summary.index.names = ['model']
    #print(summary)

    # Extract rows with labels and final step value
    rows_with_labels = []
    for idx, row in summary.iterrows():
        if len(idx) == 2:
            if idx[0] in ['zero', 'persistence']:
                label = f"{idx[0]} - basline"
            else:
                label = f"{idx[0]} - {idx[1]}"
        else:
            label = idx
        values = row[RES].values.astype(float)
        final_value = values[0]
        rows_with_labels.append((label, values, final_value))

    rows_with_labels = natsort.natsorted(rows_with_labels, key=lambda x: x[0], reverse=False)

    # Plotting
    min_v = min([min(values) for _, values, _ in rows_with_labels])
    max_v = max([max(values) for _, values, _ in rows_with_labels])

    grouped = defaultdict(list)
    for label, values, final_value in rows_with_labels:
        group_key = label[:6]
        grouped[group_key].append((label, values, final_value))

    # Assign a color to each group
    color_cycle = itertools.cycle(plt.cm.tab20.colors)  # or use another colormap if you prefer
    group_colors = {}
    for group_key in grouped:
        if group_key[:4] in ["zero", "pers"]:
            group_colors[group_key] = 'black'
        else:
            group_colors[group_key] = next(color_cycle)

    # Define a list of marker styles
    
    marker_styles = ['s', '*', 'p', 'D', 'H', '^', 'v', 'X', 'P', '<', '>', 'd', '|'] #, '_']
    
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
            if 'grid' in label.lower():
                alpha = 1.0
            else:
                alpha = 1.0

            if marker == '*':
                plot_list.append((np.arange(1, 5), values, label, marker, 8, 2, color, alpha))
                # plt.plot(
                #     np.arange(1, 5),
                #     values,
                #     label=label,
                #     marker=marker,
                #     markersize=8,
                #     linewidth=2,
                #     color=color,
                #     alpha=alpha
                # )
            else:
                plot_list.append((np.arange(1, 5), values, label, marker, 5, 2, color, alpha))
                # plt.plot(
                #     np.arange(1, 5), # + np.random.uniform(-jitter_strength, jitter_strength, size=len(step_columns)),
                #     values,
                #     label=label,
                #     marker=marker,
                #     markersize=5,
                #     linewidth=2,
                #     color=color,
                #     alpha=alpha
                # )
    return plot_list, min_v, max_v

plots = []

min_v = float('inf')

for n in [1,16]:
    plot_list, mins, maxs = gen_plot_list(n)

    plots.append(plot_list+ [mins,maxs])

    # if mins < min_v:
    #     min_v = mins

# # Axis labels, title, legend
# plt.xlabel("Resolution")
# plt.yscale("log")
# plt.xticks(np.arange(1, 5), RES)
# plt.ylim(bottom=min_v * 0.5, top=10)
# plt.xlim(left=0.5, right=4.5)
# if SINGLE_EQ:
#     plt.title(rf"16 Step MSE per Resolution for {eq} Equation", fontsize=14)
#     plt.ylabel("MSE")
# else:
#     plt.title(rf"{n} Step Median MSE per Resolution over all Equations", fontsize=14)
#     plt.ylabel("Median MSE across all Equations")
# plt.legend(fontsize='small', bbox_to_anchor=(1.02, 1.005), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# if SINGLE_EQ:
#     plt.savefig(os.path.join(output_dir, f'median_mse_{eq}_{n}.png'), bbox_inches='tight', dpi=300)
# else:
#     plt.savefig(os.path.join(output_dir, f'median_mse_{n}.png'), bbox_inches='tight', dpi=300)
# #plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 6)) #, sharey=True)  # 1 row, 2 columns

for ax, (plot_list, n) in zip(axes, zip(plots, [1, 16])):
    min_v, max_v = plot_list.pop(-2), plot_list.pop(-1)
    for x, y, label, marker, ms, lw, color, alpha in plot_list:
        ax.plot(x, y, label=label, marker=marker, markersize=ms, linewidth=lw, color=color, alpha=alpha)
    ax.set_xlabel("resolution", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(RES, fontsize=14)
    ax.set_ylim(bottom=min_v * 0.5, top=max_v + max_v * 0.5)
    ax.set_xlim(left=0.5, right=4.5)
    if SINGLE_EQ:
        ax.set_title(rf"16 Step MSE per Resolution for {eq} Equation", fontsize=14)
        ax.set_ylabel("MSE", fontsize=14)
    else:
        ax.set_title(rf"{n} step median MSE", fontsize=14)
        ax.set_ylabel("median MSE", fontsize=14)
    #ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.legend(fontsize='small', bbox_to_anchor=(1.02, 1.005), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'median_mse_subplots.png'), bbox_inches='tight', dpi=300)
# plt.show()