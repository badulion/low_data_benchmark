import os
import pandas as pd
import numpy as np
import natsort
import matplotlib.pyplot as plt
import re
import yaml

def test_all(df):
    import itertools

    # Your expected values
    domains = ["cloud", "grid"]
    resolutions = ["low", "medium", "high", "full"]
    models = ["cnn", "fno", "gat", "gcn", "geo_fno", "graphpde", "neuralpde", "pointgnn", "pointnet", "ptv1", "ptv3", "resnet", "grind", "feast"]
    equations = ["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]

    # All possible combinations
    all_combinations = set(itertools.product(domains, resolutions, models, equations))

    # impossible combinations
    impossible_combinations_1 = set(itertools.product(["cloud"], resolutions, ["cnn", "fno", "resnet", "neuralpde"], equations))
    impossible_combinations_2 = set(itertools.product(["cloud"], ["full"], models, equations))

    # Existing combinations in the DataFrame (assuming df is your DataFrame)
    existing_combinations = set(df[['domain', 'resolution', 'model', 'equation']].itertuples(index=False, name=None))

    # Find missing combinations
    missing_combinations = all_combinations - existing_combinations - impossible_combinations_1 - impossible_combinations_2

    # Print missing combinations
    print(f"Missing {len(missing_combinations)} combinations:")
    for combo in sorted(missing_combinations):
        print(combo)

def process_csv_files(data_directory='../../results/low', safe_dir='.'):
    # Dictionary to hold data from each subdirectory
    data_frames = {}
    hparams = {}

    # Walk through all subdirectories and gather CSV files
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.csv'):
                # Get the subdirectory name
                subdir_name = os.path.basename(os.path.dirname(root))

                ks = [df[:-4] for df in data_frames.keys()]

                if subdir_name in ks and subdir_name not in data_frames.keys():
                    print(subdir_name)

                # Load the CSV file into a DataFrame
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Store the DataFrame in the dictionary
                if subdir_name in data_frames:
                    # Append data if the subdirectory key already exists
                    data_frames[subdir_name] = pd.concat([data_frames[subdir_name], df])
                else:
                    # Create a new entry if it doesn't exist
                    data_frames[subdir_name] = df
            if file.endswith('.yaml'):
                subdir_name = os.path.basename(os.path.dirname(root))
                file_path = os.path.join(root, file)
                with open(f'{file_path}', 'r') as file:
                    data = yaml.safe_load(file)
                hparams[subdir_name] = data


    print("READ ALL FILES... ")

    # Initialize a list to store processed data for the final DataFrame
    processed_data = []

    # Process each DataFrame in the dictionary
    for df, hparam in zip(data_frames.items(), hparams.values()):
        name, df = df
        # Create a dictionary to hold the original data along with computed mean and std
        s = name.split(':')
        if len(s[0].split('_')) == 3:
            temp = s[0].split('_')
            dataset, model = temp[0], f'{temp[1]}_{temp[2]}'
        else:
            dataset, model = s[0].split('_')
        equation = s[1]
        resolution = s[2]
        version = f'{s[3]}:{s[4]}'
        summary_dict = {}

        #summary_dict['dataset'] = dataset
        summary_dict['model'] = model.lower() #.replace('_', ' ')
        summary_dict['version'] = version
        summary_dict['equation'] = equation
        summary_dict['resolution'] = resolution
        summary_dict['domain'] = hparam['hparams']['Structure']
        summary_dict['params'] = hparam['trainable_parameters']
        if hparam['hparams']['Epochs'] >= 10:
            summary_dict['epochs'] = 10
        elif hparam['hparams']['Epochs'] >= 3:
            summary_dict['epochs'] = 3
        summary_dict['batch_size'] = hparam['hparams']['Batch_size']
        summary_dict['lookback'] = hparam['hparams']['lookback']
        summary_dict['TrainRollout'] = hparam['hparams']['TrainRollout']
        summary_dict['TestRollout'] = hparam['hparams']['TestRollout']
        summary_dict['lr'] = hparam['hparams']['LearningRate']
        summary_dict['wd'] = hparam['hparams']['WeightDecay']

        # Iterate over columns to calculate statistics for columns with multiple non-NaN values
        for col in df.columns:
            if 'step' in col or 'epoch' in col or 'val_loss' in col or 'train_loss' in col:
                if 'train_step' in col:
                    pass
                else:
                    continue
            if col == 'test_loss':
                continue
            if 'test' in col or 'train':
                pass
            else:
                continue
            # Filter out NaN values for the column
            valid_values = df[col].dropna()

            # If there are multiple valid values, compute mean and std; otherwise, set to NaN
            if len(valid_values) > 1 and 'test' not in col:
                summary_dict[f"{col}_mean"] = valid_values.mean()
                summary_dict[f"{col}_std"] = valid_values.std()
            else:
                summary_dict[f"{col}"] = valid_values.iloc[0]

        # Append the summary to the processed data list
        processed_data.append(summary_dict)

    # Convert the processed data to a DataFrame
    result_df = pd.DataFrame(processed_data)
    result_df = result_df.reindex(columns=natsort.humansorted(result_df.columns))

    c_low = 0
    c_medium = 0
    c_high = 0
    c_full = 0

    test_all(result_df)

    for value in result_df['resolution']:
        if value == 'low':
            c_low += 1
        elif value == 'medium':
            c_medium += 1
        elif value == 'high':   
            c_high += 1
        elif value == 'full':
            c_full += 1
    print(f"Number of low resolution models: {c_low}")
    print(f"Number of medium resolution models: {c_medium}")
    print(f"Number of high resolution models: {c_high}")
    print(f"Number of full resolution models: {c_full}")

    # Reset index for a clean look and save the DataFrame to an Excel file
    result_df.reset_index(inplace=True)
    result_df.set_index('model', inplace=True)
    result_df.sort_index(inplace=True)
    result_df.drop(columns=['index'], inplace=True)

    result_df.to_csv(f"{safe_dir}/aggregated_data.csv", index=True)

    print("Data aggregated and saved to 'aggregated_data.csv'.")

# Run the function
process_csv_files(f'{os.getcwd()}/results/csv', safe_dir=f'{os.getcwd()}/results')
