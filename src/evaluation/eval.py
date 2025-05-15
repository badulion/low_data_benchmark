import os
import pandas as pd
import numpy as np
import natsort
import matplotlib.pyplot as plt
import re

def process_csv_files(data_directory='data/', plot=False):
    # Dictionary to hold data from each subdirectory
    data_frames = {}

    # Walk through all subdirectories and gather CSV files
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.csv'):
                # Get the subdirectory name
                subdir_name = os.path.basename(os.path.dirname(root))

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

    # Initialize a list to store processed data for the final DataFrame
    processed_data = []

    # Process each DataFrame in the dictionary
    for name, df in data_frames.items():
        # Create a dictionary to hold the original data along with computed mean and std
        summary_dict = {'subdir_name': name}

        # Iterate over columns to calculate statistics for columns with multiple non-NaN values
        for col in df.columns:
            if 'step' in col or 'epoch' in col or 'val_loss' in col or 'train_loss' in col:
                continue
            if 'test_loss' in col and plot==False:
                if 'test_loss-1' == col or 'test_loss-16' == col:
                    pass
                else:
                    continue
            # Filter out NaN values for the column
            valid_values = df[col].dropna()

            # Store original values in the summary dictionary
            # summary_dict[f"{col}_original_values"] = list(valid_values)

            # If there are multiple valid values, compute mean and std; otherwise, set to NaN
            if len(valid_values) > 1:
                summary_dict[f"{col}_mean"] = valid_values.mean()
                summary_dict[f"{col}_std"] = valid_values.std()
            else:
                summary_dict[f"{col}"] = valid_values.iloc[0] if len(valid_values) == 1 else np.nan

        # Append the summary to the processed data list
        processed_data.append(summary_dict)

    # Convert the processed data to a DataFrame
    result_df = pd.DataFrame(processed_data)
    result_df = result_df.reindex(columns=natsort.humansorted(result_df.columns))

    # Reset index for a clean look and save the DataFrame to an Excel file
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'column_name'}, inplace=True)
    result_df.set_index('subdir_name', inplace=True)
    result_df.to_csv("evaluation/aggregated_data.csv", index=True)

    print("Data aggregated and saved to 'aggregated_data.csv'.")

    if plot:
        # plot test loss
        # Filter columns starting with 'test_loss'
        test_loss_columns = [col for col in result_df.columns if col.startswith("test_loss")]
        df_test_loss = result_df[test_loss_columns].T

        # Plot each 'test_loss' column
        plt.figure(figsize=(10, 6))
        # Sort the DataFrame by column names
        df_test_loss = df_test_loss[natsort.natsorted(df_test_loss.columns)]
        for col in df_test_loss.columns:
            plt.plot(np.arange(1,17), df_test_loss[col], label=col, marker='o', linewidth=2)

        # Add labels and title
        #plt.yscale("log")
        plt.xlabel("Number of Steps")
        plt.ylabel("Test Loss Value")
        plt.title("Test Loss with Number of Steps")
        plt.legend()
        plt.grid(True)
        plt.show()

# Run the function
process_csv_files(f'{os.getcwd()}/evaluation/data', plot=True)
