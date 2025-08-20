import pandas as pd
import glob
import os

# Folder containing all the training files (.tsv).
training_data_folder = '/content/project/features_ring'

# Name of the final file that will contain the merged dataset.
output_filename = 'complete_training_dataset.tsv'

# Find all files in the training folder.
search_path = os.path.join(training_data_folder, "*.tsv") # Crea il percorso di ricerca corretto
all_files = glob.glob(search_path)

if not all_files:
    print(f"ERROR: No .tsv files were found in '{training_data_folder}'.")
    print("Check that the path is correct and the files are present.")
else:
    print(f"Found {len(all_files)} files. Starting merge...")

    # Create a list to collect each file after reading it.
    dataframe_list = []
    
    for file_path in all_files:
        
        df = pd.read_csv(file_path, sep='\t')
        # Add the DataFrame to the list.
        dataframe_list.append(df)

    # Concatenate (merge) all DataFrames into a single one.
    print("Merging files...")
    combined_df = pd.concat(dataframe_list, ignore_index=True)

    # Save the final DataFrame to a new .tsv file.
    print(f"Saving merged dataset to '{output_filename}'...")
    combined_df.to_csv(output_filename, sep='\t', index=False)

    print(f"File '{output_filename}' created successfully.")
    print(f"Combined dataset size: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns.")