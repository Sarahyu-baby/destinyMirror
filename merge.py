import pandas as pd

# Load the datasets
df_features = pd.read_csv('celebrity_face_features.csv')
df_labels = pd.read_csv('celebrity_labels.csv')

# Function to clean and standardize celebrity names
# This handles differences like "Elon Musk" (space) vs "Elon_Musk" (underscore)
def clean_name(name):
    if isinstance(name, str):
        # Strip whitespace and replace internal spaces/underscores with a single underscore
        # This ensures 'Elon Musk' and 'Elon_Musk' become identical ('Elon_Musk')
        return '_'.join(name.strip().replace('_', ' ').split())
    return name

# Apply the cleaning function to create a common key for merging
df_features['Celebrity_Key'] = df_features['Celebrity'].apply(clean_name)
df_labels['Celebrity_Key'] = df_labels['Celebrity'].apply(clean_name)

# Merge the DataFrames
# perform a Left Join to keep all rows from the benchmark (features) file
merged_df = pd.merge(df_features, df_labels, on='Celebrity_Key', how='left', suffixes=('', '_label'))

# Calculate match statistics
# We use the 'Career' column to check if a label was found (since it comes from the labels file)
num_matches = merged_df['Career'].notna().sum()
total_benchmark = len(df_features)
unmatched_count = total_benchmark - num_matches

# Print statistics to the console
print(f"Total celebrities in benchmark file: {total_benchmark}")
print(f"Successfully matched with labels: {num_matches}")
print(f"Unmatched celebrities: {unmatched_count}")

# Identify and print unmatched celebrities for verification
unmatched_list = merged_df[merged_df['Career'].isna()]['Celebrity'].tolist()
print(f"List of unmatched celebrities: {unmatched_list}")

# Cleanup: Drop temporary keys and redundant columns
# We remove the helper column 'Celebrity_Key' and any duplicate 'Celebrity' column from the labels file
cols_to_drop = ['Celebrity_Key', 'Celebrity_label']
final_df = merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns])

# Save the merged data to a CSV file
output_filename = 'merged_celebrity_data.csv'
final_df.to_csv(output_filename, index=False)
print(f"Merged data saved to {output_filename}")