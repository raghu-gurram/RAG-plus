# downloads 200 data points of level 5 

from datasets import load_dataset
import pandas as pd

# Load the MATH dataset
print("Loading MATH dataset...")
dataset = load_dataset("EleutherAI/hendrycks_math",'counting_and_probability')

# Combine all splits (train + test)
all_data = []
for split in ['train', 'test']:
    df = pd.DataFrame(dataset[split])
    all_data.append(df)

# Combine into single dataframe
df_combined = pd.concat(all_data, ignore_index=True)

print(f"\nTotal problems in dataset: {len(df_combined)}")
print(f"Column names: {df_combined.columns.tolist()}")

# Filter for Counting & Probability, Level 5
filtered_df = df_combined[
    (df_combined['type'] == 'Counting & Probability') & 
    (df_combined['level'] == 'Level 5')
]

print(f"\nProblems matching 'Counting & Probability' Level 5: {len(filtered_df)}")

# Take first 200
top_200 = filtered_df.head(200).copy()

print(f"Selected: {len(top_200)} problems")

# Display sample
print("\n=== Sample Problem ===")
print(f"Problem: {top_200.iloc[0]['problem'][:200]}...")
print(f"Level: {top_200.iloc[0]['level']}")
print(f"Type: {top_200.iloc[0]['type']}")
print(f"Solution: {top_200.iloc[0]['solution'][:200]}...")

# Save to CSV
output_file = 'math_counting_probability_level5_top200.csv'
top_200.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")

# Also save as JSON for easier processing later
top_200.to_json('math_counting_probability_level5_top200.json', 
                orient='records', indent=2)
print(f"✅ Also saved as JSON")
