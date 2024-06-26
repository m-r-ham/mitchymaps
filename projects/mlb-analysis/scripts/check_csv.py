import pandas as pd

df = pd.read_csv('/Users/mitchellhamilton/m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/data/mlb_stadiums.csv', sep='\t')
print("Columns in the CSV file:")
print(df.columns)
print("\nFirst few rows:")
print(df.head())
