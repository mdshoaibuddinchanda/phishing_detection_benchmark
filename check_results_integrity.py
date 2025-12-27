import pandas as pd
import numpy as np

df = pd.read_csv('results/tables/results_summary.csv')
print('Results DataFrame:')
print(df)
print('\n' + '='*60)
print('Data Quality Checks:')
print('='*60)
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'\nNaN values per column:')
print(df.isnull().sum())
print(f'\nDuplicate rows: {df.duplicated().sum()}')
print(f'\nModels present: {list(df["model"].unique())}')
print(f'\nAll metrics numeric?')
for col in df.columns:
    if col != 'model':
        print(f'  {col}: {pd.api.types.is_numeric_dtype(df[col])}')
