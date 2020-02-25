#!/usr/bin/env python
# Python Pandas Cheatsheet
import pandas as pd

# Create a synthetic dataset
df = pd.DataFrame(np.random.rand(5,5))

# Create new empty 2D DataFrame
df = pd.DataFrame()

# Append a row to a data frame
df = df.append([(1, 2, 3)])

# Access row (by label)
df.loc['A']

# Access row (by index)
df.iloc[2]

# Access column
df[2]

# Find row idx of min/max for all columns
df.idxmax()

# Find row idx of min/max in specific column
df['B'].idxmax()

# rename columns
df.rename(columns={'pop':'population',
                          'lifeExp':'life_exp',
                          'gdpPercap':'gdp_per_cap'}, 
                 inplace=True)
df.columns.array

# Check for row equality
row_exists = df2[(df2==row).all(axis=1)].empty
row_idx = df2[(df2==row).all(axis=1)].index[0]

# load from csv file
df = pd.read_csv('evolution_history.csv', delimiter='\t', header=None)

# get statistics for columns in DataFrame
df.describe()
