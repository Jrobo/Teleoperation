# Step 1: Import the pandas library
import pandas as pd
from tabulate import tabulate

# Step 2: Read CSV file and specify to read only the 4th column
# Replace 'results/Training_results.csv' with the actual path to your CSV file
df = pd.read_csv('results/Training_results.csv', usecols=[1,2,3,4])

# Step 3: Convert DataFrame to a tabular format
table = tabulate(df, headers='keys', tablefmt='grid')

# Step 4: Print the tabulated DataFrame
print(table)
