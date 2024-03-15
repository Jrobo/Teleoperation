import pandas as pd
from tabulate import tabulate

# Load the saved data from the CSV file
df = pd.read_csv('results/training_results.csv')

# Select only the first two columns
#first_two_columns = df.iloc[:, :]
first_two_columns = df.iloc[:, -1] = df.iloc[:, -1].str[-12:]


# Convert the DataFrame to a tabular format
table = tabulate(df, headers='keys', tablefmt='pretty', numalign='left')


# Display the tabular format
print(table)
