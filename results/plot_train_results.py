import pandas as pd
from tabulate import tabulate

# Load the CSV file into a DataFrame
df = pd.read_csv('results/Training_results.csv')

# # Extract the last n characters from the 'Image Path' column
# df['Image Path'] = df['Image Path'].str[-2:]

# # Extract the last n characters from the 'Numpy Path' column if it exists
# if 'Numpy Path' in df.columns:
#     df['Numpy Path'] = df['Numpy Path'].str[-2:]

# Select only the first 5 columns
df_first_5_columns = df.iloc[:, :10]

# Convert the DataFrame to a tabular format
table = tabulate(df_first_5_columns, headers='keys', tablefmt='pretty', showindex=False)


# Convert the DataFrame to a tabular format
#table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False)

# Display the tabular format
print(table)
