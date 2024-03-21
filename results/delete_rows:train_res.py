import csv

def delete_rows_interactive(csv_file):
    """Delete specified rows from a CSV file interactively."""
    with open(csv_file, 'r') as file:
        rows = list(csv.reader(file))

    # Display information about the dataset
    print("Dataset length:", len(rows))
    print("Dataset dimensions:", len(rows), "rows x", len(rows[0]), "columns")
    print("Last 5 rows:")
    for row in rows[-5:]:
        print(row)

    # Prompt user to input indices of rows to delete
    row_indices_str = input("Enter the indices of the rows you want to delete (comma-separated): ")
    row_indices = [int(idx.strip()) for idx in row_indices_str.split(',')]

    # Delete specified rows
    row_indices = sorted(row_indices, reverse=True)  # Sort indices in descending order
    for index in row_indices:
        del rows[index]

    # Write updated data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Example usage:
csv_file = 'results/Training_results.csv'  # Replace with the path to your CSV file
delete_rows_interactive(csv_file)
