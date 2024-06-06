# Import necessary libraries
import pandas as pd
import numpy as np

# Define the filenames of the CSV files to be processed
filenames = ["Input_Data_saved.csv", 
             "Input_Data_saved1.csv", 
             "Input_Data_saved2.csv"]#, 
            #  "Input_Data.csv"]

# Define the labels to be counted
labs=['Parasitic',
      'Wire_Straight_Defect', 
      'Wire_Straight_Perfect', 
      'Wire_Tilted_Defect', 
      'Wire_Tilted_Perfect', 
      'delete']

# Define a function to count the occurrences of a label in the 'label' column of the csv DataFrame
def counter(lab, count = 0):
    for label_ in csv.label:
        if label_ == lab:
            count += 1
        else:
            continue
    return count

# Process each CSV file
for filename in filenames:
    # Read the CSV file into a DataFrame
    csv = pd.read_csv(filename, dtype="string")
    
    # Drop the rows with all missing values
    csv.dropna(axis=0, how='all', inplace=True)
    
    # Save the DataFrame back to the CSV file
    csv.to_csv(filename, index=False)
    
    # Print the indices of the missing values in the DataFrame
    print(np.where(pd.isnull(csv)))

    # Initialize the total percentage
    perc_tot = 0
    
    # Calculate and print the percentage of each label in the 'label' column
    for lab in labs:
        counting = counter(lab)
        perc = 100 * counter(lab) / len(csv.label)
        perc_tot += perc
        print(f"{lab} percentage: {perc}%")

    # Print the total percentage of the accounted wires
    print(f"Accounted wires: {perc_tot}%")