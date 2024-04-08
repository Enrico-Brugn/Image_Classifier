import pandas as pd
import numpy as np
filenames = ["Input_Data_saved.csv", 
             "Input_Data_saved1.csv", 
             "Input_Data_saved2.csv"]#, 
            #  "Input_Data.csv"]

labs=['Parasitic',
      'Wire_Straight_Defect', 
      'Wire_Straight_Perfect', 
      'Wire_Tilted_Defect', 
      'Wire_Tilted_Perfect', 
      'delete']

def counter(lab, count = 0):
    for label_ in csv.label:
        if label_ == lab:
            count += 1
        else:
            continue
    return count

for filename in filenames:
    csv = csv = pd.read_csv(filename, dtype="string")
    csv.dropna(axis=0, how='all', inplace=True)
    csv.to_csv(filename, index=False)
    print(np.where(pd.isnull(csv)))

    perc_tot = 0
    for lab in labs:
        counting = counter(lab)
        perc = 100 * counter(lab) / len(csv.label)
        perc_tot += perc
        print(f"{lab} percentage: {perc}%")

    print(f"Accounted wires: {perc_tot}%")