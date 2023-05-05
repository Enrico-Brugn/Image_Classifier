import pandas as pd

csv = csv = pd.read_csv("Input_Data.csv", dtype="string")
nulls = 0
for label in csv.label:
    if label == "Null":
        nulls += 1
    else:
        continue



null_perc = 100 * nulls / len(csv.label)

print(f"Null percentage: {null_perc}%")
