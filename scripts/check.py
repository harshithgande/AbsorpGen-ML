import pandas as pd

df = pd.read_csv("data/processed/drug_indications.csv")
print(df.columns.tolist())
