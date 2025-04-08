import sqlite3
import pandas as pd
import os

# Final database location
DB_PATH = "data/chembl/chembl_35.db"

# Connect to ChEMBL database
conn = sqlite3.connect(DB_PATH)

# SQL query: Join molecule name + molecular properties
query = """
SELECT
    md.pref_name AS drug_name,
    cp.mw_freebase AS molecular_weight,
    cp.alogp AS logP,
    cp.cx_most_bpka AS pKa
FROM
    compound_properties cp
JOIN
    molecule_dictionary md ON cp.molregno = md.molregno
WHERE
    md.pref_name IS NOT NULL
    AND cp.mw_freebase IS NOT NULL
    AND cp.alogp IS NOT NULL
    AND cp.cx_most_bpka IS NOT NULL
LIMIT 100;
"""

# Run and load into DataFrame
df = pd.read_sql_query(query, conn)
conn.close()

# Add placeholder fields for ML integration
df["bioavailability"] = 0.8  # Placeholder
df["strength_mg_per_unit"] = 200  # Default tablet size
df["formulation_concentration"] = 40  # mg/mL for liquids

# Save to CSV
output_path = "data/raw/chembl_drug_database.csv"
df.to_csv(output_path, index=False)

print(f"✅ Extracted {len(df)} entries to {output_path}")
