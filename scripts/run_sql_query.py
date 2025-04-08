import sqlite3
import pandas as pd
from pathlib import Path

# Path to your ChEMBL database
db_path = Path("data/chembl/chembl_35.db")

if not db_path.exists():
    raise FileNotFoundError(f"❌ Database file not found: {db_path}")

# Connect to the database
conn = sqlite3.connect(db_path)

# Check available tables
print("\n📦 Available Tables:")
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print(tables)

# Run the drug-condition query
query = """
SELECT md.pref_name AS drug_name, di.efo_term AS condition
FROM drug_indication di
JOIN molecule_dictionary md ON di.molregno = md.molregno
WHERE di.efo_term IS NOT NULL;
"""

try:
    df = pd.read_sql_query(query, conn)
    print(f"\n✅ Retrieved {len(df)} rows from drug_indication + molecule_dictionary")
    
    output_path = Path("data/processed/drug_indications.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"📁 Saved to {output_path}")

except Exception as e:
    print(f"❌ Error running SQL query: {e}")

conn.close()
