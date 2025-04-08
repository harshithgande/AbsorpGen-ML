import sqlite3

db_path = "data/chembl/chembl_35.db"
tables_to_check = ["compound_properties", "molecule_dictionary", "products"]

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

for table in tables_to_check:
    print(f"\n📑 Columns in '{table}':")
    try:
        cursor.execute(f"PRAGMA table_info({table});")
        for col in cursor.fetchall():
            print(f" - {col[1]} ({col[2]})")
    except Exception as e:
        print(f"  ❌ Error reading {table}:", e)

conn.close()
input("\nPress Enter to finish...")
