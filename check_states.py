import sqlite3
conn = sqlite3.connect('scanner_data_live.db')
rows = conn.execute("SELECT key FROM system_state WHERE key LIKE '%entry%' OR key LIKE '%action%' OR key LIKE '%CRWD%' OR key LIKE '%NVDA%'").fetchall()
for r in rows:
    print(r)
conn.close()
