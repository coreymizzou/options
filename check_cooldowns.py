import sqlite3
conn = sqlite3.connect('scanner_data_live.db')
rows = conn.execute("SELECT * FROM cooldowns").fetchall()
print(conn.execute("PRAGMA table_info(cooldowns)").fetchall())
for r in rows:
    print(r)
conn.close()
