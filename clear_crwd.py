import sqlite3
conn = sqlite3.connect('scanner_data_live.db')
conn.execute("DELETE FROM system_state WHERE key LIKE 'action_entry_CRWD%'")
conn.commit()
print('Cleared CRWD action state')
conn.close()
