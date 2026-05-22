import sqlite3, json
conn = sqlite3.connect('scanner_data_live.db')
conn.execute("DELETE FROM system_state WHERE key = 'action_state'")
conn.commit()
print('Cleared action state')
conn.close()
