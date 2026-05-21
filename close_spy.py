import sqlite3
conn = sqlite3.connect('scanner_data_live.db')
conn.execute("UPDATE positions SET status='CLOSED', exit_reason='MANUAL_CLOSE', exit_price=0 WHERE id=53 AND status='OPEN'")
conn.commit()
print('Done')
conn.close()
