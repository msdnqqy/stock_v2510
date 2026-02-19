
import pymysql
from config.config import DB_CONFIG

try:
    # Try connecting with the configured password (which is 'your_password' currently)
    # This will likely fail, but let's see.
    print(f"Attempting to connect with password: {DB_CONFIG['password']}")
    conn = pymysql.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    print("Connection successful!")
    conn.close()
except pymysql.err.OperationalError as e:
    print(f"Connection failed: {e}")
    
    # Try with empty password
    if DB_CONFIG['password'] != '':
        try:
            print("Attempting to connect with empty password...")
            conn = pymysql.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                user=DB_CONFIG['user'],
                password=''
            )
            print("Connection successful with empty password!")
            conn.close()
        except pymysql.err.OperationalError as e:
            print(f"Connection failed with empty password: {e}")

