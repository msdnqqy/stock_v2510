
import pymysql
from config.config import DB_CONFIG

def test_connection(host, user, password, port):
    print(f"Testing connection to {host}:{port} as {user}...")
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        print(f"SUCCESS: Connected to {host}!")
        conn.close()
        return True
    except pymysql.err.OperationalError as e:
        print(f"FAILURE: Could not connect to {host}. Error: {e}")
        return False

# Test 1: localhost with configured password
test_connection('localhost', DB_CONFIG['user'], DB_CONFIG['password'], DB_CONFIG['port'])

# Test 2: 127.0.0.1 with configured password
test_connection('127.0.0.1', DB_CONFIG['user'], DB_CONFIG['password'], DB_CONFIG['port'])

# Test 3: localhost with empty password (just in case)
if DB_CONFIG['password'] != '':
    test_connection('localhost', DB_CONFIG['user'], '', DB_CONFIG['port'])
