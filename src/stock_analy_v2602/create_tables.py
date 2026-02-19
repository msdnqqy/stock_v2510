
import sys
import logging
# Ensure the current directory is in the path
sys.path.append('.')

from database.db_manager import DatabaseManager
from config.config import DB_CONFIG

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

print(f"Connecting to database with user: {DB_CONFIG['user']}")

try:
    db_manager = DatabaseManager()
    print("Database connection established.")
    
    print("Creating tables...")
    db_manager.create_tables()
    print("Tables created successfully.")
    
except Exception as e:
    print(f"Error occurred: {e}")
    sys.exit(1)
