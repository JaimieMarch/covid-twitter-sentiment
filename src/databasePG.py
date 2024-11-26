import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

# Load environment variables
load_dotenv()

# Database connection URL from .env
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("DB_URL not found. Ensure it's set in the .env file.")

# Create database engine
engine = create_engine(DB_URL)

# Function to list all tables in the database
def list_all_tables():
    print("Listing all tables in the database...")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if tables:
        print("Tables in the database:")
        for table in tables:
            print(f"- {table}")
    else:
        print("No tables found in the database.")

# Entry point
if __name__ == "__main__":
    list_all_tables()






