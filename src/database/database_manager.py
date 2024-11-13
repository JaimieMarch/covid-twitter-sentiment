import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, Integer, Float, String, MetaData, Text, DateTime, Boolean, inspect
from alembic import command
from alembic.config import Config
import os
from dotenv import load_dotenv
import subprocess
load_dotenv()


"""
Global Vars
"""
db_url = os.getenv("DB_URL")
engine = create_engine(db_url)
metadata = MetaData() 
__all__ = ['metadata']


"""
Infers schema based on pandas data types
source: https://medium.com/@anusoosanbaby/efficiently-importing-csv-data-into-postgresql-using-python-and-sqlalchemy-052693aa921a
"""
def infer_sqllchemy_type(dtype):
   """ Map pandas dtype to SQLAlchemy's types """
   if "int" in dtype.name:
         return Integer
   elif "float" in dtype.name:
         return Float 
   elif "datetime" in dtype.name:
      return DateTime
   elif "bool" in dtype.name:
      return Boolean
   elif "object" in dtype.name:
      return Text
   else:
      return Text

"""
Creates a table from a csv

Args:
   table_name (str): name of the table
   file_path (str): path to the csv
   replace (bool): determines if the table should be replaced if duplicate table is found
   dataframe (str): pandas dataframe
"""
def create_table(table_name, file_path=None, dataframe=None, replace=False):
    if not table_name:
        raise ValueError("Error: 'table_name' parameter is required.")
    if file_path:
        df = pd.read_csv(file_path)
    elif dataframe is not None and not dataframe.empty: 
        df = dataframe
    else:
        raise ValueError("Error: Either 'file_path' or 'dataframe' must be provided.")
    

   # print(df.head())
   
    inspector = inspect(engine)
    table_exists = inspector.has_table(table_name)

    # check if table exists
    if table_exists and replace:
        print(f"The table '{table_name}' already exists. Replacing entries.")
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, method="multi", chunksize=50000)
    elif table_exists and not replace:
        print(f"The table '{table_name}' already exists. Use 'replace=True' to overwrite.")
        return
    else:
        # infer the column types
        columns = [Column(name, infer_sqllchemy_type(dtype)) for name, dtype in df.dtypes.items()]
        table = Table(table_name, metadata, *columns, extend_existing=True)

        # create the table
        table.create(engine)

        print(f"Creating {table_name} table\n")

        # update the entries with dataframe
        df.to_sql(table_name, con=engine, if_exists='append', index=False, method="multi", chunksize=50000)

#    tuples = [(name, dtype.name, type(df[name].dropna().iloc[0]).__name__) for name, dtype in df.dtypes.items()]
#    print(tuples)

   # confirm table creation
   # query = f"SELECT * FROM {table_name}"
   # tabledata = pd.read_sql_query(query, engine)
   # print(f"\n{table_name} table created")
   

"""
Deletes the specified table
"""
def delete_table(table_name):
   if not table_name:
        raise ValueError("Error: 'table_name' parameter is required.")

   query = f"\nDROP TABLE IF EXISTS \"{table_name}\""
   with engine.connect() as conn:
      conn.execute(text(query))
      conn.commit()
      print(f"{table_name} table dropped")

"""
Runs a query on the db
"""
import pandas as pd
from sqlalchemy import create_engine, text

def query_db(query):
    if not query:
        raise ValueError("Error: 'query' parameter is required.")
    query = text(query)
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


"""
Inserts a record in the db
"""
def insert_record(table_name, record):
    if not table_name:
        raise ValueError("Error: 'table_name' parameter is required.")
    if not record:
        raise ValueError("Error: 'record' parameter is required.")
    df = pd.DataFrame([record])
    df.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f"Record inserted into {table_name}")

"""
Updates a record in the db
"""
def update_record(table_name, record, condition):
    if not table_name:
        raise ValueError("Error: 'table_name' parameter is required.")
    if not record:
        raise ValueError("Error: 'record' parameter is required.")
    if not condition:
        raise ValueError("Error: 'condition' parameter is required.")
    set_clause = ", ".join([f"{key} = :{key}" for key in record.keys()])
    query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
    with engine.connect() as conn:
        conn.execute(text(query), **record)
        conn.commit()
    print(f"Record updated in {table_name}")

"""
Deletes a record in the db
"""
def delete_record(table_name, condition):
    if not table_name:
        raise ValueError("Error: 'table_name' parameter is required.")
    if not condition:
        raise ValueError("Error: 'condition' parameter is required.")
    query = f"DELETE FROM {table_name} WHERE {condition}"
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()
    print(f"Record deleted from {table_name}")


def list_tables():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    with open("../data/table_list.py", "w") as file:
        for table in tables:
            file.write(f"{table.upper()}={table}\n")
    print("Database table list updated.")


# sql_query=f""" SELECT * FROM "clean_twitter_data" """ 
# delete_table("clean_twitter_data") 
# create_table("clean_twitter_data", file_path="../../data/processed/cleanCovidTwitterData.zip") 
# # create_table("raw_twitter_data", file_path="../../data/raw/covidTwitterData.zip") 
# res = query_db(sql_query) 
# print(res.head())
# table = "processed_twitter_data"
# query = """
#     SELECT * FROM "{table}"
# """
# query_db(query)


"""
Other write methods
"""
# with open(file_path, 'r') as f:    
      #    cursor = engine.cursor()
      #    query = f"""
      #          COPY {table_name} ( 
      #          user_name, user_location, user_description, user_created, 
      #          user_followers, user_friends, user_favourites, user_verified, date, text, hashtags, source, is_retweet) 
      #          FROM stdin WITH CSV HEADER
      #    """
      #    cursor.copy_expert(query, f)
      #    engine.commit()

      # with engine.connect() as conn: 
      #    with conn.begin():
      #       with conn.connection.cursor() as cursor:
      #          with open(file_path, "r") as f: 
      #             cursor.copy_expert( 
      #                f""" COPY {table_name} ( 
      #                   user_name, user_location, user_description, user_created, 
      #                   user_followers, user_friends, user_favourites, user_verified, date, text, hashtags, source, is_retweet) 
      #                FROM stdin WITH CSV HEADER """, f)